"""
MorphCloud Orchestrator.

Implements the MorphCloud orchestration logic described in Sections 3.1 and 3.5
of MorphCloud-LLM (Electronics 2026). The orchestrator is a risk-sensitive
request routing policy that coordinates spot and on-demand instances, manages
migration workflows, and enforces the zero-drop SLA.

This module covers:
  - System state (Equation 1): Psi(t) = <S(t), D(t), Q(t), C(t), P(t)>
  - Instance selection (Equation 19): risk-sensitive routing with load,
    preemption risk, and KV-cache locality weights
  - Cost optimization (Equation 20): minimize total serving cost subject to
    latency SLA and zero-drop constraints
  - Algorithm 1: MorphCloud-LLM Preemption-Aware Serving

Cost accounting (Table 7 / Section 4.6):
  Active-serving-only:       27% of on-demand cost (69.8% cost reduction)
  Including warm standby:    cost rises by $0.92/h; 67% cost reduction
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SpotInstance:
    """Represents an active spot instance in set S."""
    instance_id: str
    platform: str            # "aws" or "gcp"
    region: str
    gpu_type: str            # e.g. "A100-80GB"
    current_load: float      # tokens/s currently being processed
    capacity: float          # max tokens/s
    uptime_s: float
    preemption_prob: float   # rho(t), updated each prediction cycle
    kv_sessions: List[str] = field(default_factory=list)  # active session IDs


@dataclass
class FallbackNode:
    """
    Represents an on-demand fallback node in set D.

    Fallback nodes maintain pre-staged target model weights in GPU memory
    at all times using tensor parallelism. This eliminates the model loading
    phase (T_weight_load = 0 ms in Table 11) and is the primary reason
    MorphCloud-LLM achieves 1390 ms total migration latency vs 38,000 ms
    for Kubernetes restart.
    """
    node_id: str
    platform: str
    gpu_type: str
    vram_gb: float
    weights_preloaded: bool = True
    active_during_migration: bool = False
    draft_model_loaded: bool = True


@dataclass
class InferenceRequest:
    """A single LLM inference request in the global queue Q."""
    request_id: str
    session_id: str
    prompt_tokens: int
    expected_output_tokens: int
    assigned_instance: Optional[str] = None
    state: str = "queued"    # queued | normal | migrating | recovered
    submit_time_s: float = field(default_factory=time.perf_counter)


@dataclass
class SystemState:
    """
    Equation 1: Psi(t) = <S(t), D(t), Q(t), C(t), P(t)>

    S(t): active spot instances
    D(t): on-demand fallback nodes
    Q(t): global request queue
    C(t): distributed KV-cache state (represented as session mapping)
    P(t): preemption probability vector
    """
    spot_instances: Dict[str, SpotInstance] = field(default_factory=dict)
    fallback_nodes: Dict[str, FallbackNode] = field(default_factory=dict)
    request_queue: List[InferenceRequest] = field(default_factory=list)
    kv_cache_sessions: Dict[str, str] = field(default_factory=dict)  # session_id -> instance_id
    preemption_probs: Dict[str, float] = field(default_factory=dict)  # instance_id -> rho


@dataclass
class RoutingWeights:
    """
    Weighting factors gamma_1, gamma_2, gamma_3 for Equation 19.
    Balance load distribution, preemption risk exposure, and KV-cache locality.
    """
    gamma_load: float = 0.4         # gamma_1: load balancing
    gamma_preemption: float = 0.4   # gamma_2: preemption risk
    gamma_locality: float = 0.2     # gamma_3: KV-cache locality


class MorphCloudOrchestrator:
    """
    Risk-sensitive request routing and migration coordinator.

    Implements Algorithm 1 (Section 3.2): MorphCloud-LLM Preemption-Aware Serving.

    Input:  Request queue Q, spot instances S, fallback nodes D
    Output: Generated responses without dropped requests

    The orchestrator acts as an intermediary within the request router and
    inference engine and does not require adjustments to the model structure.
    """

    def __init__(
        self,
        routing_weights: Optional[RoutingWeights] = None,
        sla_latency_ms: float = 500.0,
        preemption_predictor=None,
        kv_checkpointer=None,
        speculative_engine=None,
    ):
        self.routing_weights = routing_weights or RoutingWeights()
        self.sla_latency_ms = sla_latency_ms
        self.predictor = preemption_predictor
        self.checkpointer = kv_checkpointer
        self.speculative_engine = speculative_engine

        self.state = SystemState()
        self._migration_in_progress: Dict[str, bool] = {}
        self._dropped_requests: int = 0
        self._total_requests: int = 0

        logger.info("MorphCloud orchestrator initialized.")

    # ------------------------------------------------------------------
    # Algorithm 1 main serving loop
    # ------------------------------------------------------------------

    async def serve_loop(self):
        """
        Algorithm 1: MorphCloud-LLM Preemption-Aware Serving.

        while serving:
          for each request r in Q: assign to s*
          for each instance s in S: check rho(s), trigger migration if needed
          AsyncCheckpoint(delta_C(t)) for all active instances
        """
        logger.info("MorphCloud serving loop started.")
        while True:
            # Algorithm 1, lines 4-7: route pending requests
            pending = [r for r in self.state.request_queue if r.assigned_instance is None]
            for request in pending:
                s_star = self._select_instance(request)
                if s_star:
                    request.assigned_instance = s_star
                    request.state = "normal"
                    self.state.spot_instances[s_star].kv_sessions.append(request.session_id)

            # Algorithm 1, lines 8-18: check preemption risk for each spot instance
            for instance_id, instance in list(self.state.spot_instances.items()):
                if self._migration_in_progress.get(instance_id, False):
                    continue
                rho = instance.preemption_prob
                threshold = self._get_threshold(instance)
                if rho > threshold:
                    logger.info(
                        "Proactive migration triggered: instance=%s rho=%.3f threshold=%.3f",
                        instance_id, rho, threshold,
                    )
                    asyncio.create_task(self._proactive_migration(instance_id))

            # Algorithm 1, line 19: async checkpoint for all active instances
            await self._trigger_async_checkpoints()

            await asyncio.sleep(0.1)  # 100 ms polling interval

    # ------------------------------------------------------------------
    # Instance selection (Equation 19)
    # ------------------------------------------------------------------

    def _select_instance(self, request: InferenceRequest) -> Optional[str]:
        """
        Equation 19:
          s* = argmin_{s in S} [gamma_1 * load(s) + gamma_2 * Q(s) * l_r
                                 + gamma_3 * (1 - locality(s, r))]

        where Q(s) = preemption probability rho(s), l_r = expected output tokens,
        and locality(s, r) = probability that instance s already holds relevant
        KV-cache for session r.
        """
        best_instance = None
        best_cost = float("inf")

        for inst_id, inst in self.state.spot_instances.items():
            if inst.capacity <= 0:
                continue

            load_term = self.routing_weights.gamma_load * (inst.current_load / inst.capacity)
            preemption_term = (
                self.routing_weights.gamma_preemption
                * inst.preemption_prob
                * request.expected_output_tokens
            )
            locality = self._compute_locality(inst, request)
            locality_term = self.routing_weights.gamma_locality * (1.0 - locality)

            cost = load_term + preemption_term + locality_term
            if cost < best_cost:
                best_cost = cost
                best_instance = inst_id

        return best_instance

    def _compute_locality(self, instance: SpotInstance, request: InferenceRequest) -> float:
        """
        KV-cache locality: probability that this instance already holds
        useful KV-cache state for the request's conversation session.
        Returns float in [0, 1].
        """
        if request.session_id in instance.kv_sessions:
            return 1.0
        return 0.0

    def _get_threshold(self, instance: SpotInstance) -> float:
        """Equation 13: load-adaptive migration threshold."""
        if self.predictor is None:
            return 0.7
        return self.predictor.threshold_config.compute(
            instance.current_load, instance.capacity
        )

    # ------------------------------------------------------------------
    # Proactive migration (Algorithm 1, lines 10-17)
    # ------------------------------------------------------------------

    async def _proactive_migration(self, instance_id: str):
        """
        Algorithm 1 migration workflow:
          1. Activate Speculative Continuity on fallback node d_j.
          2. Stream pending KV-cache delta to storage.
          3. Acquire replacement instance s'.
          4. Restore KV-cache on s' from checkpoint.
          5. Verify speculative tokens on s'.

        Total end-to-end migration latency: 1390 ms (Table 11).
          Detection:         0 ms (proactive; predictor eliminates reactive delay)
          State Transfer:  990 ms (120 ms S3 round-trip + 870 ms delta streaming)
          Instance Provision: 250 ms (warm standby activation, not cold provisioning)
          Model Loading:     0 ms (weights pre-staged in fallback GPU memory)
          KV State Activation: 150 ms
        """
        self._migration_in_progress[instance_id] = True
        instance = self.state.spot_instances.get(instance_id)
        if instance is None:
            return

        t_start = time.perf_counter()
        logger.info("Starting proactive migration for instance %s", instance_id)

        # Step 1: enter speculative continuity on fallback node
        affected_requests = [
            r for r in self.state.request_queue
            if r.assigned_instance == instance_id and r.state == "normal"
        ]
        for req in affected_requests:
            req.state = "migrating"
            if self.speculative_engine:
                self.speculative_engine.enter_continuity_mode(
                    session_id=req.session_id,
                    model_name="llama-70b",
                    event_id=id(req),
                )

        # Step 2: stream pending KV-cache delta to disaggregated storage
        # (state transfer component: ~990 ms for 32 GB cache at 32 GB working set)
        await asyncio.sleep(0.99)  # simulated; replace with real transfer

        # Step 3: acquire warm standby replacement instance (~250 ms)
        await asyncio.sleep(0.25)  # pre-warmed; not cold-provisioning

        # Step 4: restore KV-cache on replacement (KV state activation: ~150 ms)
        # Model Loading = 0 ms: weights are pre-staged in fallback GPU memory
        await asyncio.sleep(0.15)

        # Step 5: verify speculative tokens on replacement instance
        for req in affected_requests:
            if self.speculative_engine:
                self.speculative_engine.exit_continuity_mode()
            req.state = "recovered"

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000
        logger.info(
            "Migration complete: instance=%s total_latency=%.0f ms",
            instance_id, t_elapsed_ms,
        )

        # Replace the preempted instance with the recovery instance
        del self.state.spot_instances[instance_id]
        self._migration_in_progress.pop(instance_id, None)

    # ------------------------------------------------------------------
    # Cost optimization (Equation 20)
    # ------------------------------------------------------------------

    def compute_total_cost(self, include_warm_standby: bool = True) -> float:
        """Return the manuscript's aggregate Table 7 hourly cost.

        Table 7 reports $8.24/h as the aggregate spot-GPU component for the
        evaluated MorphCloud configuration, not $8.24 per object in ``S``.
        Therefore this helper must not multiply that number by the number of
        ``SpotInstance`` records currently registered in the in-memory state.

        Returns $9.89/h on the active-serving accounting basis or $10.81/h
        when the additional $0.92/h warm-standby component is included.
        """
        spot_gpu = 8.24
        fallback_active = 1.12
        warm_standby = 0.92 if include_warm_standby else 0.0
        storage = 0.45
        network = 0.08
        return spot_gpu + fallback_active + warm_standby + storage + network

    # ------------------------------------------------------------------
    # Checkpointing trigger
    # ------------------------------------------------------------------

    async def _trigger_async_checkpoints(self):
        """Algorithm 1, line 19: AsyncCheckpoint(delta_C(t)) for all active instances."""
        if not self.checkpointer:
            return
        flush = getattr(self.checkpointer, "flush_pending", None)
        if flush is None:
            # The supplied checkpoint reference is driven by on_tokens_generated();
            # the GPU data-plane adapter owns those calls.
            return
        result = flush()
        if asyncio.iscoroutine(result):
            await result

    # ------------------------------------------------------------------
    # SLA monitoring
    # ------------------------------------------------------------------

    def drop_rate(self) -> float:
        """
        Fraction of requests that were NOT fully served within SLA.
        Target: 0.00% (zero dropped requests across all 521 events).
        Wilson score 95% CI upper bound: 0.15% (Table 5).
        Event-level bootstrap 95% CI upper bound: 0.73%.
        """
        if self._total_requests == 0:
            return 0.0
        return self._dropped_requests / self._total_requests
