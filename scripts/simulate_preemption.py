"""Manuscript-parameterized synthetic preemption simulator.

The supplied archive does not contain the empirical 90-day trace/CDF artifacts
used by the paper's live-cloud trace-injection experiment. This utility therefore
uses the manuscript-reported counts, notice windows, platform/model splits, and
mean inter-event intervals with a clearly labeled parametric approximation.
It is useful for exercising analysis code, not for replacing the original data.

Canonical parameters from Section 4.1.2:
  - 521 events: 274 AWS + 247 GCP.
  - Severity counts: 312 single, 156 concurrent dual, 53 full unavailability.
  - Mean inter-event interval: 47.3 min AWS, 38.9 min GCP.
  - AWS notice window: 120 s; GCP notice window: 30 s.

Usage:
    python scripts/simulate_preemption.py \
        --n-events 521 \
        --n-runs 1 \
        --platforms aws gcp \
        --output-dir evaluation/results/synthetic-simulation
"""

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("preemption_sim")


class PreemptionSeverity(Enum):
    SINGLE_INSTANCE = "single"          # n=312 in paper
    CONCURRENT_DUAL = "concurrent_dual" # n=156 in paper
    FULL_SPOT_UNAVAILABLE = "full"      # n=53 in paper


@dataclass
class PreemptionEvent:
    """A synthetic event parameterized from manuscript-reported statistics."""
    event_id: int
    platform: str                      # "aws" or "gcp"
    region: str
    model: str                         # "llama-70b" or "mixtral-8x7b"
    severity: PreemptionSeverity
    scheduled_time_s: float            # seconds from start of evaluation window
    notice_window_s: float             # 120 s (AWS) or 30 s (GCP)
    affected_instances: List[str]      # instance IDs to preempt
    inter_arrival_s: float             # synthetic interval calibrated to the reported platform mean


@dataclass
class EventOutcome:
    """Outcome of a single injected preemption event."""
    event_id: int
    severity: str
    recovery_time_ms: float
    drop_rate: float
    kv_delta_bytes_transferred: int
    n_speculative_tokens: int
    acceptance_rate: float
    p99_latency_ms: float
    predictor_triggered: bool          # True if ML predictor caught it; False if native notify
    migration_phase_breakdown: Dict[str, float] = field(default_factory=dict)


class ManuscriptParameterizedInterArrivalDistribution:
    """Synthetic Weibull approximation calibrated to manuscript-reported means.

    The original empirical CDFs are not included in the supplied artifact.
    """

    TARGET_MEAN_S = {"aws": 47.3 * 60.0, "gcp": 38.9 * 60.0}
    SHAPE = {"aws": 1.2, "gcp": 1.1}

    def sample(self, platform: str, rng: np.random.Generator) -> float:
        shape = self.SHAPE.get(platform, self.SHAPE["aws"])
        target_mean = self.TARGET_MEAN_S.get(platform, self.TARGET_MEAN_S["aws"])
        scale = target_mean / math.gamma(1.0 + 1.0 / shape)
        return max(float(rng.weibull(shape) * scale), 60.0)

    def expected_mean_s(self, platform: str) -> float:
        return self.TARGET_MEAN_S.get(platform, self.TARGET_MEAN_S["aws"])


class PreemptionSimulator:
    """
    Manuscript-parameterized synthetic preemption event simulator.

    Exercises the event counts, platform/model split, notice windows, and mean
    intervals reported in Section 4.1.2. The empirical CDFs are not present in
    this archive, so this class uses a documented parametric approximation.
    Its outputs are synthetic and must not be presented as original measurements.
    """

    # Distribution of preemption severities (Section 4.1.2)
    SEVERITY_DISTRIBUTION = {
        PreemptionSeverity.SINGLE_INSTANCE:   (312, 0.599),
        PreemptionSeverity.CONCURRENT_DUAL:   (156, 0.300),
        PreemptionSeverity.FULL_SPOT_UNAVAILABLE: (53, 0.101),
    }

    # Per-platform event distribution
    PLATFORM_EVENTS = {
        "aws": {
            "total": 274,
            "llama-70b": 147,
            "mixtral-8x7b": 127,
            "regions": ["us-east-1", "us-west-2"],
            "notice_window_s": 120.0,   # AWS 2-min spot interruption notification
        },
        "gcp": {
            "total": 247,
            "llama-70b": 132,
            "mixtral-8x7b": 115,
            "regions": ["us-central1", "europe-west1"],
            "notice_window_s": 30.0,    # GCP 30-s shutdown signal
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.inter_arrival_dist = ManuscriptParameterizedInterArrivalDistribution()
        self._events: List[PreemptionEvent] = []

    def generate_event_schedule(
        self,
        n_total: int = 521,
        platforms: Optional[List[str]] = None,
    ) -> List[PreemptionEvent]:
        """Generate up to the canonical 521-event manuscript schedule.

        Each platform has its own event clock, matching the paper's separately
        reported AWS/GCP mean inter-event intervals. For n_total < 521, events
        are truncated after global time ordering. Values above the canonical
        event count are rejected rather than invented.
        """
        selected = platforms or list(self.PLATFORM_EVENTS)
        invalid = [p for p in selected if p not in self.PLATFORM_EVENTS]
        if invalid:
            raise ValueError(f"Unsupported platform(s): {invalid}")
        canonical_total = sum(self.PLATFORM_EVENTS[p]["total"] for p in selected)
        if n_total <= 0:
            raise ValueError("n_total must be positive")
        if n_total > canonical_total:
            raise ValueError(
                f"Requested {n_total} events, but only {canonical_total} canonical "
                f"events are defined for platforms {selected}."
            )

        events: List[PreemptionEvent] = []
        event_id = 0

        # For the full two-platform 521-event schedule, preserve the manuscript's
        # exact severity counts (312/156/53) instead of merely sampling their
        # probabilities. For subsets, sampling is used because no per-platform
        # severity allocation is reported in the manuscript.
        severity_pool: List[PreemptionSeverity] = []
        if set(selected) == {"aws", "gcp"} and canonical_total == 521:
            for severity, (count, _prob) in self.SEVERITY_DISTRIBUTION.items():
                severity_pool.extend([severity] * count)
            self.rng.shuffle(severity_pool)

        for platform in selected:
            cfg = self.PLATFORM_EVENTS[platform]
            platform_count = cfg["total"]
            intervals = np.array(
                [self.inter_arrival_dist.sample(platform, self.rng) for _ in range(platform_count)],
                dtype=float,
            )
            # Calibrate the finite canonical schedule to the exact mean reported
            # in the manuscript while retaining the sampled relative variation.
            target_mean = self.inter_arrival_dist.expected_mean_s(platform)
            intervals *= target_mean / float(intervals.mean())

            platform_time_s = 0.0
            interval_idx = 0
            model_counts = [("llama-70b", cfg["llama-70b"]), ("mixtral-8x7b", cfg["mixtral-8x7b"])]
            for model, n_model in model_counts:
                for _ in range(n_model):
                    if severity_pool:
                        severity = severity_pool.pop()
                    else:
                        severities = list(self.SEVERITY_DISTRIBUTION.keys())
                        probs = [v[1] for v in self.SEVERITY_DISTRIBUTION.values()]
                        severity = self.rng.choice(severities, p=probs)
                    inter_arrival = float(intervals[interval_idx])
                    interval_idx += 1
                    platform_time_s += inter_arrival
                    region = str(self.rng.choice(cfg["regions"]))
                    n_affected = 1 if severity == PreemptionSeverity.SINGLE_INSTANCE else (
                        2 if severity == PreemptionSeverity.CONCURRENT_DUAL else 8
                    )
                    affected = [f"{platform}-{region}-{model}-inst-{i}" for i in range(n_affected)]
                    events.append(PreemptionEvent(
                        event_id=event_id, platform=platform, region=region, model=model,
                        severity=severity, scheduled_time_s=platform_time_s,
                        notice_window_s=cfg["notice_window_s"], affected_instances=affected,
                        inter_arrival_s=inter_arrival,
                    ))
                    event_id += 1

        events.sort(key=lambda e: (e.scheduled_time_s, e.platform, e.event_id))
        self._events = events[:n_total]
        # Reassign IDs after sorting/truncation to keep output deterministic and contiguous.
        for idx, event in enumerate(self._events):
            event.event_id = idx
        logger.info("Generated %d synthetic events.", len(self._events))
        return self._events

    def inject_event(self, event: PreemptionEvent) -> EventOutcome:
        """
        Inject a single preemption event into the serving cluster and
        collect recovery outcome metrics.

        For AWS events: simulates the 2-min spot interruption notification.
        For GCP events: simulates the 30-s shutdown signal.

        Returns a synthetic EventOutcome with fields that can be compared with
        Tables 8, 9, 10, 11, and 13.
        """
        logger.info(
            "Injecting event %d: platform=%s model=%s severity=%s",
            event.event_id, event.platform, event.model, event.severity.value,
        )

        # Simulate MorphCloud-LLM recovery performance from Table 11
        outcome = self._simulate_morphcloud_recovery(event)
        return outcome

    def _simulate_morphcloud_recovery(self, event: PreemptionEvent) -> EventOutcome:
        """
        Simulate MorphCloud-LLM recovery for a given preemption event.
        Values drawn from the distributions reported in the paper:
          - Table 11 migration breakdown
          - Table 10 per-model-platform performance
          - Section 4.10 speculative decoding quality
        """
        model = event.model

        # Recovery time parameters from Table 8 (16 GB working set, Table 10)
        recovery_means = {
            ("llama-70b", "aws"): 580.0,
            ("llama-70b", "gcp"): 650.0,
            ("mixtral-8x7b", "aws"): 420.0,
            ("mixtral-8x7b", "gcp"): 480.0,
        }
        recovery_mean_ms = recovery_means.get((model, event.platform), 580.0)
        recovery_time_ms = float(self.rng.normal(recovery_mean_ms, 30.0))

        # Speculative decoding parameters (Section 4.10)
        spec_params = {
            "llama-70b": {"mean_k": 41.2, "sd_k": 8.7, "acceptance_rate": 0.873},
            "mixtral-8x7b": {"mean_k": 38.7, "sd_k": 6.4, "acceptance_rate": 0.891},
        }
        sp = spec_params.get(model, spec_params["llama-70b"])
        k = max(1, int(self.rng.normal(sp["mean_k"], sp["sd_k"])))
        acceptance_rate = float(self.rng.normal(sp["acceptance_rate"], 0.02))
        acceptance_rate = min(max(acceptance_rate, 0.5), 1.0)

        # KV delta bytes for 16 GB working set (Table 8 interpolated)
        # 16 GB -> approximately 688 MB delta (4.3% ratio)
        delta_bytes = int(self.rng.normal(688 * 1024 * 1024, 20 * 1024 * 1024))

        # p99 latency (Table 10)
        p99_means = {
            ("llama-70b", "aws"): 49.0,
            ("llama-70b", "gcp"): 52.0,
            ("mixtral-8x7b", "aws"): 43.0,
            ("mixtral-8x7b", "gcp"): 47.0,
        }
        p99_ms = float(self.rng.normal(p99_means.get((model, event.platform), 49.0), 2.0))

        # Native-Notify covers 82% of events; predictor covers the remaining 18%
        # All 521 events yield zero drops (Table 5)
        predictor_triggered = self.rng.random() < 0.18

        # Table 11 migration phase breakdown (ms)
        breakdown = {
            "detection_ms": 0.0,          # 0 ms: proactive prediction eliminates detection delay
            "state_transfer_ms": float(self.rng.normal(990.0, 30.0)),
            "s3_round_trip_ms": float(self.rng.normal(120.0, 10.0)),
            "kv_delta_streaming_ms": float(self.rng.normal(870.0, 25.0)),
            "instance_provision_ms": float(self.rng.normal(250.0, 15.0)),
            "model_loading_ms": 0.0,      # 0 ms: weights pre-staged in fallback GPU memory
            "kv_state_activation_ms": float(self.rng.normal(150.0, 10.0)),
        }
        breakdown["total_ms"] = (
            breakdown["detection_ms"]
            + breakdown["state_transfer_ms"]
            + breakdown["instance_provision_ms"]
            + breakdown["model_loading_ms"]
            + breakdown["kv_state_activation_ms"]
        )

        return EventOutcome(
            event_id=event.event_id,
            severity=event.severity.value,
            recovery_time_ms=recovery_time_ms,
            drop_rate=0.0,
            kv_delta_bytes_transferred=delta_bytes,
            n_speculative_tokens=k,
            acceptance_rate=acceptance_rate,
            p99_latency_ms=p99_ms,
            predictor_triggered=predictor_triggered,
            migration_phase_breakdown=breakdown,
        )

    def run_full_evaluation(
        self, n_events: int = 521, n_runs: int = 5,
        output_dir: str = "evaluation/results/synthetic-simulation",
        platforms: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run the manuscript-parameterized synthetic harness.

        This does not reproduce the original live-cloud experiment because the
        empirical traces and GPU data plane are not present in this archive.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        events = self.generate_event_schedule(n_events, platforms=platforms)

        all_outcomes: List[EventOutcome] = []
        for run_idx in range(n_runs):
            logger.info("Run %d / %d", run_idx + 1, n_runs)
            for event in events:
                outcome = self.inject_event(event)
                all_outcomes.append(outcome)

        # Aggregate metrics
        drop_rates = [o.drop_rate for o in all_outcomes]
        recovery_times = [o.recovery_time_ms for o in all_outcomes]
        p99_latencies = [o.p99_latency_ms for o in all_outcomes]
        acceptance_rates = [o.acceptance_rate for o in all_outcomes]

        summary = {
            "artifact_type": "synthetic_manuscript_parameterized_simulation",
            "n_events": len(events),
            "n_runs": n_runs,
            "total_opportunities": len(all_outcomes),
            "mean_drop_rate": float(np.mean(drop_rates)),
            "max_drop_rate": float(np.max(drop_rates)),
            "mean_recovery_ms": float(np.mean(recovery_times)),
            "sd_recovery_ms": float(np.std(recovery_times)),
            "mean_p99_latency_ms": float(np.mean(p99_latencies)),
            "mean_acceptance_rate": float(np.mean(acceptance_rates)),
        }

        out_path = os.path.join(output_dir, "preemption_sim_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info("Evaluation summary written to %s", out_path)
        return summary


def main():
    parser = argparse.ArgumentParser(description="MorphCloud-LLM preemption simulator")
    parser.add_argument("--n-events", type=int, default=521)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--platforms", nargs="+", default=["aws", "gcp"])
    parser.add_argument("--output-dir", default="evaluation/results/synthetic-simulation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sim = PreemptionSimulator(seed=args.seed)
    summary = sim.run_full_evaluation(
        n_events=args.n_events,
        n_runs=args.n_runs,
        output_dir=args.output_dir,
        platforms=args.platforms,
    )
    print("\n=== Evaluation Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
