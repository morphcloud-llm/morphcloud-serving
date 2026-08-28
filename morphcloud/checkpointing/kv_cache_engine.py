"""
Asynchronous Incremental KV-Cache Checkpointing Engine.

Implements the streaming checkpoint mechanism described in Section 3.2 of
MorphCloud-LLM (Electronics 2026). The engine continuously synchronizes
incremental KV-cache deltas to disaggregated persistent storage (Amazon S3
Express One Zone or Google Cloud Storage regional endpoints) with less than
3% throughput overhead.

Key design points:
  - Double-buffering: inference writes to one pinned memory region while the
    other is asynchronously transferred to storage.
  - Version vector (Equation 6): monotonically increasing per-layer token
    position counters guarantee checkpoint consistency.
  - Recovery (Equation 7): restores to min version across all layers, then
    recomputes from that point to the current position.
  - Delta size: 3.8–8.2% of raw KV-cache size for caches of 2–32 GB (Table 8).
"""

import asyncio
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for the KV-cache checkpointing engine."""
    delta_interval: int = 16          # Checkpoint interval in decoding steps (delta)
    base_snapshot_epoch: int = 512    # Full base snapshot every N tokens
    backend: str = "memory"            # "memory" for dry-run; "s3"/"gcs" require an adapter
    s3_bucket: str = ""
    s3_prefix: str = "kv-cache/"
    gcs_bucket: str = ""
    gcs_prefix: str = "kv-cache/"
    num_layers: int = 80              # LLaMA-70B default
    num_kv_heads: int = 8             # GQA KV heads for LLaMA-70B
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    max_seq_len: int = 4096
    enable_double_buffer: bool = True


@dataclass
class VersionVector:
    """
    Per-layer checkpoint version vector (Equation 6, Section 3.2).

    V(t) = [v_1(t), v_2(t), ..., v_L(t)]
    where v_l(t) is the most recent token position checkpointed on layer l.
    """
    versions: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.versions:
            self.versions = [0] * 80  # default L=80

    def recovery_position(self) -> int:
        """
        Equation 7: n_recover = min(v_l(t)) for l = 1, ..., L.
        The system restores KV-cache to the lowest version across all layers.
        """
        return min(self.versions)

    def update(self, layer: int, token_position: int):
        self.versions[layer] = token_position

    def update_all(self, token_position: int):
        self.versions = [token_position] * len(self.versions)


class KVCacheMemorySize:
    """
    Computes KV-cache memory requirements per Equation 2, Section 3.2.

    M_kv = 2 * L * H_kv * d_h * n * sizeof(dtype)

    For LLaMA-70B with GQA (H_kv=8, L=80, d_h=128):
      n=2048 tokens -> ~0.67 GB per sequence
    For full-head models (H=H_kv=64, L=80, d_h=128):
      n=2048 tokens -> ~5.37 GB per sequence
    """

    @staticmethod
    def compute(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        try:
            dtype_size = torch.empty((), dtype=dtype).element_size()
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"Unsupported torch dtype for KV-cache sizing: {dtype}") from exc
        return 2 * num_layers * num_kv_heads * head_dim * seq_len * dtype_size

    @staticmethod
    def compute_gb(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        dtype: torch.dtype = torch.float16,
    ) -> float:
        return KVCacheMemorySize.compute(
            num_layers, num_kv_heads, head_dim, seq_len, dtype
        ) / (1024 ** 3)


class AsyncKVCacheCheckpointer:
    """
    Asynchronous incremental KV-cache checkpoint engine.

    Implements Algorithm 2 (Section 3.2): Incremental KV-Cache Checkpoint Streaming.

    The engine uses a double-buffer scheme:
      - Buffer A receives new KV entries during active inference.
      - Buffer B is asynchronously transferred to disaggregated storage.
      - Buffers swap every `delta` decoding steps.

    Checkpoint overhead (Equation 4):
      T_ckpt = max(T_compute, T_transfer(delta_C))

    Transfer time (Equation 5):
      T_transfer = |delta_C| / B_net
    """

    def __init__(self, config: CheckpointConfig, session_id: str, storage_client=None):
        self.config = config
        self.session_id = session_id
        self.storage_client = storage_client

        self.version_vector = VersionVector(versions=[0] * config.num_layers)
        self.token_position = 0
        self.last_checkpoint_position = 0
        self.base_snapshot_position = 0

        # Double buffer indices
        self._active_buffer_idx = 0
        self._transfer_lock = threading.Lock()
        self._transfer_event = threading.Event()
        self._stop_event = threading.Event()
        self._transfer_thread: Optional[threading.Thread] = None

        # Statistics
        self.total_bytes_transferred = 0
        self.checkpoint_count = 0
        self.overhead_ms_total = 0.0

        # KV-cache state: dict of layer_index -> tensor (pinned memory)
        self._kv_buffers: List[Dict[int, torch.Tensor]] = [{}, {}]

        logger.info(
            "KVCacheCheckpointer initialized: session=%s backend=%s delta=%d",
            session_id, config.backend, config.delta_interval,
        )

    # ------------------------------------------------------------------
    # Algorithm 2 main loop
    # ------------------------------------------------------------------

    def start(self):
        """Start the background async transfer thread."""
        self._stop_event.clear()
        self._transfer_thread = threading.Thread(
            target=self._transfer_loop, daemon=True, name="kv-ckpt-transfer"
        )
        self._transfer_thread.start()
        logger.info("Checkpoint transfer thread started for session %s", self.session_id)

    def stop(self):
        """Stop the background transfer thread gracefully."""
        self._stop_event.set()
        self._transfer_event.set()
        if self._transfer_thread:
            self._transfer_thread.join(timeout=5.0)

    def on_tokens_generated(self, n_current: int, kv_delta: Dict[int, torch.Tensor]):
        """
        Called by the inference engine after every `delta` new tokens.

        Parameters
        ----------
        n_current : int
            Current token position in the sequence.
        kv_delta : dict
            Dict mapping layer index to the new KV entries since last checkpoint.
            Shape per layer: (2, num_kv_heads, delta_tokens, head_dim) where
            axis 0 indexes Key vs Value.
        """
        if n_current < self.token_position:
            raise ValueError(
                f"Token position must be monotonic: current={self.token_position}, received={n_current}"
            )
        self.token_position = n_current

        # Write delta to the active transfer buffer. pin_memory() is only valid
        # when an accelerator-capable PyTorch runtime is available; CPU-only CI
        # and documentation runs use a normal contiguous clone.
        active = self._active_buffer_idx
        for layer_idx, delta_tensor in kv_delta.items():
            if not 0 <= layer_idx < self.config.num_layers:
                raise ValueError(f"layer_idx {layer_idx} outside [0, {self.config.num_layers})")
            prepared = delta_tensor.detach().contiguous()
            if torch.cuda.is_available():
                try:
                    prepared = prepared.pin_memory()
                except RuntimeError:
                    logger.debug("Pinned-memory allocation unavailable; using pageable CPU memory")
                    prepared = prepared.clone()
            else:
                prepared = prepared.clone()
            if layer_idx not in self._kv_buffers[active]:
                self._kv_buffers[active][layer_idx] = prepared
            else:
                self._kv_buffers[active][layer_idx] = torch.cat(
                    [self._kv_buffers[active][layer_idx], prepared], dim=2
                )

        # Trigger async transfer if checkpoint interval reached
        if (n_current - self.last_checkpoint_position) >= self.config.delta_interval:
            self._swap_buffers_and_trigger_transfer(n_current)

        # Trigger full base snapshot refresh at epoch boundaries
        if (n_current - self.base_snapshot_position) >= self.config.base_snapshot_epoch:
            self._trigger_base_snapshot(n_current)

    def _swap_buffers_and_trigger_transfer(self, n_current: int):
        """Swap double buffers and signal the transfer thread."""
        with self._transfer_lock:
            self._active_buffer_idx = 1 - self._active_buffer_idx
            self._pending_transfer_position = n_current
            self._pending_transfer_buffer_idx = 1 - self._active_buffer_idx

        self._transfer_event.set()
        self.last_checkpoint_position = n_current

    def _trigger_base_snapshot(self, n_current: int):
        """
        Write a full base snapshot to persistent storage.
        Base snapshots refresh every `base_snapshot_epoch` tokens (default 512).
        The Delta Bytes Transferred column in Table 8 counts only sub-epoch
        incremental deltas; the base snapshot is stored separately.
        """
        self.base_snapshot_position = n_current
        logger.debug("Base snapshot triggered at token position %d", n_current)
        # In production, serialize the full KV-cache to storage here.
        # The base snapshot is loaded first during recovery, then incremental
        # deltas are replayed in version-vector order to restore full state.

    # ------------------------------------------------------------------
    # Background transfer thread (Algorithm 2, lines 8-13)
    # ------------------------------------------------------------------

    def _transfer_loop(self):
        while not self._stop_event.is_set():
            triggered = self._transfer_event.wait(timeout=1.0)
            if not triggered:
                continue
            self._transfer_event.clear()

            if self._stop_event.is_set():
                break

            try:
                self._do_transfer()
            except Exception as exc:
                logger.error("Checkpoint transfer failed: %s", exc)

    def _do_transfer(self):
        t0 = time.perf_counter()

        with self._transfer_lock:
            transfer_buf_idx = getattr(self, "_pending_transfer_buffer_idx", None)
            transfer_position = getattr(self, "_pending_transfer_position", 0)

        if transfer_buf_idx is None:
            return

        delta_bytes = self._serialize_and_upload(
            self._kv_buffers[transfer_buf_idx], transfer_position
        )

        # Update version vector for all layers (Algorithm 2, lines 9-11)
        self.version_vector.update_all(transfer_position)

        # Persist version vector to metadata store ONLY after confirmed transfer
        # (Algorithm 2, line 13: version committed only after confirmed successful transfer)
        self._persist_version_vector()

        # Clear the transferred buffer
        self._kv_buffers[transfer_buf_idx] = {}

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.total_bytes_transferred += delta_bytes
        self.checkpoint_count += 1
        self.overhead_ms_total += elapsed_ms

        logger.debug(
            "Checkpoint %d: %d bytes transferred in %.1f ms (position=%d)",
            self.checkpoint_count, delta_bytes, elapsed_ms, transfer_position,
        )

    def _serialize_and_upload(
        self, kv_buffer: Dict[int, torch.Tensor], token_position: int
    ) -> int:
        """
        Serialize KV delta buffer and upload to configured storage backend.
        Returns number of bytes uploaded.
        """
        if not kv_buffer:
            return 0

        total_bytes = 0
        prefix = self.config.gcs_prefix if self.config.backend == "gcs" else self.config.s3_prefix
        key = f"{prefix}{self.session_id}/delta_{token_position:010d}.pt"

        # In production: serialize with torch.save to a bytes buffer,
        # then upload to S3 Express One Zone or GCS regional endpoint.
        for layer_idx, tensor in kv_buffer.items():
            total_bytes += tensor.numel() * tensor.element_size()

        if self.config.backend in {"s3", "gcs"} and self.storage_client is None:
            raise RuntimeError(
                f"backend={self.config.backend!r} requires the experiment storage adapter; "
                "no checkpoint was committed"
            )
        if self.storage_client is not None:
            upload = getattr(self.storage_client, "upload", None)
            if upload is None:
                raise NotImplementedError(
                    "storage_client must provide upload(key, kv_buffer) for this reference adapter"
                )
            upload(key, kv_buffer)

        # memory backend is a bookkeeping/dry-run mode: it measures bytes but
        # deliberately does not claim durable recovery capability.
        return total_bytes

    def _persist_version_vector(self):
        """Persist version vector to metadata store after confirmed transfer."""
        if self.config.backend in {"s3", "gcs"} and self.storage_client is None:
            raise RuntimeError(
                f"backend={self.config.backend!r} requires a metadata persistence adapter"
            )
        if self.storage_client is not None:
            persist = getattr(self.storage_client, "persist_version_vector", None)
            if persist is None:
                raise NotImplementedError(
                    "storage_client must provide persist_version_vector(session_id, versions)"
                )
            persist(self.session_id, list(self.version_vector.versions))

    # ------------------------------------------------------------------
    # Recovery
    # ------------------------------------------------------------------

    def get_recovery_position(self) -> int:
        """
        Equation 7: n_recover = min(v_l(t)) for l = 1, ..., L.
        Returns the token position to which the KV-cache can be reliably restored.
        """
        return self.version_vector.recovery_position()

    def restore_kv_cache(self) -> Tuple[int, Dict[int, torch.Tensor]]:
        """
        Restore the KV-cache state on a replacement instance.

        Recovery procedure (Section 3.2):
          1. Load the most recent full base snapshot from persistent storage.
          2. Replay all incremental deltas since the last epoch base snapshot
             in version-vector order.
          3. Return the recovery token position for recomputation.

        The Delta Bytes Transferred for a 32 GB cache is 1216 MB (3.8%),
        yielding a KV-cache delta streaming and reconstruction time of 0.87 s
        (Table 8). Total end-to-end migration latency is 1390 ms (Table 11).
        """
        n_recover = self.get_recovery_position()
        logger.info(
            "Restoring KV-cache to token position %d for session %s",
            n_recover, self.session_id,
        )

        if self.storage_client is None:
            raise NotImplementedError(
                "Durable KV-cache recovery requires the original storage/base-snapshot adapter, "
                "which is not present in the supplied artifact."
            )

        # Step 1: load base snapshot (not counted in Delta Bytes Transferred)
        base_kv = self._load_base_snapshot()

        # Step 2: replay incremental deltas since last epoch boundary
        restored_kv = self._replay_deltas(base_kv, from_position=self.base_snapshot_position,
                                           to_position=n_recover)

        return n_recover, restored_kv

    def _load_base_snapshot(self) -> Dict[int, torch.Tensor]:
        """Load the full base snapshot from disaggregated storage."""
        loader = getattr(self.storage_client, "load_base_snapshot", None)
        if loader is None:
            raise NotImplementedError(
                "storage_client must provide load_base_snapshot(session_id, base_position)"
            )
        return loader(self.session_id, self.base_snapshot_position)

    def _replay_deltas(
        self,
        base_kv: Dict[int, torch.Tensor],
        from_position: int,
        to_position: int,
    ) -> Dict[int, torch.Tensor]:
        """
        Replay incremental deltas in version-vector order over the base snapshot.
        Only KV entries generated since the last 512-token epoch checkpoint
        are streamed; together with the pre-stored base snapshot, this fully
        reconstructs the complete KV-cache state.
        """
        replay = getattr(self.storage_client, "replay_deltas", None)
        if replay is None:
            raise NotImplementedError(
                "storage_client must provide replay_deltas(session_id, base_kv, from_position, to_position)"
            )
        return replay(self.session_id, base_kv, from_position, to_position)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def throughput_overhead_percent(self) -> float:
        """
        Estimate checkpointing throughput overhead.
        Target: less than 3% per Section 3.2 and the abstract.
        """
        if self.checkpoint_count == 0:
            return 0.0
        avg_overhead_ms = self.overhead_ms_total / self.checkpoint_count
        # Overhead relative to a 16-token generation window at ~50k tok/s on A100
        generation_window_ms = (self.config.delta_interval / 50_000) * 1000
        return min((avg_overhead_ms / max(generation_window_ms, 1e-6)) * 100, 100.0)

    def delta_size_percent(self, raw_cache_bytes: int) -> float:
        """
        Return the ratio of bytes transferred vs raw KV-cache size.
        Expected range: 3.8% (32 GB cache) to 8.2% (2 GB cache) per Table 8.
        """
        if raw_cache_bytes == 0:
            return 0.0
        return (self.total_bytes_transferred / raw_cache_bytes) * 100
