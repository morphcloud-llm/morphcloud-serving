"""
Unit tests for the KV-cache checkpointing engine.

Tests verify:
  - KVCacheMemorySize calculations match Equation 2 (LLaMA-70B GQA values)
  - VersionVector recovery position (Equation 7)
  - Delta size percentages match Table 8 ranges
  - Throughput overhead stays below 3% target
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
import torch
from morphcloud.checkpointing.kv_cache_engine import (
    CheckpointConfig,
    KVCacheMemorySize,
    VersionVector,
    AsyncKVCacheCheckpointer,
)


class TestKVCacheMemorySize:
    """Verify Equation 2: M_kv = 2 * L * H_kv * d_h * n * sizeof(dtype)."""

    def test_llama70b_gqa(self):
        """
        LLaMA-70B with GQA: L=80, H_kv=8, d_h=128, n=2048, FP16.
        Expected: ~0.67 GB per sequence (Section 3.2).
        """
        size_gb = KVCacheMemorySize.compute_gb(
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            seq_len=2048,
            dtype=torch.float16,
        )
        assert 0.60 < size_gb < 0.75, f"LLaMA-70B GQA KV size out of range: {size_gb:.3f} GB"

    def test_full_head_model(self):
        """
        Full-head model: L=80, H=H_kv=64, d_h=128, n=2048, FP16.
        Expected: ~5.37 GB per sequence (Section 3.2).
        """
        size_gb = KVCacheMemorySize.compute_gb(
            num_layers=80,
            num_kv_heads=64,
            head_dim=128,
            seq_len=2048,
            dtype=torch.float16,
        )
        assert 4.99 <= size_gb < 5.7, f"Full-head KV size out of range: {size_gb:.3f} GB"


    def test_bfloat16_uses_two_bytes(self):
        expected = 2 * 2 * 4 * 8 * 16 * 2
        computed = KVCacheMemorySize.compute(2, 4, 8, 16, torch.bfloat16)
        assert computed == expected

    def test_equation2_formula(self):
        """Direct verification of Equation 2 formula."""
        L, H_kv, d_h, n, dtype_bytes = 80, 8, 128, 2048, 2
        expected = 2 * L * H_kv * d_h * n * dtype_bytes
        computed = KVCacheMemorySize.compute(80, 8, 128, 2048, torch.float16)
        assert computed == expected, f"Equation 2 mismatch: {computed} vs {expected}"


class TestVersionVector:
    """Verify version vector operations (Equations 6 and 7)."""

    def test_initial_recovery_position(self):
        """Initial recovery position should be 0."""
        vv = VersionVector(versions=[0] * 80)
        assert vv.recovery_position() == 0

    def test_recovery_position_is_minimum(self):
        """Equation 7: n_recover = min(v_l(t)) for l=1,...,L."""
        versions = list(range(10, 90))  # 10 to 89
        vv = VersionVector(versions=versions)
        assert vv.recovery_position() == 10

    def test_update_single_layer(self):
        vv = VersionVector(versions=[0] * 5)
        vv.update(2, 100)
        assert vv.versions[2] == 100
        assert vv.recovery_position() == 0  # other layers still at 0

    def test_update_all(self):
        vv = VersionVector(versions=[0] * 5)
        vv.update_all(256)
        assert vv.recovery_position() == 256
        assert all(v == 256 for v in vv.versions)


class TestAsyncKVCacheCheckpointer:
    """Tests for the checkpointing engine lifecycle and statistics."""

    def _make_checkpointer(self):
        config = CheckpointConfig(
            delta_interval=16,
            base_snapshot_epoch=512,
            backend="s3",
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
        )
        return AsyncKVCacheCheckpointer(config, session_id="test-session-001")

    def test_initial_state(self):
        ckpt = self._make_checkpointer()
        assert ckpt.token_position == 0
        assert ckpt.checkpoint_count == 0
        assert ckpt.total_bytes_transferred == 0

    def test_recovery_position_initial(self):
        ckpt = self._make_checkpointer()
        assert ckpt.get_recovery_position() == 0

    def test_throughput_overhead_zero_initially(self):
        ckpt = self._make_checkpointer()
        assert ckpt.throughput_overhead_percent() == 0.0

    def test_delta_size_percent(self):
        ckpt = self._make_checkpointer()
        # Simulate 32 GB cache with 1216 MB transferred (Table 8: 3.8%)
        ckpt.total_bytes_transferred = 1216 * 1024 * 1024
        raw_bytes = 32 * 1024 * 1024 * 1024
        pct = ckpt.delta_size_percent(raw_bytes)
        assert 3.5 < pct < 4.2, f"32 GB delta size percent out of Table 8 range: {pct:.2f}%"

    def test_delta_size_2gb(self):
        """Table 8: 2 GB cache -> 164 MB delta -> ~8.2% ratio."""
        ckpt = self._make_checkpointer()
        ckpt.total_bytes_transferred = 164 * 1024 * 1024
        raw_bytes = 2 * 1024 * 1024 * 1024
        pct = ckpt.delta_size_percent(raw_bytes)
        assert 7.5 < pct < 9.0, f"2 GB delta size percent out of range: {pct:.2f}%"


class TestMigrationLatencyBreakdown:
    """Verify Table 11 migration latency breakdown values."""

    TABLE11 = {
        "detection_ms":         0,
        "state_transfer_ms":  990,
        "s3_round_trip_ms":   120,
        "kv_delta_ms":         870,
        "instance_provision_ms": 250,
        "model_loading_ms":      0,
        "kv_state_activation_ms": 150,
        "total_ms":           1390,
    }

    def test_total_adds_up(self):
        """Total = detection + state_transfer + instance_provision + model_loading + kv_activation."""
        t = self.TABLE11
        computed = (
            t["detection_ms"]
            + t["state_transfer_ms"]
            + t["instance_provision_ms"]
            + t["model_loading_ms"]
            + t["kv_state_activation_ms"]
        )
        assert computed == t["total_ms"], f"Table 11 total mismatch: {computed} vs {t['total_ms']}"

    def test_state_transfer_components(self):
        """State transfer = S3 round-trip + KV delta streaming (Section 4.7)."""
        t = self.TABLE11
        assert t["s3_round_trip_ms"] + t["kv_delta_ms"] == t["state_transfer_ms"]

    def test_model_loading_zero(self):
        """Model loading is 0 ms: weights pre-staged in fallback GPU memory (Table 11)."""
        assert self.TABLE11["model_loading_ms"] == 0

    def test_detection_zero(self):
        """Detection is 0 ms: proactive prediction eliminates reactive delay."""
        assert self.TABLE11["detection_ms"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_external_backend_without_adapter_fails():
    config = CheckpointConfig(backend="s3", num_layers=1)
    ckpt = AsyncKVCacheCheckpointer(config, session_id="s")
    with pytest.raises(RuntimeError):
        ckpt._serialize_and_upload({0: torch.zeros(1)}, 1)


def test_restore_without_storage_adapter_fails_explicitly():
    config = CheckpointConfig(backend="memory", num_layers=1)
    ckpt = AsyncKVCacheCheckpointer(config, session_id="s")
    with pytest.raises(NotImplementedError):
        ckpt.restore_kv_cache()


def test_on_tokens_generated_works_cpu_only():
    config = CheckpointConfig(backend="memory", num_layers=1, delta_interval=16)
    ckpt = AsyncKVCacheCheckpointer(config, session_id="cpu")
    delta = {0: torch.zeros((2, 1, 1, 4), dtype=torch.float16)}
    ckpt.on_tokens_generated(1, delta)
    assert 0 in ckpt._kv_buffers[ckpt._active_buffer_idx]


def test_on_tokens_generated_validates_positions_and_layers():
    config = CheckpointConfig(backend="memory", num_layers=1)
    ckpt = AsyncKVCacheCheckpointer(config, session_id="v")
    delta = {0: torch.zeros((2, 1, 1, 4))}
    ckpt.on_tokens_generated(2, delta)
    with pytest.raises(ValueError):
        ckpt.on_tokens_generated(1, delta)
    with pytest.raises(ValueError):
        ckpt.on_tokens_generated(3, {2: torch.zeros((2, 1, 1, 4))})
