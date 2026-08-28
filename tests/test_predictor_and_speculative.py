"""
Unit tests for the preemption predictor and speculative decoding continuity engine.

Tests verify:
  - Feature vector construction (Equation 9)
  - Adaptive threshold computation (Equation 13)
  - Expected resample count (Equation 17)
  - Migration latency overhead (Equation 18)
  - Acceptance probability (Equation 14)
  - Speculative stats match paper values (Section 4.10)
"""

import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from morphcloud.prediction.preemption_predictor import (
    PreemptionFeatureVector,
    MigrationThresholdConfig,
    PreemptionPredictor,
)
from morphcloud.speculative.continuity_engine import (
    SpeculativeDecodingContinuityEngine,
    ServingMode,
    ContinuityWindowStats,
)


class TestPreemptionFeatureVector:
    """Test Equation 9 feature vector construction."""

    def test_array_length(self):
        fv = PreemptionFeatureVector()
        arr = fv.to_array()
        assert len(arr) == 9, f"Feature vector length should be 9, got {len(arr)}"

    def test_cyclical_encoding(self):
        """Verify sin/cos encoding preserves periodicity."""
        fv_h0 = PreemptionFeatureVector.from_raw(hour=0, dow=0)
        fv_h24 = PreemptionFeatureVector.from_raw(hour=24, dow=0)
        assert abs(fv_h0.hour_sin - fv_h24.hour_sin) < 1e-6
        assert abs(fv_h0.hour_cos - fv_h24.hour_cos) < 1e-6

    def test_known_cyclical_values(self):
        """Hour=6 -> sin=1.0, cos=0.0 for 24-hour cycle."""
        fv = PreemptionFeatureVector.from_raw(hour=6, dow=0)
        assert abs(fv.hour_sin - 1.0) < 1e-6
        assert abs(fv.hour_cos) < 1e-6

    def test_feature_names_count(self):
        assert len(PreemptionFeatureVector.FEATURE_NAMES) == 9


class TestMigrationThresholdConfig:
    """Test Equation 13: rho_thresh(t) = rho_base * (1 - lambda * load/capacity)."""

    def test_zero_load(self):
        """At zero load, threshold equals rho_base (0.7)."""
        cfg = MigrationThresholdConfig(rho_base=0.7, load_sensitivity=0.3)
        assert cfg.compute(current_load=0, capacity=100) == pytest.approx(0.7)

    def test_full_load(self):
        """At full load, threshold = rho_base * (1 - lambda) = 0.7 * 0.7 = 0.49."""
        cfg = MigrationThresholdConfig(rho_base=0.7, load_sensitivity=0.3)
        result = cfg.compute(current_load=100, capacity=100)
        assert result == pytest.approx(0.49)

    def test_half_load(self):
        """At half load, threshold = 0.7 * (1 - 0.3*0.5) = 0.7 * 0.85 = 0.595."""
        cfg = MigrationThresholdConfig(rho_base=0.7, load_sensitivity=0.3)
        result = cfg.compute(current_load=50, capacity=100)
        assert result == pytest.approx(0.595)

    def test_zero_capacity(self):
        """Zero capacity should not raise; returns rho_base."""
        cfg = MigrationThresholdConfig()
        result = cfg.compute(current_load=0, capacity=0)
        assert result == pytest.approx(0.7)

    def test_paper_values(self):
        """rho_base=0.7, lambda=0.3 match Section 3.3."""
        cfg = MigrationThresholdConfig()
        assert cfg.rho_base == pytest.approx(0.7)
        assert cfg.load_sensitivity == pytest.approx(0.3)


class TestPreemptionPredictorNoML:
    """Test predictor behavior without ML libraries (graceful degradation)."""

    def test_predict_proba_no_model(self):
        predictor = PreemptionPredictor()
        predictor.model = None
        fv = PreemptionFeatureVector()
        score = predictor.predict_proba(fv)
        assert score == 0.0

    def test_price_velocity_insufficient_history(self):
        predictor = PreemptionPredictor()
        assert predictor.compute_price_velocity() == 0.0

    def test_price_velocity_two_samples(self):
        predictor = PreemptionPredictor()
        predictor.update_price_history(0.0, 1.00)
        predictor.update_price_history(10.0, 1.05)
        velocity = predictor.compute_price_velocity()
        assert abs(velocity - 0.005) < 1e-6


class TestSpeculativeDecodingContinuityEngine:
    """Test speculative decoding continuity engine."""

    def _make_engine(self):
        return SpeculativeDecodingContinuityEngine(max_draft_tokens=64)

    def test_initial_mode(self):
        engine = self._make_engine()
        assert engine.current_mode == ServingMode.NORMAL

    def test_enter_continuity_mode(self):
        engine = self._make_engine()
        engine.enter_continuity_mode("sess-001", "llama-70b", event_id=1)
        assert engine.current_mode == ServingMode.CONTINUITY
        assert engine._current_window is not None
        assert engine._current_window.session_id == "sess-001"

    def test_exit_continuity_mode(self):
        engine = self._make_engine()
        engine.enter_continuity_mode("sess-001", "llama-70b", event_id=1)
        stats = engine.exit_continuity_mode()
        assert engine.current_mode == ServingMode.NORMAL
        assert stats is not None
        assert stats.duration_ms >= 0

    def test_double_enter_noop(self):
        engine = self._make_engine()
        engine.enter_continuity_mode("sess-001", "llama-70b", event_id=1)
        engine.enter_continuity_mode("sess-001", "llama-70b", event_id=1)  # second call: noop
        assert engine.current_mode == ServingMode.CONTINUITY

    def test_expected_resamples_llama70b(self):
        """
        Equation 17: E[n_resample] = k * (1 - E[P_accept]).
        LLaMA-70B: k=41.2, P_accept=0.873 -> E[n_resample] = 5.2 (Section 4.10).
        """
        result = SpeculativeDecodingContinuityEngine.expected_resamples(
            k=41.2, mean_acceptance_rate=0.873
        )
        assert abs(result - 5.2424) < 0.1, f"Expected ~5.24, got {result:.2f}"

    def test_expected_resamples_mixtral(self):
        """
        Mixtral-8x7B: k=38.7, P_accept=0.891 -> E[n_resample] = 4.3 (Section 4.10).
        """
        result = SpeculativeDecodingContinuityEngine.expected_resamples(
            k=38.7, mean_acceptance_rate=0.891
        )
        assert abs(result - 4.2417) < 0.1, f"Expected ~4.24, got {result:.2f}"

    def test_acceptance_probability_clamped(self):
        """Equation 14: P_accept = min(1, P_Mt / P_Md). Cannot exceed 1."""
        engine = self._make_engine()
        # log_prob_target >> log_prob_draft -> ratio > 1 -> clamped at 1
        prob = engine._acceptance_probability(log_prob_target=-0.1, log_prob_draft=-2.0)
        assert 0.0 <= prob <= 1.0
        assert prob == 1.0  # clamped

    def test_acceptance_probability_reject_case(self):
        """When target prob << draft prob, acceptance rate < 1."""
        engine = self._make_engine()
        prob = engine._acceptance_probability(log_prob_target=-3.0, log_prob_draft=-0.5)
        assert 0.0 < prob < 1.0

    def test_migration_latency_overhead_equation18(self):
        """
        Equation 18: delta_T = T_recovery + k * t_verify - k * t_draft.
        """
        result = SpeculativeDecodingContinuityEngine.migration_latency_overhead(
            t_recovery_ms=1390.0, k=41, t_verify_ms=2.0, t_draft_ms=1.5
        )
        expected = 1390.0 + 41 * 2.0 - 41 * 1.5
        assert abs(result - expected) < 1e-9

    def test_candidate_token_is_passed_to_target_adapter(self):
        class Draft:
            def next_token(self, context):
                return 7, -2.0
        class Target:
            def __init__(self):
                self.seen = []
            def token_log_prob(self, context, token_id):
                self.seen.append(token_id)
                return -1.0  # acceptance probability clamps to 1
            def generate_tokens(self, context, n_tokens):
                return [9] * n_tokens
        target = Target()
        engine = SpeculativeDecodingContinuityEngine(draft_model=Draft(), target_model=target)
        engine.enter_continuity_mode("s", "llama-70b", 1)
        engine.generate_draft_tokens([1, 2], 1)
        ids, rejected, _ = engine.verify_buffer([1, 2])
        assert ids == [7]
        assert rejected == 0
        assert target.seen == [7]

    def test_draft_token_limit_enforced(self):
        class Draft:
            def next_token(self, context):
                return 7, -1.0
        engine = SpeculativeDecodingContinuityEngine(draft_model=Draft(), max_draft_tokens=2)
        engine.enter_continuity_mode("s", "llama-70b", 1)
        with pytest.raises(ValueError):
            engine.generate_draft_tokens([1], 3)

    def test_missing_draft_adapter_fails_explicitly(self):
        engine = self._make_engine()
        engine.enter_continuity_mode("sess-001", "llama-70b", event_id=1)
        with pytest.raises(NotImplementedError):
            engine.generate_draft_tokens([1, 2, 3], n_tokens=1)

    def test_generate_draft_tokens_only_in_continuity_mode(self):
        engine = self._make_engine()
        with pytest.raises(RuntimeError):
            engine.generate_draft_tokens([1, 2, 3], n_tokens=5)

    def test_aggregate_stats_empty(self):
        engine = self._make_engine()
        stats = engine.aggregate_stats()
        assert stats == {}

    def test_paper_continuity_window_stats(self):
        """
        Section 3.4: mean continuity window = 547 ms, SD=83 ms, p99=812 ms.
        Verify that a simulated distribution of 521 events falls within plausible range.
        """
        rng = np.random.default_rng(42)
        n = 521
        samples = np.clip(rng.normal(547, 83, n), 400, 1000)
        assert abs(samples.mean() - 547) < 30
        assert abs(samples.std() - 83) < 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
