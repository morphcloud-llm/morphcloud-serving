from pathlib import Path

import numpy as np
import pytest

from scripts.simulate_preemption import PreemptionSimulator
from scripts.train_predictor import chronological_split, load_telemetry

ROOT = Path(__file__).resolve().parents[1]


def test_no_accidental_brace_directories():
    bad = [p for p in ROOT.rglob("*") if p.is_dir() and ("{" in p.name or "}" in p.name)]
    assert bad == []


def test_canonical_simulator_generates_521_events():
    events = PreemptionSimulator(seed=42).generate_event_schedule(521, platforms=["aws", "gcp"])
    assert len(events) == 521
    assert sum(e.platform == "aws" for e in events) == 274
    assert sum(e.platform == "gcp" for e in events) == 247
    from collections import Counter
    severity = Counter(e.severity.value for e in events)
    assert severity == {"single": 312, "concurrent_dual": 156, "full": 53}
    aws_mean_min = np.mean([e.inter_arrival_s for e in events if e.platform == "aws"]) / 60
    gcp_mean_min = np.mean([e.inter_arrival_s for e in events if e.platform == "gcp"]) / 60
    assert aws_mean_min == pytest.approx(47.3)
    assert gcp_mean_min == pytest.approx(38.9)
    assert [e.event_id for e in events] == list(range(521))


def test_simulator_rejects_invented_extra_events():
    with pytest.raises(ValueError):
        PreemptionSimulator(seed=1).generate_event_schedule(522, platforms=["aws", "gcp"])


def test_missing_telemetry_is_not_synthesized(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_telemetry(str(tmp_path), ["us-east-1"])


def test_chronological_split_lengths():
    X = np.arange(100 * 9, dtype=np.float32).reshape(100, 9)
    y = np.zeros(100, dtype=np.int32)
    parts = chronological_split(X, y)
    assert [len(parts[i]) for i in (0, 1, 2, 3, 4, 5)] == [70, 70, 15, 15, 15, 15]


def test_table7_cost_is_aggregate_not_per_registered_instance():
    from morphcloud.orchestrator.morphcloud_orchestrator import MorphCloudOrchestrator, SpotInstance
    orch = MorphCloudOrchestrator()
    for i in range(8):
        orch.state.spot_instances[str(i)] = SpotInstance(str(i), "aws", "us-east-1", "A100-80GB", 0, 1, 0, 0)
    assert orch.compute_total_cost(include_warm_standby=False) == pytest.approx(9.89)
    assert orch.compute_total_cost(include_warm_standby=True) == pytest.approx(10.81)
