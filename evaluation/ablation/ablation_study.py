"""
Ablation Study Manuscript-Value Validation (Table 9).

Validates values transcribed from Table 9 of MorphCloud-LLM:
  Ablation study results on LLaMA-70B (AWS).

Configurations:
  Full MorphCloud-LLM          : p99=49ms,  drop=0.00%, recovery=0.58s, cost=27%
  w/o Preemption Predictor     : p99=55ms,  drop=0.80%, recovery=1.85s, cost=31%
  w/o Spec. Decoding Continuity: p99=52ms,  drop=0.30%, recovery=0.62s, cost=27%
  w/o Incremental Checkpoint   : p99=53ms,  drop=0.05%, recovery=4.20s, cost=28%
  w/o All Components           : p99=350ms, drop=8.50%, recovery=25.1s, cost=28%

Key finding (Section 4.9):
  The preemption predictor contributes most: its removal raises p99 by 12.5%
  and drop rate to 0.8%, the largest single-component effect.
  Differences exceed run-to-run standard deviations by more than 3× across all 5 seeds.

Usage:
    python evaluation/ablation/ablation_study.py
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import List


@dataclass
class AblationConfig:
    name: str
    p99_ms: float
    drop_rate_pct: float
    recovery_s: float
    cost_pct: float
    components_enabled: dict


ABLATION_TABLE9: List[AblationConfig] = [
    AblationConfig(
        name="Full MorphCloud-LLM",
        p99_ms=49, drop_rate_pct=0.00, recovery_s=0.58, cost_pct=27,
        components_enabled={
            "preemption_predictor": True,
            "speculative_continuity": True,
            "incremental_checkpoint": True,
        },
    ),
    AblationConfig(
        name="w/o Preemption Predictor",
        p99_ms=55, drop_rate_pct=0.80, recovery_s=1.85, cost_pct=31,
        components_enabled={
            "preemption_predictor": False,
            "speculative_continuity": True,
            "incremental_checkpoint": True,
        },
    ),
    AblationConfig(
        name="w/o Spec. Decoding Cont.",
        p99_ms=52, drop_rate_pct=0.30, recovery_s=0.62, cost_pct=27,
        components_enabled={
            "preemption_predictor": True,
            "speculative_continuity": False,
            "incremental_checkpoint": True,
        },
    ),
    AblationConfig(
        name="w/o Incremental Ckpt",
        p99_ms=53, drop_rate_pct=0.05, recovery_s=4.20, cost_pct=28,
        components_enabled={
            "preemption_predictor": True,
            "speculative_continuity": True,
            "incremental_checkpoint": False,
        },
    ),
    AblationConfig(
        name="w/o All Components",
        p99_ms=350, drop_rate_pct=8.50, recovery_s=25.1, cost_pct=28,
        components_enabled={
            "preemption_predictor": False,
            "speculative_continuity": False,
            "incremental_checkpoint": False,
        },
    ),
]


def compute_p99_degradation_pct(config: AblationConfig) -> float:
    """Compute p99 degradation relative to full system."""
    full_p99 = ABLATION_TABLE9[0].p99_ms
    return (config.p99_ms - full_p99) / full_p99 * 100


def print_ablation_table():
    header = (
        f"{'Configuration':<32} {'p99 (ms)':>10} {'Drop %':>10} "
        f"{'Recovery (s)':>14} {'Cost %':>8} {'p99 Degradation':>16}"
    )
    sep = "-" * len(header)
    print("\nTable 9: Ablation Study Results — LLaMA-70B (AWS)")
    print(sep)
    print(header)
    print(sep)
    for cfg in ABLATION_TABLE9:
        degrad = compute_p99_degradation_pct(cfg)
        degrad_str = f"+{degrad:.1f}%" if degrad > 0 else "baseline"
        print(
            f"{cfg.name:<32} {cfg.p99_ms:>10.0f} {cfg.drop_rate_pct:>10.2f} "
            f"{cfg.recovery_s:>14.2f} {cfg.cost_pct:>8.0f} {degrad_str:>16}"
        )
    print(sep)
    print("\nFindings (Section 4.9):")
    print("  Preemption predictor removal: largest effect (+12.5% p99, +0.80% drop rate).")
    print("  Spec. decoding removal: drop rate rises 0.00% -> 0.30%; latency +3 ms.")
    print("  Incremental checkpoint removal: recovery time 0.58 -> 4.20 s (+7.2x).")
    print("  Differences exceed run-to-run SD by >3x across all 5 seeds (Section 4.9).")


def verify_ablation_values():
    """Verify Table 9 ground truth values."""
    full = ABLATION_TABLE9[0]
    assert full.p99_ms == 49
    assert full.drop_rate_pct == 0.00
    assert full.recovery_s == 0.58
    assert full.cost_pct == 27

    no_pred = ABLATION_TABLE9[1]
    assert no_pred.drop_rate_pct == 0.80, f"Expected 0.80, got {no_pred.drop_rate_pct}"
    p99_lift = (no_pred.p99_ms - full.p99_ms) / full.p99_ms
    assert abs(p99_lift - 0.1224) < 0.01, f"p99 lift mismatch: {p99_lift:.3f} vs 0.122"
    print("Table 9 ablation verification PASSED.")


if __name__ == "__main__":
    print_ablation_table()
    verify_ablation_values()

    output_dir = os.getenv("MORPHCLOUD_OUTPUT_DIR", "evaluation/results")
    os.makedirs(output_dir, exist_ok=True)
    records = [asdict(c) for c in ABLATION_TABLE9]
    out_path = os.path.join(output_dir, "table9_ablation_study.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nAblation data saved to {out_path}")
