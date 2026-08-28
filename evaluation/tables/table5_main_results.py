"""
Table 5 Manuscript-Value Validation: Main Performance Results.

Validates the numeric values transcribed from Table 5 of the MorphCloud-LLM
manuscript. These constants are manuscript values; this script does not derive
them from the missing raw cloud experiment artifacts.

Table 5 columns: p50 Latency (ms), p99 Latency (ms), Drop Rate,
                 Throughput (tok/s), Cost (%)

Key MorphCloud-LLM results (Table 5):
  p50: 44 ± 0.9 ms
  p99: 49 ± 1.4 ms
  Drop Rate: 0.00% (Wilson 95% CI upper bound: 0.15%)
  Throughput: 121 ± 2.3 tok/s
  Cost: 27 ± 1.1% of on-demand

Usage:
    python evaluation/tables/table5_main_results.py
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SystemResult:
    """Result record for one baseline/system."""
    name: str
    p50_mean: float
    p50_sd: float
    p99_mean: float
    p99_sd: float
    drop_rate: str          # formatted string (Table 5 format)
    throughput_mean: float
    throughput_sd: float
    cost_mean: float
    cost_sd: Optional[float] = None
    reference: str = ""


# ---------------------------------------------------------------------------
# Table 5 ground truth (from the paper)
# ---------------------------------------------------------------------------

TABLE5_RESULTS: List[SystemResult] = [
    SystemResult(
        name="On-Demand",
        p50_mean=42, p50_sd=0.8,
        p99_mean=48, p99_sd=1.2,
        drop_rate="0.00%",
        throughput_mean=125, throughput_sd=2.1,
        cost_mean=100,
    ),
    SystemResult(
        name="MorphCloud-LLM (Ours)",
        p50_mean=44, p50_sd=0.9,
        p99_mean=49, p99_sd=1.4,
        drop_rate="0.00% (95% CI upper bound 0.15%)",
        throughput_mean=121, throughput_sd=2.3,
        cost_mean=27, cost_sd=1.1,
    ),
    SystemResult(
        name="SpotServe",
        p50_mean=51, p50_sd=2.1,
        p99_mean=78, p99_sd=4.3,
        drop_rate="0.12% ± 0.03%",
        throughput_mean=108, throughput_sd=3.8,
        cost_mean=38, cost_sd=1.5,
        reference="[1]",
    ),
    SystemResult(
        name="ServerlessLLM",
        p50_mean=58, p50_sd=2.8,
        p99_mean=95, p99_sd=5.6,
        drop_rate="0.35% ± 0.06%",
        throughput_mean=96, throughput_sd=4.2,
        cost_mean=45, cost_sd=1.8,
        reference="[2]",
    ),
    SystemResult(
        name="GFS",
        p50_mean=55, p50_sd=2.4,
        p99_mean=88, p99_sd=5.1,
        drop_rate="0.28% ± 0.05%",
        throughput_mean=102, throughput_sd=3.6,
        cost_mean=42, cost_sd=1.7,
        reference="[3]",
    ),
    SystemResult(
        name="ServerlessPD",
        p50_mean=53, p50_sd=2.2,
        p99_mean=82, p99_sd=4.7,
        drop_rate="0.20% ± 0.04%",
        throughput_mean=105, throughput_sd=3.4,
        cost_mean=40, cost_sd=1.6,
        reference="[4]",
    ),
    SystemResult(
        name="vLLM+Ckpt",
        p50_mean=45, p50_sd=1.1,
        p99_mean=350, p99_sd=28.4,
        drop_rate="2.80% ± 0.21%",
        throughput_mean=85, throughput_sd=5.1,
        cost_mean=32, cost_sd=1.3,
    ),
    SystemResult(
        name="Vanilla Spot",
        p50_mean=42, p50_sd=0.9,
        p99_mean=float("inf"), p99_sd=0,
        drop_rate="8.50% ± 0.62%",
        throughput_mean=125, throughput_sd=2.1,
        cost_mean=28, cost_sd=1.0,
    ),
    SystemResult(
        name="K8s Restart",
        p50_mean=42, p50_sd=1.0,
        p99_mean=520, p99_sd=42.3,
        drop_rate="5.20% ± 0.38%",
        throughput_mean=90, throughput_sd=4.8,
        cost_mean=30, cost_sd=1.2,
    ),
    SystemResult(
        name="Cong et al.",
        p50_mean=46, p50_sd=1.3,
        p99_mean=72, p99_sd=3.8,
        drop_rate="0.45% ± 0.07%",
        throughput_mean=112, throughput_sd=3.1,
        cost_mean=100,
        reference="[14]",
    ),
    SystemResult(
        name="PhoenixOS",
        p50_mean=48, p50_sd=1.6,
        p99_mean=95, p99_sd=5.9,
        drop_rate="0.55% ± 0.08%",
        throughput_mean=100, throughput_sd=3.9,
        cost_mean=35, cost_sd=1.4,
        reference="[12]",
    ),
    SystemResult(
        name="Native-Notify",
        p50_mean=43, p50_sd=1.2,
        p99_mean=58, p99_sd=3.1,
        drop_rate="0.31% ± 0.05%",
        throughput_mean=122, throughput_sd=2.8,
        cost_mean=28, cost_sd=1.1,
    ),
]


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_table5():
    header = (
        f"{'Method':<30} {'p50 (ms)':>12} {'p99 (ms)':>12} "
        f"{'Drop Rate':>28} {'Tput (tok/s)':>14} {'Cost %':>8}"
    )
    sep = "-" * len(header)
    print("\nTable 5: Main Performance Results")
    print(sep)
    print(header)
    print(sep)
    for r in TABLE5_RESULTS:
        p50 = f"{r.p50_mean:.0f} ± {r.p50_sd:.1f}"
        p99_val = "inf" if r.p99_mean == float("inf") else f"{r.p99_mean:.0f} ± {r.p99_sd:.1f}"
        tput = f"{r.throughput_mean:.0f} ± {r.throughput_sd:.1f}"
        cost = f"{r.cost_mean:.0f}"
        if r.cost_sd:
            cost += f" ± {r.cost_sd:.1f}"
        name = r.name + (f" {r.reference}" if r.reference else "")
        print(
            f"{name:<30} {p50:>12} {p99_val:>12} "
            f"{r.drop_rate:>28} {tput:>14} {cost:>8}"
        )
    print(sep)
    print("\nNotes:")
    print("  MorphCloud-LLM cost 27%: active-serving basis (69.8% cost reduction).")
    print("  Including warm standby ($0.92/h): total $10.81/h, 67% cost reduction.")
    print("  Bootstrap 95% CI (event-level) upper bound for drop rate: 0.73%.")
    print("  Wilson score 95% CI upper bound (independent-trial): 0.15%.")


# ---------------------------------------------------------------------------
# Verification against Table 5 values
# ---------------------------------------------------------------------------

def verify_morphcloud_values():
    """Verify that MorphCloud-LLM results match Table 5 exactly."""
    morph = next(r for r in TABLE5_RESULTS if "MorphCloud" in r.name)
    assert morph.p50_mean == 44, f"p50 mismatch: {morph.p50_mean}"
    assert morph.p99_mean == 49, f"p99 mismatch: {morph.p99_mean}"
    assert morph.throughput_mean == 121, f"throughput mismatch: {morph.throughput_mean}"
    assert morph.cost_mean == 27, f"cost mismatch: {morph.cost_mean}"
    print("Table 5 verification PASSED for MorphCloud-LLM.")


if __name__ == "__main__":
    print_table5()
    verify_morphcloud_values()

    # Optionally save to JSON for downstream use
    output_dir = os.getenv("MORPHCLOUD_OUTPUT_DIR", "evaluation/results")
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for r in TABLE5_RESULTS:
        d = {
            "method": r.name,
            "p50_mean_ms": r.p50_mean,
            "p50_sd_ms": r.p50_sd,
            "p99_mean_ms": r.p99_mean if r.p99_mean != float("inf") else "inf",
            "p99_sd_ms": r.p99_sd,
            "drop_rate": r.drop_rate,
            "throughput_mean_tok_s": r.throughput_mean,
            "throughput_sd_tok_s": r.throughput_sd,
            "cost_pct": r.cost_mean,
            "reference": r.reference,
        }
        records.append(d)
    out_path = os.path.join(output_dir, "table5_main_results.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nTable 5 data saved to {out_path}")
