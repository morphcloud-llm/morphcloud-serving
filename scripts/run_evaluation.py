"""Run manuscript-value validation/plotting helpers.

Raw traces/model artifacts are required for full experimental reproduction and
are not silently synthesized here.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "evaluation/tables/table5_main_results.py",
    "evaluation/ablation/ablation_study.py",
    "evaluation/figures/fig2_loss_convergence.py",
    "evaluation/figures/fig3_fig4_classification.py",
    "evaluation/figures/fig5_fig6_prediction_metrics.py",
    "evaluation/figures/fig7_fig8_fig10_latency_cost_throughput.py",
    "evaluation/figures/fig9_kv_recovery.py",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MorphCloud-LLM manuscript values")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--output-dir", default="evaluation/results")
    args = parser.parse_args()

    output_dir = (ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MORPHCLOUD_OUTPUT_DIR"] = str(output_dir)

    selected = [s for s in SCRIPTS if not (args.skip_figures and "/figures/" in s)]
    for rel in selected:
        print(f"==> {rel}", flush=True)
        subprocess.run([sys.executable, str(ROOT / rel)], cwd=ROOT, env=env, check=True)

    print(f"\nPublished-result validation completed. Outputs: {output_dir}")
    print("For clearly labeled synthetic event simulation, run scripts/simulate_preemption.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
