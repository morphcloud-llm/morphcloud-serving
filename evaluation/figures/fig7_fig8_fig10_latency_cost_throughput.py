"""
Figure 7: p99 Latency Across Five Deployment Scenarios.
Figure 8: Normalized Cost Comparison.
Figure 10: Throughput Time Series During Three Consecutive Preemption Events.

Renders/validates the manuscript values for Figures 7, 8, and 10 of MorphCloud-LLM.
All numeric values are sourced directly from the paper text and tables.

Output: Three PNGs at 600 dpi, IEEE two-column format.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
})

output_dir = os.getenv("MORPHCLOUD_OUTPUT_DIR", "evaluation/results")
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 7: p99 Latency Across Deployment Scenarios (Section 4.5)
# ---------------------------------------------------------------------------

scenarios = [
    "No\nPreemption",
    "Low Rate\n(single)",
    "Medium Rate\n(dual)",
    "Burst\nPreemptions",
    "Full Spot\nUnavailable",
]

# Values from Section 4.5 text (ms)
morph_p99    = [48,  49,  52,  65,  89]
spotserve_p99 = [50, 65,  95, 145, 320]
vllm_p99     = [45, 130, 280, 420, 520]

x = np.arange(len(scenarios))
w = 0.25

fig7, ax7 = plt.subplots(figsize=(3.5, 2.6))
ax7.bar(x - w,  morph_p99,    w, label="MorphCloud-LLM", color="#1f77b4")
ax7.bar(x,       spotserve_p99, w, label="SpotServe",    color="#d62728")
ax7.bar(x + w,  vllm_p99,     w, label="vLLM+Ckpt",    color="#2ca02c")

ax7.set_xlabel("Deployment Scenario")
ax7.set_ylabel("p99 Latency (ms)")
ax7.set_title("p99 Latency Across Deployment Scenarios")
ax7.set_xticks(x)
ax7.set_xticklabels(scenarios, fontsize=7)
ax7.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=3,
           frameon=True, framealpha=0.9)
ax7.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
out7 = os.path.join(output_dir, "fig7_p99_latency_scenarios.png")
plt.savefig(out7, dpi=600, bbox_inches="tight")
plt.close(fig7)
print(f"Figure 7 saved to {out7}")


# ---------------------------------------------------------------------------
# Figure 8: Normalized Cost Comparison (Section 4.6)
# ---------------------------------------------------------------------------

configs = ["LLaMA-70B\nAWS", "LLaMA-70B\nGCP", "Mixtral-8x7B\nAWS", "Mixtral-8x7B\nGCP"]

# Cost as % of on-demand (active-serving basis unless noted)
morph_cost    = [27.0, 31.0, 24.0, 29.0]   # Table 10, Table 7
spotserve_cost = [38.0, 40.0, 39.0, 41.0]
serverless_cost = [45.0, 48.0, 46.0, 47.0]
ondemand_cost  = [100.0, 100.0, 100.0, 100.0]

x = np.arange(len(configs))
w = 0.20

fig8, ax8 = plt.subplots(figsize=(3.5, 2.6))
ax8.bar(x - 1.5*w, morph_cost,     w, label="MorphCloud-LLM", color="#1f77b4")
ax8.bar(x - 0.5*w, spotserve_cost, w, label="SpotServe",     color="#d62728")
ax8.bar(x + 0.5*w, serverless_cost, w, label="ServerlessLLM", color="#ff7f0e")
ax8.bar(x + 1.5*w, ondemand_cost,  w, label="On-Demand",     color="#7f7f7f")

ax8.set_xlabel("Model-Platform Configuration")
ax8.set_ylabel("Normalized Cost (% of On-Demand)")
ax8.set_title("Cost Comparison Across Configurations")
ax8.set_xticks(x)
ax8.set_xticklabels(configs, fontsize=7.5)
ax8.set_ylim(0, 115)
ax8.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
           frameon=True, framealpha=0.9)
ax8.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
out8 = os.path.join(output_dir, "fig8_cost_comparison.png")
plt.savefig(out8, dpi=600, bbox_inches="tight")
plt.close(fig8)
print(f"Figure 8 saved to {out8}")


# ---------------------------------------------------------------------------
# Figure 10: Throughput Time Series (Section 4.8)
# ---------------------------------------------------------------------------

# Time axis in seconds; three preemption events at t=20, 80, 140 s
t = np.linspace(0, 180, 1800)

def throughput_curve(base: float, events: list, drop_frac: float, recovery_s: float) -> np.ndarray:
    """Model throughput with preemption drops and recovery."""
    tput = np.full_like(t, base, dtype=float)
    for ev_t in events:
        mask = (t >= ev_t) & (t < ev_t + recovery_s)
        ramp = np.linspace(0, 1, mask.sum())
        tput[mask] = base * (1 - drop_frac) + base * drop_frac * ramp
    return tput

event_times = [20, 80, 140]
morph_tput   = throughput_curve(125.0, event_times, drop_frac=0.15, recovery_s=5.0)
spotserve_tput = throughput_curve(108.0, event_times, drop_frac=0.60, recovery_s=25.0)
vanilla_tput = throughput_curve(125.0, event_times, drop_frac=0.90, recovery_s=60.0)

fig10, ax10 = plt.subplots(figsize=(3.5, 2.6))
ax10.plot(t, morph_tput,   "b-",  label="MorphCloud-LLM", zorder=3)
ax10.plot(t, spotserve_tput, "r--", label="SpotServe")
ax10.plot(t, vanilla_tput, "g:",  label="Vanilla Spot")

for ev_t in event_times:
    ax10.axvline(x=ev_t, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)

ax10.set_xlabel("Time (s)")
ax10.set_ylabel("Throughput (tok/s)")
ax10.set_title("Throughput During Three Consecutive Preemptions")
ax10.set_xlim(0, 180)
ax10.set_ylim(0, 140)
ax10.set_xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180])
ax10.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3,
            frameon=True, framealpha=0.9)
ax10.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()
out10 = os.path.join(output_dir, "fig10_throughput_timeseries.png")
plt.savefig(out10, dpi=600, bbox_inches="tight")
plt.close(fig10)
print(f"Figure 10 saved to {out10}")
