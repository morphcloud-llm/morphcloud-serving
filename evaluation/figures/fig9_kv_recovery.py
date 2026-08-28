"""
Figure 9: KV-Cache Recovery Time vs Cache Size.

Renders/validates the manuscript values for Figure 9 of MorphCloud-LLM.
Data sourced from Table 8 (exact values used; no interpolation).

Three curves:
  1. MorphCloud-LLM (incremental streaming)
  2. SpotServe (synchronous full-state migration)
  3. Full Checkpointing (synchronous full-state)

Note on recovery time definitions (Section 4.7):
  The values in Table 8 measure ONLY the KV-cache delta streaming and
  reconstruction phase, corresponding to the "KV-Cache Delta Streaming &
  Reconstruction" sub-row (870 ms) of Table 11. They do NOT represent
  total end-to-end migration latency (1390 ms, Table 11).

Output: PNG at 600 dpi, sized for IEEE two-column format.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# IEEE publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

# ---------------------------------------------------------------------------
# Table 8 data (exact values from the paper)
# ---------------------------------------------------------------------------

cache_sizes_gb = [2, 4, 8, 16, 32, 64]

# MorphCloud-LLM: mean ± std (Table 8)
morph_mean  = [0.12, 0.19, 0.34, 0.58, 0.87, 1.42]
morph_std   = [0.01, 0.02, 0.03, 0.04, 0.06, 0.09]

# SpotServe (Table 8)
spotserve   = [0.35, 0.62, 1.10, 2.00, 3.80, 7.20]

# Full Checkpointing (Table 8)
full_ckpt   = [0.80, 1.60, 3.20, 6.10, 12.40, 25.10]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(3.5, 2.6))

x = np.array(cache_sizes_gb)
morph_mean_arr = np.array(morph_mean)
morph_std_arr  = np.array(morph_std)

ax.plot(x, morph_mean_arr, "b-o", label="MorphCloud-LLM", zorder=3)
ax.fill_between(
    x,
    morph_mean_arr - morph_std_arr,
    morph_mean_arr + morph_std_arr,
    alpha=0.15, color="blue",
)

ax.plot(x, spotserve, "r--s", label="SpotServe")
ax.plot(x, full_ckpt,  "g:^", label="Full Checkpoint")

# Reference line: sub-second boundary
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7,
           label="1.0 s reference")

ax.set_xlabel("KV-Cache Size (GB)")
ax.set_ylabel("Recovery Time (s)")
ax.set_title("KV-Cache Recovery Time vs Cache Size")
ax.set_xticks(cache_sizes_gb)
ax.set_xscale("log", base=2)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
ax.set_yscale("log")
ax.set_ylim(0.05, 40)

# Legend outside plot, below x-axis label
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.28),
    ncol=2,
    frameon=True,
    framealpha=0.9,
)

ax.grid(True, which="major", linestyle="--", linewidth=0.4, alpha=0.5)
plt.tight_layout()

output_dir = os.getenv("MORPHCLOUD_OUTPUT_DIR", "evaluation/results")
os.makedirs(output_dir, exist_ok=True)
out_path = os.path.join(output_dir, "fig9_kv_recovery.png")
plt.savefig(out_path, dpi=600, bbox_inches="tight")
plt.close()
print(f"Figure 9 saved to {out_path}")
