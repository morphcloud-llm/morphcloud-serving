# MorphCloud-LLM: Elastic Spot-Instance-Aware LLM Serving with Transparent Preemption Recovery and Speculative Decoding Continuity

[![Electronics](https://img.shields.io/badge/Published-Electronics%20(MDPI)-blue)](https://www.mdpi.com/2079-9292/15/17/3865)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Felectronics15173865-blue)](https://doi.org/10.3390/electronics15173865)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://www.python.org/)

Research artifact accompanying the published article **MorphCloud-LLM: Elastic Spot-Instance-Aware LLM Serving with Transparent Preemption Recovery and Speculative Decoding Continuity**, published in *Electronics* (2026), Volume 15, Issue 17, Article 3865.

**Paper:** https://www.mdpi.com/2079-9292/15/17/3865  
**DOI:** https://doi.org/10.3390/electronics15173865

## What is in this repository

This repository intentionally separates three artifact types:

1. **Executable reference components** for checkpoint bookkeeping, preemption prediction, orchestration logic, and speculative-continuity state management.
2. **Manuscript-value validation/plotting scripts** that encode values reported in the paper and verify that tables/figures are internally consistent. These scripts do not re-create missing raw measurements.
3. **A manuscript-parameterized synthetic simulator** for exercising event schedules and recovery metrics when the original empirical trace files are unavailable. Simulator output must not be represented as the original experimental data.

The GPU-backed vLLM data plane, original anonymized event traces, held-out predictor outputs, trained predictor artifact, and cloud-account-specific infrastructure are not contained in the supplied archive. They must be added from the original experiment artifacts before claiming full end-to-end reproduction.

## Paper summary

MorphCloud-LLM combines:

- **Asynchronous incremental KV-cache checkpointing** with delta streaming and version-vector consistency.
- **Gradient-boosted preemption prediction** with a 30-second horizon and 89% recall reported in the manuscript.
- **Speculative decoding continuity** that buffers draft tokens during migration and verifies them before user-visible release.

## Repository structure

```text
morphcloud-serving/
├── morphcloud/
│   ├── checkpointing/        # Incremental checkpoint/reference logic
│   ├── prediction/           # XGBoost predictor and request-state classifier
│   ├── speculative/          # Continuity state and verification logic
│   ├── orchestrator/         # Routing and migration coordination reference logic
│   ├── serving/              # Control-plane API scaffold
│   └── utils/
├── evaluation/
│   ├── figures/              # Manuscript-value plotting/validation scripts
│   ├── tables/               # Manuscript table validation
│   ├── ablation/             # Manuscript ablation validation
│   └── results/              # Generated outputs (ignored except .gitkeep)
├── scripts/
│   ├── train_predictor.py    # Train from supplied empirical telemetry
│   ├── simulate_preemption.py# Clearly labeled synthetic simulator
│   └── run_evaluation.py     # Run manuscript-value validations
├── data/
│   ├── traces/               # Trace format/release notes
│   └── predictions/          # Held-out prediction release notes
├── configs/                  # Deployment notes
├── docker/                   # Runnable local control-plane image/Compose setup
├── helm/                     # Minimal control-plane Helm chart + paper-aligned reference values
├── .github/workflows/        # CI tests
└── tests/
```

## System versions reported in the manuscript

| Component | Version / configuration |
|---|---|
| Python | 3.11 |
| PyTorch | 2.3 |
| CUDA | 12.4 environment reported in the paper |
| XGBoost | 2.0.3 |
| scikit-learn | 1.4 |
| boto3 | 1.34 |
| google-cloud-storage | 2.16 |
| vLLM | 0.4.2 |
| Kubernetes | 1.29 |

Paper hardware configuration:

- AWS primary: 8× NVIDIA A100-80GB (`p4d.24xlarge`)
- AWS fallback: 4× NVIDIA A10G (`g5.xlarge`, 96 GB total VRAM)
- GCP primary: 8× NVIDIA A100-80GB (`a2-highgpu-8g`)
- GCP fallback: 8× NVIDIA T4 (`n1-standard-8` + T4 accelerator, 128 GB total VRAM)

## Installation

```bash
git clone https://github.com/morphcloud-llm/morphcloud-serving.git
cd morphcloud-serving
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pinned research stack is GPU-heavy. For CI/unit tests, the workflow installs a smaller CPU-compatible subset.

## Train the preemption predictor

Training requires the original telemetry CSVs described in `data/traces/README.md`. The script now fails explicitly when those files are absent; it does **not** fabricate a replacement dataset.

```bash
python scripts/train_predictor.py \
  --data-dir data/traces/raw \
  --regions us-east-1 us-west-2 us-central1 europe-west1 \
  --horizon 30 \
  --alpha 3.5 \
  --output configs/predictor_weights.pkl \
  --metrics-output evaluation/results/predictor_metrics.json
```

## Launch the control-plane scaffold

```bash
python -m morphcloud.serving.server \
  --model meta-llama/Llama-2-70b-hf \
  --predictor-weights configs/predictor_weights.pkl \
  --checkpoint-interval 16 \
  --checkpoint-backend s3 \
  --s3-bucket your-kv-cache-bucket \
  --fallback-nodes 4 \
  --preemption-threshold 0.7
```

`/health` reports control-plane status. `/generate` deliberately returns HTTP 501 until the experiment's GPU/vLLM data-plane adapter is attached; the repository does not pretend that the missing adapter is functional.

For a local containerized control-plane check:

```bash
docker compose -f docker/docker-compose-local.yml up morphcloud-server
```

## Validate manuscript tables and figures

```bash
python scripts/run_evaluation.py --output-dir evaluation/results
```

This generates manuscript-value validation outputs in the directory you specify. It does **not** claim to recompute the paper from missing raw artifacts.

Individual examples:

```bash
python evaluation/tables/table5_main_results.py
python evaluation/ablation/ablation_study.py
python evaluation/figures/fig9_kv_recovery.py
```

You can direct individual validation scripts with:

```bash
MORPHCLOUD_OUTPUT_DIR=evaluation/results python evaluation/figures/fig9_kv_recovery.py
```

## Run the synthetic event simulator

```bash
python scripts/simulate_preemption.py \
  --n-events 521 \
  --n-runs 1 \
  --platforms aws gcp \
  --output-dir evaluation/results/synthetic-simulation
```

The simulator uses manuscript-reported event counts/means and parametric approximations when the empirical CDFs are unavailable. Its outputs are synthetic and are labeled accordingly.

## Tests

```bash
pytest -q
```

CI runs these tests on every push and pull request.

## Data/artifact release status

The paper states that anonymized event logs and held-out XGBoost test predictions will be included in the public repository. Those original artifacts were not present in the ZIP supplied for this repository preparation. This repository therefore includes release locations and format notes but does not invent the missing files:

- `data/traces/anonymized/` — intended for anonymized released traces/event logs
- `data/predictions/` — intended for held-out prediction outputs
- `configs/predictor_weights.pkl` — generated locally from approved telemetry and Git-ignored by default; add it deliberately only if the trained artifact is approved for public release

Raw/private telemetry should be kept under `data/traces/raw/`, which is excluded from Git.

## Datasets used for workload characterization

| Dataset | URL | License noted in the manuscript |
|---|---|---|
| ShareGPT | https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered | See dataset card |
| LMSYS-Chat-1M | https://huggingface.co/datasets/lmsys/lmsys-chat-1m | Non-commercial research terms described in the manuscript |

## Citation

## Citation

If you use MorphCloud-LLM, please cite the published article:

> Jari, H. MorphCloud-LLM: Elastic Spot-Instance-Aware LLM Serving with Transparent Preemption Recovery and Speculative Decoding Continuity. *Electronics* **2026**, *15*(17), 3865. https://doi.org/10.3390/electronics15173865

```bibtex
@article{Jari2026MorphCloudLLM,
  author  = {Jari, Hassan},
  title   = {MorphCloud-LLM: Elastic Spot-Instance-Aware LLM Serving with Transparent Preemption Recovery and Speculative Decoding Continuity},
  journal = {Electronics},
  year    = {2026},
  volume  = {15},
  number  = {17},
  pages   = {3865},
  doi     = {10.3390/electronics15173865},
  url     = {https://www.mdpi.com/2079-9292/15/17/3865}
}
```

## License

MIT License. See [LICENSE](LICENSE).
