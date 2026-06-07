# DynaFlow Reproduction (English)

This is a scratch implementation for dynamic graph edge anomaly detection inspired by DynaFlow (for research reproduction, not an official line-by-line clone).

## Documentation
- Main README (short): [README.md](./README.md)
- Chinese version: [README_zh.md](./README_zh.md)

## What This Repo Provides
- Dataset download and preprocessing for 6 public dynamic graph datasets
- Time-based train/test split with controllable anomaly injection
- DynaFlow-style pipeline:
  - Spectral enhancement
  - Graph structure aggregation
  - Temporal modeling for edge anomaly scoring
- JSON result export with AUC/AP/F1 and run metadata

## Project Layout
```text
dynaflow_scratch/
  scripts/
    download_datasets.sh
  src/dynaflow/
    preprocess.py
    data.py
    model.py
    train.py
  data/
    raw/
    processed/
  results/
```

## Quick Start

### 1) Install dependencies
```bash
cd dynaflow_scratch
pip install -r requirements.txt
```

### 2) Download raw datasets
```bash
./scripts/download_datasets.sh
```

### 3) Preprocess to unified CSV format
```bash
PYTHONPATH=src python3 -m dynaflow.preprocess --raw-dir data/raw --out-dir data/processed
```

### 4) Train (minimal command)
```bash
PYTHONPATH=src python3 -m dynaflow.train --dataset digg --anomaly-ratio 0.10
```

## Common Training Examples

### Run in module mode (recommended)
```bash
PYTHONPATH=src python3 -m dynaflow.train \
  --dataset uci_messages \
  --anomaly-ratio 0.05
```

### Run as a script
```bash
cd src/dynaflow
python train.py --dataset digg --anomaly-ratio 0.10
```

### Use sample cache for faster reruns
```bash
PYTHONPATH=src python3 -m dynaflow.train \
  --dataset digg \
  --anomaly-ratio 0.10 \
  --samples-cache data/processed/cache/digg_r10_cache.npz
```

## Key Arguments
- `--dataset`: `uci_messages | digg | email_dnc | bitcoin_alpha | bitcoin_otc | topology`
- `--anomaly-ratio`: anomaly ratio in test split (e.g., `0.01`, `0.05`, `0.10`)
- Negative sampling uses the built-in StrGNN-style context-dependent sampler.
- `--no-use-spectral`: w/o spectral ablation
- `--no-use-low-pass`: keep eigendecomposition but remove the `exp(-beta*lambda)` low-pass filter
- `--temporal-cell`: `garu | gru | none` for GARU vs GRU and w/o GARU
- `--hop`: subgraph radius; `--hop 0` is the w/o subgraph setting
- `--window`, `--beta`: temporal window and low-pass strength for sensitivity scans
- `--samples-cache`: save/load sampled train/test edges
- `--batch-size`, `--epochs`, `--patience`: core training controls

## Ablations

Run the bundled ablation suite for w/o spectral, w/o low-pass, w/o GARU, GARU vs GRU, w/o subgraph, and h/w/beta scans:

```bash
cd dynaflow_scratch
PYTHONPATH=src python3 scripts/run_ablation.py \
  --datasets uci_messages digg email_dnc bitcoin_alpha bitcoin_otc topology \
  --anomaly-ratio 0.05 \
  --epochs 50 \
  --batch-size 32
```

Outputs are written to `results/ablations/`, including per-run JSON files, checkpoints, `ablation_summary.json`, and `ablation_summary.csv`.

## Output
- Default output path: `results/{dataset}_rXX.json`
- Metrics include:
  - `auc`, `ap`
  - `precision`, `f1`
  - `tpr`, `tnr`
  - `threshold`
  - sample counts (`n_train`, `n_val`, `n_test`)

## Troubleshooting
- Import error when running script directly:
  - Prefer module mode: `PYTHONPATH=src python3 -m dynaflow.train ...`
- Long wait before the first epoch:
  - Usually sample construction + anomaly injection
  - Use `--samples-cache ...` for faster reruns
- No progress bars:
  - Check whether `--no-progress` is enabled

## Disclaimer
- This repository is intended for research reproduction and experimentation.
- Results may differ from official paper/release implementations.
