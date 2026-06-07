from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as a plain script: `python scripts/reproduce_all.py ...`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from dynaflow.train import train_one_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scratch DynaFlow reproduction on all datasets and anomaly ratios.")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--strict-deterministic", action="store_true")
    args = parser.parse_args()

    datasets = [
        "uci_messages",
        "digg",
        "email_dnc",
        "bitcoin_alpha",
        "bitcoin_otc",
        "topology",
    ]
    ratios = [0.01, 0.05, 0.10]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for ds in datasets:
        for r in ratios:
            run_args = argparse.Namespace(
                dataset=ds,
                data_path=str(args.processed_dir / f"{ds}.csv"),
                out_json=str(args.out_dir / f"{ds}_r{int(r*100):02d}.json"),
                seed=args.seed,
                train_ratio=0.5,
                val_ratio=0.1,
                anomaly_ratio=r,
                window=5,
                hop=1,
                hidden_dim=32,
                gnn_layers=4,
                temporal_hidden=256,
                top_k=20,
                beta=0.7,
                lr=1e-4,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=10,
                grad_clip=5.0,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                samples_cache="",
                cpu=args.cpu,
                no_progress=args.no_progress,
                deterministic=args.deterministic,
                strict_deterministic=args.strict_deterministic,
            )
            print(f"\\n=== Running {ds} @ anomaly={r:.2f} ===")
            metrics = train_one_run(run_args)
            out_path = Path(run_args.out_json)
            out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            all_rows.append(metrics)

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    print(f"Saved summary -> {summary_path}")


if __name__ == "__main__":
    main()
