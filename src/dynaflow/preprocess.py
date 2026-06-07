from __future__ import annotations

import argparse
import gzip
import json
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SNAPSHOT_BINS = {
    "uci_messages": 190,
    "digg": 16,
    "email_dnc": 20,
    "bitcoin_alpha": 21,
    "bitcoin_otc": 63,
    "topology": 63,
}

# Dataset-specific timestamp binning mode.
# `uniform`: equal-width bins over [min(ts_raw), max(ts_raw)].
# `quantile`: approximately equal-number bins, better for highly skewed timestamp distributions.
BINNING_MODE = {
    "email_dnc": "quantile",
    "topology": "quantile",
}


def _assign_timestamp_bins(ts_raw: np.ndarray, n_bins: int, mode: str) -> np.ndarray:
    ts_min = float(np.min(ts_raw))
    ts_max = float(np.max(ts_raw))
    if ts_max == ts_min:
        return np.zeros_like(ts_raw, dtype=np.int64)

    if mode == "quantile":
        q = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(ts_raw, q)
        # Keep edges monotonic even when repeated quantiles appear.
        edges = np.maximum.accumulate(edges)
        return np.clip(np.digitize(ts_raw, edges[1:-1], right=False), 0, n_bins - 1).astype(np.int64)

    # Default: uniform binning.
    edges = np.linspace(ts_min, ts_max + 1e-9, n_bins + 1)
    return np.clip(np.digitize(ts_raw, edges[1:-1], right=False), 0, n_bins - 1).astype(np.int64)


def _read_gzip_csv(path: Path, sep: str, cols: list[str]) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        df = pd.read_csv(f, sep=sep, header=None, names=cols)
    return df


def _read_zip_edges(path: Path, member: str, names: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        with zf.open(member) as f:
            rows = []
            for line in f:
                s = line.decode("utf-8", errors="ignore").strip().lstrip("\ufeff")
                if not s or s.startswith("%"):
                    continue
                parts = s.replace(",", " ").split()
                rows.append(parts[: len(names)])
    df = pd.DataFrame(rows, columns=names)
    return df


def load_dataset(name: str, raw_dir: Path) -> pd.DataFrame:
    if name == "bitcoin_alpha":
        df = _read_gzip_csv(raw_dir / "bitcoin_alpha.csv.gz", sep=",", cols=["src", "dst", "weight", "ts_raw"])
    elif name == "bitcoin_otc":
        df = _read_gzip_csv(raw_dir / "bitcoin_otc.csv.gz", sep=",", cols=["src", "dst", "weight", "ts_raw"])
    elif name == "uci_messages":
        df = _read_gzip_csv(raw_dir / "uci_messages.txt.gz", sep=r"\s+", cols=["src", "dst", "ts_raw"])
        df["weight"] = 1
    elif name == "digg":
        df = _read_zip_edges(raw_dir / "digg.zip", "ia-digg-reply.edges", names=["src", "dst", "weight", "ts_raw"])
    elif name == "email_dnc":
        df = _read_zip_edges(raw_dir / "email_dnc.zip", "email-dnc.edges", names=["src", "dst", "ts_raw"])
        df["weight"] = 1
    elif name == "topology":
        df = _read_zip_edges(raw_dir / "topology.zip", "tech-as-topology.edges", names=["src", "dst", "weight", "ts_raw"])
    else:
        raise ValueError(f"Unknown dataset: {name}")

    df = df[["src", "dst", "ts_raw", "weight"]].copy()
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    df["ts_raw"] = pd.to_numeric(df["ts_raw"], errors="coerce")
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    df = df.dropna(subset=["src", "dst", "ts_raw"]).reset_index(drop=True)
    df = df[df["src"] != df["dst"]].reset_index(drop=True)

    # Map arbitrary node IDs to contiguous IDs for faster tensor indexing.
    all_nodes = pd.Index(pd.concat([df["src"], df["dst"]], ignore_index=True).unique())
    node_map = pd.Series(np.arange(len(all_nodes), dtype=np.int64), index=all_nodes)
    df["src"] = node_map.loc[df["src"]].to_numpy()
    df["dst"] = node_map.loc[df["dst"]].to_numpy()

    n_bins = SNAPSHOT_BINS[name]
    mode = BINNING_MODE.get(name, "uniform")
    df["ts"] = _assign_timestamp_bins(df["ts_raw"].to_numpy(), n_bins=n_bins, mode=mode)

    df = df.sort_values("ts").reset_index(drop=True)
    return df


def summarize(df: pd.DataFrame, n_bins: int) -> dict:
    n_nodes = int(pd.concat([df["src"], df["dst"]], ignore_index=True).nunique())
    n_edges = int(len(df))
    avg_deg = float((2 * n_edges) / max(1, n_nodes))
    return {
        "num_edges": n_edges,
        "num_nodes": n_nodes,
        "num_timestamps": int(n_bins),
        "avg_degree": round(avg_deg, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess dynamic graph datasets for DynaFlow reproduction.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--datasets", nargs="*", default=list(SNAPSHOT_BINS.keys()))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_stats: dict[str, dict] = {}

    for name in args.datasets:
        df = load_dataset(name, args.raw_dir)
        out_path = args.out_dir / f"{name}.csv"
        df.to_csv(out_path, index=False)
        stats = summarize(df, SNAPSHOT_BINS[name])
        all_stats[name] = stats
        print(f"[{name}] -> {out_path} | {stats}")

    stats_path = args.out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"Saved dataset summary: {stats_path}")


if __name__ == "__main__":
    main()
