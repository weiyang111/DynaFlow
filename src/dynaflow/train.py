from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, roc_auc_score
from tqdm import tqdm

# Allow running as a plain script: `python train.py ...`
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    __package__ = "dynaflow"

from .data import (
    batchify_cached,
    batchify,
    build_snapshot_graphs,
    load_processed,
    load_samples_cache,
    make_samples,
    precompute_sample_sequences,
    save_samples_cache,
)
from .model import DynaFlow

DATASET_ALIASES = {
    "uci": "uci_messages",
    "dnc": "email_dnc",
    "alpha": "bitcoin_alpha",
    "otc": "bitcoin_otc",
}


def set_seed(seed: int, deterministic: bool = False, strict: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=(not strict))
        except TypeError:
            if strict:
                torch.use_deterministic_algorithms(True)
            else:
                try:
                    torch.use_deterministic_algorithms(True)
                except Exception:
                    pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    candidates = np.unique(np.concatenate([np.array([0.1, 0.2, 0.3, 0.4, 0.5]), y_prob]))
    best_thr = 0.5
    best_f1 = -1.0
    for thr in candidates:
        y_pred = (y_prob >= thr).astype(np.int64)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def eval_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "ap": float(average_precision_score(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tpr": float(tp / max(1, tp + fn)),
        "tnr": float(tn / max(1, tn + fp)),
        "threshold": float(threshold),
    }


def iterate_minibatches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def samples_fingerprint(samples) -> str:
    if not samples:
        return "empty"
    arr = np.array([[s.u, s.v, s.t, s.y] for s in samples], dtype=np.int64)
    return hashlib.sha1(arr.tobytes()).hexdigest()


def split_train_val(samples, val_ratio: float, mode: str, seed: int):
    """Split generated training samples into train/validation samples.

    mode='tail' preserves the original implementation: use the last val_ratio
    fraction after sample generation/shuffle. mode='stratified' draws validation
    samples independently within each label using only the training split labels;
    no test samples or test labels are consulted.
    """
    if not samples:
        return [], []
    if mode == "tail":
        val_cut = int(len(samples) * (1 - val_ratio))
        train_split = list(samples[:val_cut])
        val_split = list(samples[val_cut:])
    elif mode == "random":
        shuffled = list(samples)
        random.Random(seed + 707).shuffle(shuffled)
        val_cut = int(len(shuffled) * (1 - val_ratio))
        train_split = list(shuffled[:val_cut])
        val_split = list(shuffled[val_cut:])
    elif mode == "stratified":
        rng = random.Random(seed + 707)
        by_label = {0: [], 1: []}
        other = []
        for s in samples:
            if int(s.y) in by_label:
                by_label[int(s.y)].append(s)
            else:
                other.append(s)
        train_split = []
        val_split = []
        for label, group in by_label.items():
            group = list(group)
            rng.shuffle(group)
            n_val = int(round(len(group) * val_ratio))
            if len(group) > 1:
                n_val = min(max(1, n_val), len(group) - 1)
            else:
                n_val = 0
            val_split.extend(group[:n_val])
            train_split.extend(group[n_val:])
        if other:
            rng.shuffle(other)
            n_val = int(round(len(other) * val_ratio))
            val_split.extend(other[:n_val])
            train_split.extend(other[n_val:])
        rng.shuffle(train_split)
        rng.shuffle(val_split)
    else:
        raise ValueError(f"Unsupported val_split_mode={mode!r}; expected 'tail', 'random', or 'stratified'.")

    if not val_split and len(train_split) > 1:
        val_split = [train_split[-1]]
        train_split = train_split[:-1]
    return train_split, val_split


def _torch_load_cpu(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_subgraph_cache_path(args: argparse.Namespace) -> Path:
    if args.subgraph_cache:
        return Path(args.subgraph_cache).expanduser().resolve()

    ratio_tag = f"{int(round(float(args.anomaly_ratio) * 100)):02d}"
    tr_tag = f"{float(args.train_ratio):.2f}".replace(".", "p")
    name = (
        f"{args.dataset}_subgraph_w{args.window}_h{args.hop}_r{ratio_tag}_"
        f"tr{tr_tag}_strgnn_context_seed{args.seed}.pt"
    )
    return Path(args.data_path).resolve().parent / "cache" / "subgraphs" / name


def resolve_samples_cache_path(args: argparse.Namespace) -> str:
    # Allow explicit disable via --samples-cache none/off/disable
    if isinstance(args.samples_cache, str) and args.samples_cache.strip().lower() in {"none", "off", "disable"}:
        return ""
    if args.samples_cache:
        return str(Path(args.samples_cache).expanduser().resolve())

    ratio_tag = f"{int(round(float(args.anomaly_ratio) * 100)):02d}"
    tr_tag = f"{float(args.train_ratio):.2f}".replace(".", "p")
    name = (
        f"{args.dataset}_samples_w{args.window}_r{ratio_tag}_tr{tr_tag}_"
        f"strgnn_context_seed{args.seed}.npz"
    )
    p = Path(args.data_path).resolve().parent / "cache" / name
    return str(p)


def resolve_history_path(args: argparse.Namespace) -> str:
    if isinstance(args.history_json, str) and args.history_json.strip().lower() in {"none", "off", "disable"}:
        return ""
    if args.history_json:
        return str(Path(args.history_json).expanduser().resolve())

    ratio_tag = f"{int(round(float(args.anomaly_ratio) * 100)):02d}"
    project_root = Path(__file__).resolve().parents[2]
    p = project_root / "results" / f"{args.dataset}_r{ratio_tag}_history.jsonl"
    return str(p)


def balance_train_test_1to1(train_samples, test_samples, seed: int):
    """
    Align train/test counts to 1:1 without discarding any existing edge samples.
    The smaller side is upsampled with replacement.
    """
    if len(train_samples) == 0 or len(test_samples) == 0:
        return train_samples, test_samples
    n_train = len(train_samples)
    n_test = len(test_samples)
    if n_train == n_test:
        return train_samples, test_samples

    rng = random.Random(seed + 303)
    train_out = list(train_samples)
    test_out = list(test_samples)
    target = max(n_train, n_test)

def train_one_run(args: argparse.Namespace) -> dict:
    defaults = {
        "cpu": False,
        "seed": 42,
        "train_ratio": 0.5,
        "val_ratio": 0.1,
        "val_split_seed": -1,
        "val_split_mode": "tail",
        "anomaly_ratio": 0.05,
        "window": 5,
        "hop": 1,
        "hidden_dim": 32,
        "gnn_layers": 4,
        "temporal_hidden": 256,
        "top_k": 20,
        "beta": 0.7,
        "use_spectral": True,
        "use_low_pass": True,
        "temporal_cell": "garu",
        "lr": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "grad_clip": 5.0,
        "max_train_samples": 0,
        "max_test_samples": 0,
        "build_train_samples": 0,
        "build_test_samples": 0,
        "samples_cache": "",
        "history_json": "",
        "no_progress": False,
        "deterministic": False,
        "strict_deterministic": False,
        "strict_train_test_1to1": False,
        "precompute_subgraphs": True,
        "subgraph_cache": "",
        "reuse_subgraph_cache": True,
        "amp": True,
        "amp_dtype": "bf16",
        "checkpoint": "",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    det = bool(args.deterministic or args.strict_deterministic)
    set_seed(args.seed, deterministic=det, strict=bool(args.strict_deterministic))

    df = load_processed(Path(args.data_path))
    snapshots, edge_sets, n_nodes = build_snapshot_graphs(df)
    args.samples_cache = resolve_samples_cache_path(args)
    args.history_json = resolve_history_path(args)

    cache_meta = {
        "dataset": args.dataset,
        "data_path": str(Path(args.data_path).resolve()),
        "train_snapshot_ratio": float(args.train_ratio),
        "anomaly_ratio": float(args.anomaly_ratio),
        "window": int(args.window),
        "seed": int(args.seed),
        "build_train_samples": int(args.build_train_samples),
        "build_test_samples": int(args.build_test_samples),
        "sampler_version": "strgnn_context_v1",
    }

    if args.samples_cache and Path(args.samples_cache).exists():
        train_samples, test_samples, loaded_meta = load_samples_cache(Path(args.samples_cache))
        if loaded_meta != cache_meta:
            print("Sample cache exists but metadata mismatches current run config; rebuilding cache.")
            train_samples, test_samples = make_samples(
                df,
                snapshots,
                edge_sets,
                n_nodes=n_nodes,
                anomaly_ratio=args.anomaly_ratio,
                seed=args.seed,
                window=args.window,
                train_snapshot_ratio=args.train_ratio,
                show_progress=(not args.no_progress),
                train_build_cap=int(args.build_train_samples),
                test_build_cap=int(args.build_test_samples),
            )
            save_samples_cache(Path(args.samples_cache), train_samples, test_samples, meta=cache_meta)
            print(f"Saved refreshed sample cache to {args.samples_cache}")
        else:
            print(f"Loaded cached samples from {args.samples_cache}")
    else:
        print("Preparing samples (StrGNN-style snapshot split and context negative sampling)...")
        train_samples, test_samples = make_samples(
            df,
            snapshots,
            edge_sets,
            n_nodes=n_nodes,
            anomaly_ratio=args.anomaly_ratio,
            seed=args.seed,
            window=args.window,
            train_snapshot_ratio=args.train_ratio,
            show_progress=(not args.no_progress),
            train_build_cap=int(args.build_train_samples),
            test_build_cap=int(args.build_test_samples),
        )
        if args.samples_cache:
            save_samples_cache(Path(args.samples_cache), train_samples, test_samples, meta=cache_meta)
            print(f"Saved sample cache to {args.samples_cache}")

    # Keep runtime manageable for local reproduction while preserving protocol.
    n_train_before_cap = len(train_samples)
    n_test_before_cap = len(test_samples)
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]
    n_train_after_cap = len(train_samples)
    n_test_after_cap = len(test_samples)

    if args.strict_train_test_1to1:
        train_samples, test_samples = balance_train_test_1to1(train_samples, test_samples, seed=args.seed)
        print(
            "Applied strict train/test 1:1 balance (no edge dropped; smaller side upsampled) "
            f"-> n_train={len(train_samples)}, n_test={len(test_samples)}"
        )

    if not train_samples:
        raise RuntimeError("No training samples were generated. Check train_ratio/window and dataset.")
    if not test_samples:
        raise RuntimeError("No test samples were generated. Check train_ratio/window and anomaly ratio.")

    val_split_seed = int(args.val_split_seed if args.val_split_seed >= 0 else args.seed)
    train_split, val_split = split_train_val(
        train_samples,
        val_ratio=float(args.val_ratio),
        mode=str(args.val_split_mode),
        seed=val_split_seed,
    )
    if not train_split:
        raise RuntimeError("No training samples were generated. Check window size and sample split.")

    use_cache = bool(args.precompute_subgraphs)
    train_cache_x = val_cache_x = test_cache_x = None
    train_cache_y = val_cache_y = test_cache_y = None
    if use_cache:
        subgraph_cache_path = resolve_subgraph_cache_path(args)
        subgraph_meta = {
            "dataset": args.dataset,
            "data_path": str(Path(args.data_path).resolve()),
            "window": int(args.window),
            "hop": int(args.hop),
            "train_ratio": float(args.train_ratio),
            "anomaly_ratio": float(args.anomaly_ratio),
            "seed": int(args.seed),
            "sampler_version": "strgnn_context_v1",
            "max_train_samples": int(args.max_train_samples),
            "max_test_samples": int(args.max_test_samples),
            "build_train_samples": int(args.build_train_samples),
            "build_test_samples": int(args.build_test_samples),
            "strict_train_test_1to1": bool(args.strict_train_test_1to1),
            "val_split_mode": str(args.val_split_mode),
            "train_fp": samples_fingerprint(train_split),
            "val_fp": samples_fingerprint(val_split),
            "test_fp": samples_fingerprint(test_samples),
        }

        loaded_ok = False
        if args.reuse_subgraph_cache and subgraph_cache_path.exists():
            try:
                payload = _torch_load_cpu(subgraph_cache_path)
                if payload.get("meta", {}) == subgraph_meta:
                    train_cache_x = payload["train_x"]
                    train_cache_y = payload["train_y"]
                    val_cache_x = payload["val_x"]
                    val_cache_y = payload["val_y"]
                    test_cache_x = payload["test_x"]
                    test_cache_y = payload["test_y"]
                    loaded_ok = True
                    print(f"Loaded precomputed subgraph cache from {subgraph_cache_path}")
                else:
                    print("Subgraph cache metadata mismatches current run; rebuilding.")
            except Exception as e:
                print(f"Failed to load subgraph cache ({e}); rebuilding.")

        if not loaded_ok:
            print("Precomputing subgraph sequences once (no per-epoch BFS/encoding)...")
            train_cache_x, train_cache_y = precompute_sample_sequences(
                train_split,
                snapshots,
                args.window,
                args.hop,
                show_progress=(not args.no_progress),
                desc="Precompute train",
            )
            val_cache_x, val_cache_y = precompute_sample_sequences(
                val_split,
                snapshots,
                args.window,
                args.hop,
                show_progress=(not args.no_progress),
                desc="Precompute val",
            )
            test_cache_x, test_cache_y = precompute_sample_sequences(
                test_samples,
                snapshots,
                args.window,
                args.hop,
                show_progress=(not args.no_progress),
                desc="Precompute test",
            )
            subgraph_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "meta": subgraph_meta,
                    "train_x": train_cache_x,
                    "train_y": train_cache_y,
                    "val_x": val_cache_x,
                    "val_y": val_cache_y,
                    "test_x": test_cache_x,
                    "test_y": test_cache_y,
                },
                subgraph_cache_path,
            )
            print(f"Saved precomputed subgraph cache to {subgraph_cache_path}")

        input_dim = train_cache_x[0][0].x.shape[1]
    else:
        dummy_seq, _ = batchify(train_split[:1], snapshots, args.window, args.hop, device)
        input_dim = dummy_seq[0][0].x.shape[1]

    model = DynaFlow(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        gnn_layers=args.gnn_layers,
        temporal_hidden=args.temporal_hidden,
        top_k=args.top_k,
        beta=args.beta,
        use_spectral=args.use_spectral,
        use_low_pass=args.use_low_pass,
        temporal_cell=args.temporal_cell,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if str(args.amp_dtype).lower() == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    best_val_auc = -math.inf
    best_state = None
    wait = 0
    history_f = None
    if args.history_json:
        hp = Path(args.history_json)
        hp.parent.mkdir(parents=True, exist_ok=True)
        history_f = hp.open("w", encoding="utf-8")

    epoch_iter = range(1, args.epochs + 1)
    if not args.no_progress:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", unit="epoch")

    for epoch in epoch_iter:
        if use_cache:
            train_indices = list(range(len(train_cache_x)))
            random.shuffle(train_indices)
            train_batches = iterate_minibatches(train_indices, args.batch_size)
        else:
            random.shuffle(train_split)
            train_batches = iterate_minibatches(train_split, args.batch_size)
        model.train()
        losses = []
        if not args.no_progress:
            train_batches = tqdm(
                train_batches,
                total=((len(train_cache_x) if use_cache else len(train_split)) + args.batch_size - 1) // args.batch_size,
                desc=f"Train E{epoch}",
                unit="batch",
                leave=False,
            )
        for batch_data in train_batches:
            if use_cache:
                batch_x, batch_y = batchify_cached(train_cache_x, train_cache_y, batch_data, device)
            else:
                batch_x, batch_y = batchify(batch_data, snapshots, args.window, args.hop, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            losses.append(float(loss.item()))
            if not args.no_progress:
                train_batches.set_postfix(loss=f"{losses[-1]:.4f}")

        # Validation
        model.eval()
        val_prob = []
        val_true = []
        with torch.no_grad():
            if use_cache:
                val_batches = iterate_minibatches(list(range(len(val_cache_x))), args.batch_size)
            else:
                val_batches = iterate_minibatches(val_split, args.batch_size)
            if not args.no_progress:
                val_batches = tqdm(
                    val_batches,
                    total=((len(val_cache_x) if use_cache else len(val_split)) + args.batch_size - 1) // args.batch_size,
                    desc=f"Val E{epoch}",
                    unit="batch",
                    leave=False,
                )
            for batch_data in val_batches:
                if use_cache:
                    batch_x, batch_y = batchify_cached(val_cache_x, val_cache_y, batch_data, device)
                else:
                    batch_x, batch_y = batchify(batch_data, snapshots, args.window, args.hop, device)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(batch_x)
                prob = torch.sigmoid(logits).detach().float().cpu().numpy()
                val_prob.append(prob)
                val_true.append(batch_y.detach().cpu().numpy())

        if len(val_prob) == 0:
            val_auc = 0.0
        else:
            yv = np.concatenate(val_true)
            pv = np.concatenate(val_prob)
            val_auc = float(roc_auc_score(yv, pv)) if len(np.unique(yv)) > 1 else 0.0

        print(f"epoch={epoch:03d} train_loss={np.mean(losses):.4f} val_auc={val_auc:.4f}")
        if history_f is not None:
            row = {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "val_auc": float(val_auc),
                "n_train": int(len(train_split)),
                "n_val": int(len(val_split)),
            }
            history_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            history_f.flush()

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Derive threshold from validation split using best model.
    val_prob = []
    val_true = []
    with torch.no_grad():
        if use_cache:
            val_batches = iterate_minibatches(list(range(len(val_cache_x))), args.batch_size)
        else:
            val_batches = iterate_minibatches(val_split, args.batch_size)
        for batch_data in val_batches:
            if use_cache:
                batch_x, batch_y = batchify_cached(val_cache_x, val_cache_y, batch_data, device)
            else:
                batch_x, batch_y = batchify(batch_data, snapshots, args.window, args.hop, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_x)
            prob = torch.sigmoid(logits).detach().float().cpu().numpy()
            val_prob.append(prob)
            val_true.append(batch_y.detach().cpu().numpy())
    yv = np.concatenate(val_true) if val_true else np.array([0, 1], dtype=np.int64)
    pv = np.concatenate(val_prob) if val_prob else np.array([0.0, 1.0], dtype=np.float32)
    threshold = choose_threshold(yv, pv)

    # Test
    model.eval()
    test_prob = []
    test_true = []
    with torch.no_grad():
        if use_cache:
            test_batches = iterate_minibatches(list(range(len(test_cache_x))), args.batch_size)
        else:
            test_batches = iterate_minibatches(test_samples, args.batch_size)
        for batch_data in test_batches:
            if use_cache:
                batch_x, batch_y = batchify_cached(test_cache_x, test_cache_y, batch_data, device)
            else:
                batch_x, batch_y = batchify(batch_data, snapshots, args.window, args.hop, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(batch_x)
            prob = torch.sigmoid(logits).detach().float().cpu().numpy()
            test_prob.append(prob)
            test_true.append(batch_y.detach().cpu().numpy())

    yt = np.concatenate(test_true)
    pt = np.concatenate(test_prob)
    metrics = eval_metrics(yt, pt, threshold=threshold)
    train_labels = np.array([s.y for s in train_split], dtype=np.int64)
    val_labels = yv.astype(np.int64) if len(yv) else np.array([], dtype=np.int64)
    test_labels = yt.astype(np.int64) if len(yt) else np.array([], dtype=np.int64)
    metrics.update(
        {
            "dataset": args.dataset,
            "model": "DynaFlow",
            "task": "binary_dynamic_edge_anomaly_detection",
            "positive_class": "label=1 injected anomaly edge",
            "auc_metric": "sklearn.metrics.roc_auc_score(y_true, sigmoid(logits))",
            "anomaly_ratio": args.anomaly_ratio,
            "actual_test_anomaly_ratio": float(np.mean(yt)) if len(yt) > 0 else float("nan"),
            "strict_train_test_1to1": bool(args.strict_train_test_1to1),
            "n_train_before_cap": n_train_before_cap,
            "n_test_before_cap": n_test_before_cap,
            "n_train_after_cap": n_train_after_cap,
            "n_test_after_cap": n_test_after_cap,
            "n_train": len(train_split),
            "n_val": len(val_split),
            "n_test": len(test_samples),
            "train_positive_count": int(train_labels.sum()) if len(train_labels) else 0,
            "train_negative_count": int(len(train_labels) - train_labels.sum()) if len(train_labels) else 0,
            "val_positive_count": int(val_labels.sum()) if len(val_labels) else 0,
            "val_negative_count": int(len(val_labels) - val_labels.sum()) if len(val_labels) else 0,
            "test_positive_count": int(test_labels.sum()) if len(test_labels) else 0,
            "test_negative_count": int(len(test_labels) - test_labels.sum()) if len(test_labels) else 0,
            "best_val_auc": best_val_auc,
            "history_json": str(args.history_json) if args.history_json else "",
            "run_config": {
                "seed": int(args.seed),
                "deterministic": bool(args.deterministic),
                "strict_deterministic": bool(args.strict_deterministic),
                "train_snapshot_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "val_split_mode": str(args.val_split_mode),
                "val_split_seed": int(val_split_seed),
                "window": int(args.window),
                "hop": int(args.hop),
                "sampler_version": "strgnn_context_v1",
                "max_train_samples": int(args.max_train_samples),
                "max_test_samples": int(args.max_test_samples),
                "build_train_samples": int(args.build_train_samples),
                "build_test_samples": int(args.build_test_samples),
                "lr": float(args.lr),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "patience": int(args.patience),
                "grad_clip": float(args.grad_clip),
                "hidden_dim": int(args.hidden_dim),
                "gnn_layers": int(args.gnn_layers),
                "temporal_hidden": int(args.temporal_hidden),
                "top_k": int(args.top_k),
                "beta": float(args.beta),
                "use_spectral": bool(args.use_spectral),
                "use_low_pass": bool(args.use_low_pass),
                "temporal_cell": str(args.temporal_cell),
                "precompute_subgraphs": bool(args.precompute_subgraphs),
                "amp": bool(args.amp),
                "amp_dtype": str(args.amp_dtype),
            },
            "sample_fingerprints": {
                "train": samples_fingerprint(train_split),
                "val": samples_fingerprint(val_split),
                "test": samples_fingerprint(test_samples),
            },
        }
    )
    if history_f is not None:
        history_f.close()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[2] / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "input_dim": int(input_dim),
                    "hidden_dim": int(args.hidden_dim),
                    "gnn_layers": int(args.gnn_layers),
                    "temporal_hidden": int(args.temporal_hidden),
                    "top_k": int(args.top_k),
                    "beta": float(args.beta),
                    "use_spectral": bool(args.use_spectral),
                    "use_low_pass": bool(args.use_low_pass),
                    "temporal_cell": str(args.temporal_cell),
                },
                "threshold": float(threshold),
                "metrics": metrics,
            },
            checkpoint_path,
        )
        metrics["checkpoint"] = str(checkpoint_path)
        print(f"Saved best model checkpoint to {checkpoint_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate scratch DynaFlow implementation.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--out-json", type=str, default="")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument(
        "--samples-cache",
        type=str,
        default="",
        help="Path to sampled edge cache (.npz). Empty means auto path under data/processed/cache/. Use 'none' to disable.",
    )
    parser.add_argument(
        "--history-json",
        type=str,
        default="",
        help="Path to per-epoch history jsonl. Empty means auto path under results/. Use 'none' to disable.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.5, help="Train snapshot ratio in chronological split (StrGNN-style).")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=-1,
        help="Seed used only for splitting generated training samples into train/validation; -1 reuses --seed. It never sees test samples.",
    )
    parser.add_argument(
        "--val-split-mode",
        type=str,
        default="tail",
        choices=["tail", "random", "stratified"],
        help="How to split generated training samples into train/validation. 'tail' preserves original behavior; 'random' uses --val-split-seed within training samples only; 'stratified' samples validation within each label using training labels only.",
    )
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--gnn-layers", type=int, default=4)
    parser.add_argument("--temporal-hidden", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--beta", type=float, default=0.7)
    parser.add_argument(
        "--use-spectral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable spectral enhancement before graph aggregation. Use --no-use-spectral for w/o spectral.",
    )
    parser.add_argument(
        "--use-low-pass",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable exp(-beta*lambda) low-pass filtering inside spectral enhancement. Use --no-use-low-pass for w/o low-pass.",
    )
    parser.add_argument(
        "--temporal-cell",
        type=str,
        default="garu",
        choices=["garu", "gru", "none"],
        help="Temporal encoder: garu (default), gru (GARU vs GRU), or none (w/o GARU mean pooling).",
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--build-train-samples", type=int, default=0, help="Optional construction-time cap for generated train samples; 0 builds the full split before max-train-samples truncation.")
    parser.add_argument("--build-test-samples", type=int, default=0, help="Optional construction-time cap for generated test samples; 0 builds the full split before max-test-samples truncation.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode (best-effort, may run slower).",
    )
    parser.add_argument(
        "--strict-deterministic",
        action="store_true",
        help="Require deterministic algorithms; may raise runtime errors if unsupported.",
    )
    parser.add_argument(
        "--strict-train-test-1to1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force train/test sample counts to be 1:1 by upsampling the smaller side (no edge dropping).",
    )
    parser.add_argument(
        "--precompute-subgraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Precompute subgraph sequences once before training to avoid per-epoch BFS/encoding.",
    )
    parser.add_argument(
        "--subgraph-cache",
        type=str,
        default="",
        help="Path to on-disk subgraph cache (.pt). If empty, an automatic cache path is used.",
    )
    parser.add_argument(
        "--reuse-subgraph-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse matching on-disk subgraph cache when available.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic mixed precision on CUDA.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP datatype on CUDA.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional path to save the best model checkpoint for downstream plotting/analysis.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    requested_dataset = args.dataset
    args.dataset = DATASET_ALIASES.get(args.dataset, args.dataset)
    if args.dataset != requested_dataset:
        print(f"[info] dataset alias: {requested_dataset} -> {args.dataset}")

    if not args.data_path:
        args.data_path = str(project_root / args.data_dir / f"{args.dataset}.csv")
    else:
        p = Path(args.data_path)
        if not p.is_absolute() and not p.exists():
            args.data_path = str(project_root / p)

    data_path_obj = Path(args.data_path)
    if not data_path_obj.exists():
        data_dir_obj = project_root / args.data_dir
        available = sorted(x.stem for x in data_dir_obj.glob("*.csv")) if data_dir_obj.exists() else []
        raise FileNotFoundError(
            f"data csv not found: {data_path_obj}\n"
            f"requested --dataset={requested_dataset} (resolved: {args.dataset})\n"
            f"available datasets under {data_dir_obj}: {available}"
        )

    if not args.out_json:
        ratio_tag = f"{int(round(args.anomaly_ratio * 100)):02d}"
        args.out_json = str(project_root / args.results_dir / f"{args.dataset}_r{ratio_tag}.json")
    else:
        p = Path(args.out_json)
        if not p.is_absolute():
            args.out_json = str(project_root / p)

    metrics = train_one_run(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
