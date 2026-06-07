from __future__ import annotations

import bisect
import json
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .model import GraphStep


@dataclass
class EdgeSample:
    u: int
    v: int
    t: int
    y: int


def save_samples_cache(
    path: Path,
    train_samples: Sequence[EdgeSample],
    test_samples: Sequence[EdgeSample],
    meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _pack(samples: Sequence[EdgeSample]) -> np.ndarray:
        if not samples:
            return np.zeros((0, 4), dtype=np.int64)
        arr = np.array([[s.u, s.v, s.t, s.y] for s in samples], dtype=np.int64)
        return arr

    payload = {"train": _pack(train_samples), "test": _pack(test_samples)}
    if meta is not None:
        # Store as UTF-8 bytes for safe round-trip with np.load(..., allow_pickle=False).
        meta_json = json.dumps(meta, sort_keys=True, ensure_ascii=False)
        payload["meta_json"] = np.array([meta_json.encode("utf-8")], dtype="|S")
    np.savez_compressed(path, **payload)


def load_samples_cache(path: Path) -> tuple[list[EdgeSample], list[EdgeSample], dict]:
    obj = np.load(path, allow_pickle=False)

    def _unpack(arr: np.ndarray) -> list[EdgeSample]:
        out: list[EdgeSample] = []
        for row in arr:
            out.append(EdgeSample(u=int(row[0]), v=int(row[1]), t=int(row[2]), y=int(row[3])))
        return out

    meta = {}
    if "meta_json" in obj.files:
        try:
            raw = obj["meta_json"][0]
            if isinstance(raw, np.bytes_):
                raw = raw.tobytes().decode("utf-8", errors="replace")
            elif isinstance(raw, (bytes, bytearray)):
                raw = bytes(raw).decode("utf-8", errors="replace")
            elif isinstance(raw, str):
                pass
            else:
                # Backward-compat for old cache files saved with object dtype.
                raw = str(raw)
            meta = json.loads(raw)
        except Exception:
            meta = {}
    return _unpack(obj["train"]), _unpack(obj["test"]), meta


def load_processed(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req_cols = {"src", "dst", "ts"}
    if not req_cols.issubset(df.columns):
        raise ValueError(f"Missing columns in {path}: expected {req_cols}")
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    df["ts"] = df["ts"].astype(np.int64)
    return df


def build_snapshot_graphs(df: pd.DataFrame) -> tuple[dict[int, dict[int, set[int]]], dict[int, set[tuple[int, int]]], int]:
    snapshots: dict[int, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    edge_sets: dict[int, set[tuple[int, int]]] = defaultdict(set)
    n_nodes = int(pd.concat([df["src"], df["dst"]], ignore_index=True).nunique())

    for row in df.itertuples(index=False):
        u, v, t = int(row.src), int(row.dst), int(row.ts)
        snapshots[t][u].add(v)
        snapshots[t][v].add(u)
        edge_sets[t].add((u, v))
    return snapshots, edge_sets, n_nodes


def split_by_time(df: pd.DataFrame, train_ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Timestamp-level split (not edge-count split), to preserve chronology in dynamic settings.
    ts_vals = sorted(df["ts"].unique().tolist())
    cut_idx = max(1, int(len(ts_vals) * train_ratio))
    train_ts = set(ts_vals[:cut_idx])
    test_ts = set(ts_vals[cut_idx:])
    if not test_ts:
        # Ensure test is non-empty in tiny datasets.
        train_ts = set(ts_vals[:-1])
        test_ts = {ts_vals[-1]}
    train_df = df[df["ts"].isin(train_ts)].copy().sort_values("ts").reset_index(drop=True)
    test_df = df[df["ts"].isin(test_ts)].copy().sort_values("ts").reset_index(drop=True)
    return train_df, test_df


def _weighted_choice_index(cum_weights: list[float], r: float) -> int:
    return bisect.bisect_left(cum_weights, r)


def _snapshot_pos_edges(graph: dict[int, set[int]]) -> list[tuple[int, int]]:
    edges: list[tuple[int, int]] = []
    for u, nbrs in graph.items():
        for v in nbrs:
            if u < v:
                edges.append((u, v))
    return edges


def context_negative_samples_strgnn(
    pos_df: pd.DataFrame,
    snapshots: dict[int, dict[int, set[int]]],
    edge_sets: dict[int, set[tuple[int, int]]],
    n_nodes: int,
    n_neg: int,
    seed: int,
) -> List[EdgeSample]:
    """
    StrGNN-style context-dependent negative sampling:
    1) sample an observed edge e=(u,v,t) according to observed edge distribution P(E);
    2) replace one endpoint with a node drawn from snapshot-aware node distribution;
    3) accept only if new edge does not exist in that timestamp snapshot.
    """
    rng = random.Random(seed)
    pos_rows = list(pos_df.itertuples(index=False))
    if not pos_rows:
        return []

    # Edge distribution P(E): here approximated by empirical frequency in the observed split.
    edge_counter: dict[tuple[int, int, int], int] = defaultdict(int)
    for r in pos_rows:
        edge_counter[(int(r.src), int(r.dst), int(r.ts))] += 1
    edge_items = list(edge_counter.items())
    edge_weights = [float(cnt) for _, cnt in edge_items]
    edge_cum = np.cumsum(edge_weights).tolist()
    edge_total = edge_cum[-1]

    # Node distribution per timestamp from observed degree profile.
    node_cum_by_ts: dict[int, tuple[list[int], list[float], float]] = {}
    for t, g in snapshots.items():
        if t not in set(pos_df["ts"].unique()):
            continue
        nodes = sorted(g.keys())
        if not nodes:
            continue
        degs = [max(1.0, float(len(g[n]))) for n in nodes]
        cum = np.cumsum(degs).tolist()
        node_cum_by_ts[t] = (nodes, cum, cum[-1])

    negs: List[EdgeSample] = []
    tries = 0
    max_tries = max(10_000, n_neg * 50)

    while len(negs) < n_neg and tries < max_tries:
        tries += 1
        edge_idx = _weighted_choice_index(edge_cum, rng.random() * edge_total)
        (u, v, t), _ = edge_items[edge_idx]

        if t in node_cum_by_ts:
            nodes_t, cum_t, total_t = node_cum_by_ts[t]
            repl = nodes_t[_weighted_choice_index(cum_t, rng.random() * total_t)]
        else:
            repl = rng.randrange(n_nodes)

        if rng.random() < 0.5:
            u2, v2 = repl, v
        else:
            u2, v2 = u, repl
        if u2 == v2:
            continue
        if (u2, v2) in edge_sets.get(t, set()) or (v2, u2) in edge_sets.get(t, set()):
            continue
        negs.append(EdgeSample(u=u2, v=v2, t=t, y=1))
    return negs


def make_samples(
    df: pd.DataFrame,
    snapshots: dict[int, dict[int, set[int]]],
    edge_sets: dict[int, set[tuple[int, int]]],
    n_nodes: int,
    anomaly_ratio: float,
    seed: int,
    window: int,
    train_snapshot_ratio: float = 0.5,
    show_progress: bool = True,
    train_build_cap: int = 0,
    test_build_cap: int = 0,
) -> tuple[List[EdgeSample], List[EdgeSample]]:
    # StrGNN-style warmup: skip earliest snapshots without full history window.
    min_ts = window - 1
    all_ts = sorted([int(t) for t in df["ts"].unique().tolist() if t >= min_ts])
    if not all_ts:
        return [], []

    # StrGNN-style split by snapshot index (chronological), not by edge count.
    cut_idx = int(math.ceil(len(all_ts) * train_snapshot_ratio))
    cut_idx = min(max(1, cut_idx), len(all_ts) - 1) if len(all_ts) > 1 else 1
    train_ts = set(all_ts[:cut_idx])
    test_ts = set(all_ts[cut_idx:])
    if not test_ts:
        test_ts = {all_ts[-1]}
        train_ts = set(all_ts[:-1])

    def _build_for_split(split_name: str, ts_set: set[int], rng_seed: int, split_cap: int = 0) -> list[EdgeSample]:
        pos_by_t: list[tuple[int, list[tuple[int, int]]]] = []
        total_pos = 0
        # Optional construction-time cap. This prevents large datasets such as
        # topology from spending minutes constructing samples that will later be
        # discarded by max_train_samples/max_test_samples. It uses only the
        # current split timestamps and preserves chronological order.
        pos_cap = 0
        if split_cap and split_cap > 0:
            pos_cap = max(1, int(math.ceil(split_cap * (1.0 - anomaly_ratio))))
        ts_iter = sorted(ts_set)
        if show_progress:
            ts_iter = tqdm(ts_iter, desc=f"Collecting {split_name} positives", unit="ts", leave=False)
        for t in ts_iter:
            g_t = snapshots.get(t, {})
            pos_edges = _snapshot_pos_edges(g_t)
            if pos_cap > 0:
                remain = pos_cap - total_pos
                if remain <= 0:
                    break
                if len(pos_edges) > remain:
                    pos_edges = pos_edges[:remain]
            pos_by_t.append((t, pos_edges))
            total_pos += len(pos_edges)
            if pos_cap > 0 and total_pos >= pos_cap:
                break

        pos_samples: list[EdgeSample] = []
        for t, pos_edges in pos_by_t:
            pos_samples.extend([EdgeSample(u, v, t, 0) for u, v in pos_edges])

        n_neg_total = int((anomaly_ratio / max(1e-9, 1.0 - anomaly_ratio)) * total_pos)
        neg_samples: list[EdgeSample] = []
        if total_pos > 0 and n_neg_total > 0:
            pos_df = pd.DataFrame(
                [{"src": s.u, "dst": s.v, "ts": s.t} for s in pos_samples],
                columns=["src", "dst", "ts"],
            )
            if show_progress:
                print(f"Sampling {split_name} anomalies with StrGNN context sampler...")
            neg_samples = context_negative_samples_strgnn(
                pos_df=pos_df,
                snapshots=snapshots,
                edge_sets=edge_sets,
                n_nodes=n_nodes,
                n_neg=n_neg_total,
                seed=rng_seed,
            )
            if len(neg_samples) < n_neg_total:
                print(
                    f"[warn] {split_name}: requested {n_neg_total} anomalies, got {len(neg_samples)} "
                    "from StrGNN context sampler."
                )

        split_samples = pos_samples + neg_samples
        random.Random(rng_seed + 999).shuffle(split_samples)
        if split_cap and split_cap > 0 and len(split_samples) > split_cap:
            split_samples = split_samples[:split_cap]
        return split_samples

    train_samples = _build_for_split("train", train_ts, seed + 11, split_cap=int(train_build_cap or 0))
    test_samples = _build_for_split("test", test_ts, seed + 29, split_cap=int(test_build_cap or 0))
    return train_samples, test_samples


def k_hop_nodes(graph: dict[int, set[int]], u: int, v: int, h: int) -> set[int]:
    if h <= 0:
        return {u, v}
    vis = {u, v}
    dq = deque([(u, 0), (v, 0)])
    while dq:
        x, d = dq.popleft()
        if d == h:
            continue
        for nb in graph.get(x, set()):
            if nb in vis:
                continue
            vis.add(nb)
            dq.append((nb, d + 1))
    return vis


def shortest_dist(graph: dict[int, set[int]], src: int, max_depth: int) -> dict[int, int]:
    dist = {src: 0}
    dq = deque([src])
    while dq:
        x = dq.popleft()
        if dist[x] >= max_depth:
            continue
        for nb in graph.get(x, set()):
            if nb in dist:
                continue
            dist[nb] = dist[x] + 1
            dq.append(nb)
    return dist


def encode_nodes(nodes: List[int], graph: dict[int, set[int]], u: int, v: int, h: int) -> np.ndarray:
    cap = h + 2
    d_u = shortest_dist(graph, u, cap)
    d_v = shortest_dist(graph, v, cap)

    def enc(a: int, b: int) -> int:
        psi = 1 + min(a, b)
        eta = ((a + b) // 2) + abs(a - b)
        return psi * (2 * cap + 2) + eta

    label_ids = []
    for n in nodes:
        a = d_u.get(n, cap)
        b = d_v.get(n, cap)
        label_ids.append(enc(a, b))

    label_dim = (cap + 2) * (2 * cap + 2) + (2 * cap + 1)
    x = np.zeros((len(nodes), label_dim), dtype=np.float32)
    x[np.arange(len(nodes)), np.array(label_ids, dtype=np.int64)] = 1.0
    return x


def build_graph_step(graph: dict[int, set[int]], u: int, v: int, h: int, device: torch.device) -> GraphStep:
    nodes = sorted(k_hop_nodes(graph, u, v, h))
    if u not in nodes:
        nodes.append(u)
    if v not in nodes:
        nodes.append(v)
    nodes = sorted(set(nodes))

    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    adj = np.zeros((n, n), dtype=np.float32)
    for x in nodes:
        i = idx[x]
        for nb in graph.get(x, set()):
            if nb in idx:
                j = idx[nb]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    # StrGNN alignment: remove target link inside enclosing subgraph to avoid trivial leakage.
    if u in idx and v in idx:
        adj[idx[u], idx[v]] = 0.0
        adj[idx[v], idx[u]] = 0.0

    x = encode_nodes(nodes, graph, u, v, h)
    return GraphStep(
        x=torch.tensor(x, dtype=torch.float32, device=device),
        adj=torch.tensor(adj, dtype=torch.float32, device=device),
    )


def sample_to_sequence(
    sample: EdgeSample,
    snapshots: dict[int, dict[int, set[int]]],
    window: int,
    hop: int,
    device: torch.device,
) -> List[GraphStep]:
    seq: List[GraphStep] = []
    for t in range(sample.t - window + 1, sample.t + 1):
        graph = snapshots.get(t, {})
        seq.append(build_graph_step(graph, sample.u, sample.v, hop, device))
    return seq


def batchify(
    samples: Sequence[EdgeSample],
    snapshots: dict[int, dict[int, set[int]]],
    window: int,
    hop: int,
    device: torch.device,
) -> tuple[List[List[GraphStep]], torch.Tensor]:
    batch_x = [sample_to_sequence(s, snapshots, window, hop, device) for s in samples]
    labels = torch.tensor([s.y for s in samples], dtype=torch.float32, device=device)
    return batch_x, labels


def precompute_sample_sequences(
    samples: Sequence[EdgeSample],
    snapshots: dict[int, dict[int, set[int]]],
    window: int,
    hop: int,
    show_progress: bool = True,
    desc: str = "Precomputing subgraphs",
) -> tuple[List[List[GraphStep]], list[int]]:
    cpu = torch.device("cpu")
    seqs: List[List[GraphStep]] = []
    labels: list[int] = []
    it = samples
    if show_progress:
        it = tqdm(samples, desc=desc, unit="sample", leave=False)
    for s in it:
        seqs.append(sample_to_sequence(s, snapshots, window, hop, cpu))
        labels.append(int(s.y))
    return seqs, labels


def batchify_cached(
    seqs: Sequence[List[GraphStep]],
    labels: Sequence[int],
    indices: Sequence[int],
    device: torch.device,
) -> tuple[List[List[GraphStep]], torch.Tensor]:
    batch_x: List[List[GraphStep]] = []
    for i in indices:
        seq = seqs[int(i)]
        seq_dev = [
            GraphStep(
                x=step.x.to(device, non_blocking=True),
                adj=step.adj.to(device, non_blocking=True),
            )
            for step in seq
        ]
        batch_x.append(seq_dev)
    batch_y = torch.tensor([labels[int(i)] for i in indices], dtype=torch.float32, device=device)
    return batch_x, batch_y
