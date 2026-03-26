import os
import random
import torch
import pandas as pd
import numpy as np
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path
from tqdm import tqdm
from torch_geometric.data import Data

DATASET_CONFIGS = {
    'bitcoinalpha': {
        'file_name': 'soc-sign-bitcoinalpha.csv',
        'num_snapshots': 21,
        'sep': ',',
        'comment': None,
        'time_col_idx': 3
    },
    'bitcoinotc': {
        'file_name': 'soc-sign-bitcoinotc.csv',
        'num_snapshots': 63,
        'sep': ',',
        'comment': None,
        'time_col_idx': 3       
    },
    'email-dnc': {
        'file_name': 'email-dnc.edges',
        'num_snapshots': 20,
        'sep': ',',
        'comment': '%',
        'time_col_idx': 2
    },
    'topology': {
        'file_name': 'tech-as-topology.edges',
        'num_snapshots': 63,
        'sep': r'\s+',
        'comment': '%',
        'time_col_idx': 2       
    },
    'uci': {
        'file_name': 'out.opsahl-ucsocial',
        'num_snapshots': 190,
        'sep': r'\s+',
        'comment': '%',
        'time_col_idx': 3
    },
    'digg': {
        'file_name': 'out.munmun_digg_reply',
        'num_snapshots': 16,
        'sep': r'\s+',
        'comment': '%',
        'time_col_idx': 3       
    }
}

def load_real_graph_data(config, data_dir="."):
    csv_path = os.path.join(data_dir, config['file_name'])
    print(f"Loading real dataset from {csv_path}...")
    
    if not os.path.exists(csv_path):
         raise FileNotFoundError(f"找不到数据集文件: {csv_path}。请确保文件与脚本在同级目录！")

    df = pd.read_csv(
        csv_path, 
        sep=config['sep'], 
        comment=config['comment'], 
        header=None,
        on_bad_lines='skip'
    )
    
    time_idx = config['time_col_idx']
    if time_idx >= len(df.columns):
        time_idx = len(df.columns) - 1
        
    df = df.iloc[:, [0, 1, time_idx]].copy()
    df.columns = ['u', 'v', 'timestamp']
    df.dropna(inplace=True)
    df['u'] = df['u'].astype(int)
    df['v'] = df['v'].astype(int)
    df = df[df['u'] != df['v']].copy()
    
    unique_nodes = pd.concat([df['u'], df['v']]).unique()
    node_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_nodes)}
    df['u'] = df['u'].map(node_mapping)
    df['v'] = df['v'].map(node_mapping)
    N_total = len(unique_nodes)
    
    print(f"Total unique nodes mapped: {N_total}")
    print(f"Total valid dynamic edges: {len(df)}")
    
    num_snapshots = config['num_snapshots']
    df['snapshot_idx'] = pd.cut(df['timestamp'], bins=num_snapshots, labels=False)
    df = df.sort_values(by='snapshot_idx').reset_index(drop=True)
    
    time_steps = sorted(df['snapshot_idx'].unique())
    A_series = {}
    print(f"Constructing {len(time_steps)} Scipy sparse graph snapshots...")
    
    for t in time_steps:
        edges_t = df[df['snapshot_idx'] == t][['u', 'v']].values
        if len(edges_t) > 0:
            row = edges_t[:, 0]
            col = edges_t[:, 1]
            data = np.ones(len(row))
            A = ssp.csr_matrix((data, (row, col)), shape=(N_total, N_total))
            A = A + A.T
            A[A > 1] = 1
            A_series[t] = A
        else:
            A_series[t] = ssp.csr_matrix((N_total, N_total))
            
    return A_series, time_steps, N_total

def context_dependent_negative_sampling(A, num_samples):
    row, col = A.nonzero()
    edges = [(u, v) for u, v in zip(row, col) if u < v]
    nodes = list(range(A.shape[0]))
    neg_edges = set()
    
    if len(edges) == 0 or len(nodes) < 2:
        return []

    while len(neg_edges) < num_samples:
        u, v = random.choice(edges)
        if random.random() > 0.5:
            new_u = random.choice(nodes)
            new_edge = (new_u, v) if new_u < v else (v, new_u)
        else:
            new_v = random.choice(nodes)
            new_edge = (u, new_v) if u < new_v else (new_v, u)
            
        if A[new_edge[0], new_edge[1]] == 0 and new_edge[0] != new_edge[1]:
            neg_edges.add(new_edge)
            
    return list(neg_edges)

def neighbors(fringe, A):
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        res = res.union(set(nei))
    return res

def compute_dynaflow_node_label(subgraph):
    K = subgraph.shape[0]
    
    if K <= 2:
        return np.array([[1, 1], [1, 1]])
        
    subgraph_wo0 = subgraph[1:, 1:]
    subgraph_wo1 = subgraph[[0]+list(range(2, K)), :][:, [0]+list(range(2, K))]
    
    dist_to_0 = shortest_path(subgraph_wo0, directed=False, unweighted=True)[1:, 0]
    dist_to_1 = shortest_path(subgraph_wo1, directed=False, unweighted=True)[1:, 0]
    
    dist_to_0[np.isinf(dist_to_0)] = 99
    dist_to_1[np.isinf(dist_to_1)] = 99
    
    eta_v = 1 + np.minimum(dist_to_0, dist_to_1)
    eta_ab = np.floor((dist_to_0 + dist_to_1) / 2) + np.abs(dist_to_0 - dist_to_1)
    
    target_eta_v = np.array([1, 1])
    target_eta_ab = np.array([1, 1])
    
    eta_v = np.concatenate((target_eta_v, eta_v)).astype(int)
    eta_ab = np.concatenate((target_eta_ab, eta_ab)).astype(int)
    
    return np.stack([eta_v, eta_ab], axis=1)

def extract_target_subgraphs(A_series, t, target_edge, window_size=5, h_hops=1, max_nodes_per_hop=100, N_total=0):
    u, v = target_edge
    window_subgraphs = []
    
    start_t = max(0, t - window_size + 1)
    
    for i in range(start_t, t + 1):
        A = A_series.get(i, None)
        
        if A is None or A.nnz == 0:
            nodes = [u, v]
            subgraph = ssp.csr_matrix((2, 2))
        else:
            visited = set([u, v])
            fringe = set([u, v])
            nodes = set([u, v])
            
            for dist in range(1, h_hops + 1):
                fringe = neighbors(fringe, A)
                fringe = fringe - visited
                visited = visited.union(fringe)
                
                if max_nodes_per_hop is not None and len(fringe) > max_nodes_per_hop:
                    fringe = set(random.sample(list(fringe), max_nodes_per_hop))
                    
                if len(fringe) == 0:
                    break
                nodes = nodes.union(fringe)
            
            nodes.remove(u)
            nodes.remove(v)
            nodes = [u, v] + list(nodes)
            
            subgraph = A[nodes, :][:, nodes]
        
        labels_array = compute_dynaflow_node_label(subgraph)
        
        subgraph_coo = subgraph.tocoo()
        edge_index = torch.tensor([subgraph_coo.row, subgraph_coo.col], dtype=torch.long)
        node_features = torch.tensor(labels_array, dtype=torch.float)
        
        data = Data(x=node_features, edge_index=edge_index)
        window_subgraphs.append(data)
        
    return window_subgraphs

def main_preprocess(dataset_key, data_dir=".", window_size=5, h_hops=1, anomaly_ratio=0.05, max_nodes=100):
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"不支持的数据集: {dataset_key}. 可选: {list(DATASET_CONFIGS.keys())}")
        
    config = DATASET_CONFIGS[dataset_key]
    save_path = f"processed_{dataset_key}.pt"
    
    print(f"\n========== 🚀 开始处理数据集: {dataset_key.upper()} ==========")
    A_series, time_steps, N_total = load_real_graph_data(config, data_dir=data_dir)
    
    processed_dataset = []
    if len(time_steps) < window_size:
        raise ValueError(f"快照数量({len(time_steps)})少于窗口大小({window_size})，无法提取时间序列 [cite: 164]！")
    
    start_time = time_steps[window_size - 1]
    
    for t in tqdm(time_steps, desc=f"Processing {dataset_key} snapshots"):
        if t < start_time: continue
        
        A_t = A_series.get(t, None)
        if A_t is None or A_t.nnz == 0: continue
        
        row, col = A_t.nonzero()
        edges_t = [(u, v) for u, v in zip(row, col) if u < v]
        
        if len(edges_t) == 0: continue
        
        for u, v in edges_t:
            window_subgraphs = extract_target_subgraphs(A_series, t, (u, v), window_size, h_hops, max_nodes_per_hop=max_nodes, N_total=N_total)
            processed_dataset.append({'target_edge': (u, v), 'timestamp': t, 'window_subgraphs': window_subgraphs, 'label': 0})
        
        num_anomalies = int(len(edges_t) * anomaly_ratio)
        if num_anomalies > 0:
            neg_edges = context_dependent_negative_sampling(A_t, num_anomalies)
            for u, v in neg_edges:
                window_subgraphs = extract_target_subgraphs(A_series, t, (u, v), window_size, h_hops, max_nodes_per_hop=max_nodes, N_total=N_total)
                processed_dataset.append({'target_edge': (u, v), 'timestamp': t, 'window_subgraphs': window_subgraphs, 'label': 1})

    torch.save(processed_dataset, save_path)
    print(f"\n✅ [{dataset_key.upper()}] 处理完毕！共生成 {len(processed_dataset)} 个时间序列样本。")
    print(f"数据已高速缓存至: {save_path}\n")


if __name__ == "__main__":
    DATA_DIR = "data"  
    TARGET_DATASET = 'digg'
    
    main_preprocess(
        dataset_key=TARGET_DATASET, 
        data_dir=DATA_DIR,
        window_size=5,       
        h_hops=1,            
        anomaly_ratio=0.05,  
        max_nodes=150
    )