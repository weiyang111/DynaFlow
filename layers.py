import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

class GFTLayer(nn.Module):
    def __init__(self, in_channels, k_nodes=10, beta=1.0):
        super(GFTLayer, self).__init__()
        self.k = k_nodes
        self.beta = beta
        self.proj = nn.Linear(in_channels, 1, bias=False)

    def forward(self, batched_graph: Batch):
        graphs = batched_graph.to_data_list()
        
        batch_size = len(graphs)
        device = graphs[0].x.device
        in_channels = graphs[0].x.size(-1)
        
        pooled_x_list = []
        pooled_adj_list = []
        
        for data in graphs:
            x = data.x
            edge_index = data.edge_index
            N = x.size(0)
            
            adj = torch.zeros((N, N), device=device)
            if edge_index.numel() > 0:
                adj[edge_index[0], edge_index[1]] = 1.0
                
            deg = adj.sum(dim=1)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            D_inv_sqrt = torch.diag(deg_inv_sqrt)
            
            L_t = torch.eye(N, device=device) - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
            
            eigenvalues, U_t = torch.linalg.eigh(L_t)
            
            filter_diag = torch.diag(torch.exp(-self.beta * eigenvalues))
            U_filtered = torch.mm(U_t, filter_diag)
            x_freq = torch.mm(U_t.T, x)
            x_enhanced = torch.mm(U_filtered, x_freq)
            
            adj_loop = adj + torch.eye(N, device=device)
            score_deg = adj_loop.sum(dim=1).pow(-0.5)
            score_deg.masked_fill_(score_deg == float('inf'), 0)
            D_hat = torch.diag(score_deg)
            
            smooth_feat = torch.mm(torch.mm(D_hat, adj_loop), D_hat)
            smooth_feat = torch.mm(smooth_feat, x_enhanced)
            scores = torch.sigmoid(self.proj(smooth_feat)).squeeze(-1) 
            if N < self.k:
                pad_size = self.k - N
                x_pooled = F.pad(x_enhanced, (0, 0, 0, pad_size))
                adj_pooled = F.pad(adj, (0, pad_size, 0, pad_size))
            else:
                _, indices = torch.topk(scores, self.k)
                x_pooled = x_enhanced[indices]
                # 依据保留的节点裁剪邻接矩阵
                adj_pooled = adj[indices][:, indices]
                
            pooled_x_list.append(x_pooled)
            pooled_adj_list.append(adj_pooled)
            
        return torch.stack(pooled_x_list), torch.stack(pooled_adj_list)

class GNN_GRU_Layer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNN_GRU_Layer, self).__init__()
        self.W_z = nn.Linear(in_channels, hidden_channels)
        self.U_z = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.W_r = nn.Linear(in_channels, hidden_channels)
        self.U_r = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.W_h = nn.Linear(in_channels, hidden_channels)
        self.U_h = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, h_prev, adj_norm):
        Y_t = torch.bmm(adj_norm, h_prev)
        
        z_t = torch.sigmoid(self.W_z(Y_t) + self.U_z(h_prev))
        r_t = torch.sigmoid(self.W_r(Y_t) + self.U_r(h_prev))
        h_tilde = torch.tanh(self.W_h(Y_t) + self.U_h(r_t * h_prev))
        
        h_next = (1 - z_t) * h_prev + z_t * h_tilde
        return h_next

class GARUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GARUCell, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_z = nn.Linear(input_size, hidden_size)
        self.U_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(input_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_h = nn.Linear(input_size, hidden_size)
        self.U_h = nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.W_alpha = nn.Linear(input_size, hidden_size)

    def forward(self, H_t, h_prev):
        r_t = torch.sigmoid(self.W_r(H_t) + self.U_r(h_prev))
        z_t = torch.sigmoid(self.W_z(H_t) + self.U_z(h_prev))
        
        alpha_t = torch.sigmoid(self.W_alpha(H_t))
        
        h_tilde = torch.tanh(self.W_h(H_t) + self.U_h(r_t * h_prev))
        
        h_next = (1 - z_t) * h_prev + z_t * alpha_t * h_tilde
        
        return h_next