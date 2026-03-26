import torch
import torch.nn as nn
from layers import GFTLayer, GNN_GRU_Layer, GARUCell

class DynaFlow(nn.Module):
    def __init__(self, in_channels, gru_hidden=32, garu_hidden=256, k_nodes=10, num_layers=4, dropout=0.3):
        super(DynaFlow, self).__init__()
        
        self.gft = GFTLayer(in_channels=in_channels, k_nodes=k_nodes, beta=1.0)
        
        self.feature_init = nn.Linear(in_channels, gru_hidden)
        
        self.spatial_layers = nn.ModuleList([
            GNN_GRU_Layer(gru_hidden, gru_hidden) for _ in range(num_layers)
        ])
        
        self.garu_input_proj = nn.Linear(k_nodes * gru_hidden, garu_hidden)
        self.garu_cell = GARUCell(garu_hidden, garu_hidden)
        
        self.classifier = nn.Sequential(
            nn.Linear(garu_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, batched_window):
        window_size = len(batched_window)
        batch_size = batched_window[0].num_graphs
        device = batched_window[0].x.device
        
        h_garu = torch.zeros(batch_size, 256, device=device)
        
        for t in range(window_size):
            x_k, adj_k = self.gft(batched_window[t])
            
            h_0 = self.feature_init(x_k)
            h_spatial = h_0
            
            k = x_k.size(1)
            adj_loop = adj_k + torch.eye(k, device=device).unsqueeze(0)
            deg = adj_loop.sum(dim=-1, keepdim=True)
            adj_norm = adj_loop / (deg + 1e-6)
            
            for layer in self.spatial_layers:
                h_spatial = layer(h_spatial, adj_norm)
                h_spatial = h_spatial + h_0
            
            combined_spatial = h_spatial.view(batch_size, -1)
            garu_input = self.garu_input_proj(combined_spatial)
            
            h_garu = self.garu_cell(garu_input, h_garu)
            
        out = self.classifier(h_garu)
        return out