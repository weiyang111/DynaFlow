from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass
class GraphStep:
    x: torch.Tensor  # [N, F]
    adj: torch.Tensor  # [N, N]


class GatedGraphLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_z = nn.Linear(hidden_dim, hidden_dim)
        self.u_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_r = nn.Linear(hidden_dim, hidden_dim)
        self.u_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_h = nn.Linear(hidden_dim, hidden_dim)
        self.u_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, y: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z = torch.sigmoid(self.w_z(y) + self.u_z(h_prev))
        r = torch.sigmoid(self.w_r(y) + self.u_r(h_prev))
        h_tilde = torch.tanh(self.w_h(y) + self.u_h(r * h_prev))
        return (1.0 - z) * h_prev + z * h_tilde


class DynaFlow(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        gnn_layers: int = 4,
        temporal_hidden: int = 256,
        top_k: int = 20,
        beta: float = 0.7,
        use_spectral: bool = True,
        use_low_pass: bool = True,
        temporal_cell: str = "garu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_hidden = temporal_hidden
        self.top_k = top_k
        self.beta = beta
        self.use_spectral = use_spectral
        self.use_low_pass = use_low_pass
        self.temporal_cell = temporal_cell.lower()
        if self.temporal_cell not in {"garu", "gru", "none"}:
            raise ValueError(f"Unsupported temporal_cell={temporal_cell!r}; expected garu, gru, or none.")

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.sort_score = nn.Linear(hidden_dim, 1)
        self.layers = nn.ModuleList([GatedGraphLayer(hidden_dim) for _ in range(gnn_layers)])

        # GARU module
        self.w_r = nn.Linear(hidden_dim, temporal_hidden)
        self.u_r = nn.Linear(temporal_hidden, temporal_hidden, bias=False)
        self.w_z = nn.Linear(hidden_dim, temporal_hidden)
        self.u_z = nn.Linear(temporal_hidden, temporal_hidden, bias=False)
        self.w_a = nn.Linear(hidden_dim, temporal_hidden)
        self.w_h = nn.Linear(hidden_dim, temporal_hidden)
        self.u_h = nn.Linear(temporal_hidden, temporal_hidden, bias=False)

        self.gru_cell = nn.GRUCell(hidden_dim, temporal_hidden)
        self.temporal_proj = nn.Linear(hidden_dim, temporal_hidden)

        self.classifier = nn.Sequential(
            nn.Linear(temporal_hidden, temporal_hidden // 2),
            nn.ReLU(),
            nn.Linear(temporal_hidden // 2, 1),
        )

    def _spectral_enhance(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if not self.use_spectral:
            return x
        n = x.shape[0]
        if n == 0:
            return x
        # Enforce symmetric, finite adjacency for stable eigendecomposition.
        adj = 0.5 * (adj + adj.transpose(0, 1))
        adj = torch.nan_to_num(adj, nan=0.0, posinf=0.0, neginf=0.0)
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        lap = eye - d_inv_sqrt @ adj @ d_inv_sqrt
        lap = 0.5 * (lap + lap.transpose(0, 1))
        lap = torch.nan_to_num(lap, nan=0.0, posinf=0.0, neginf=0.0)

        eigvals = eigvecs = None
        for eps in (0.0, 1e-7, 1e-6, 1e-5):
            try:
                lap_try = lap if eps == 0.0 else (lap + eps * eye)
                eigvals, eigvecs = torch.linalg.eigh(lap_try)
                break
            except RuntimeError:
                continue
        if eigvals is None or eigvecs is None:
            # Last-resort fallback: skip spectral transform for this subgraph.
            return x
        x_hat = eigvecs.transpose(0, 1) @ x
        if self.use_low_pass:
            filt = torch.exp(-self.beta * eigvals).unsqueeze(1)
        else:
            filt = torch.ones_like(eigvals).unsqueeze(1)
        x_enh = eigvecs @ (filt * x_hat)
        return x_enh

    def _sort_pool(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.shape[0]
        eye = torch.eye(n, device=x.device, dtype=x.dtype)
        a_tilde = adj + eye
        deg = a_tilde.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        d_inv_sqrt = torch.diag(deg_inv_sqrt)

        score_signal = d_inv_sqrt @ a_tilde @ d_inv_sqrt @ x
        scores = torch.sigmoid(self.sort_score(score_signal)).squeeze(-1)
        keep = min(self.top_k, n)
        idx = torch.topk(scores, k=keep, sorted=False).indices
        x_sel = x[idx]
        adj_sel = adj[idx][:, idx]

        if keep < self.top_k:
            pad_x = torch.zeros(self.top_k - keep, x.shape[1], device=x.device, dtype=x.dtype)
            x_sel = torch.cat([x_sel, pad_x], dim=0)
            pad_adj = torch.zeros(self.top_k, self.top_k, device=x.device, dtype=x.dtype)
            pad_adj[:keep, :keep] = adj_sel
            adj_sel = pad_adj
        return x_sel, adj_sel

    def _step_embed(self, step: GraphStep) -> torch.Tensor:
        x = self.input_proj(step.x)
        x = self._spectral_enhance(x, step.adj)
        x, adj = self._sort_pool(x, step.adj)

        h0 = x
        h = h0
        eye = torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)
        a_hat = adj + eye
        deg = a_hat.sum(dim=1)
        d_inv = torch.diag(torch.pow(deg.clamp(min=1.0), -1.0))
        a_norm = d_inv @ a_hat
        for layer in self.layers:
            y = a_norm @ h
            h = layer(y, h) + h0
        return h.mean(dim=0)

    def _garu(
        self,
        seq: List[torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the temporal GARU block.

        When ``return_attention`` is true, also return the per-time-step
        dynamic gate ``a_t`` with shape ``[window, temporal_hidden]``.  This is
        useful for analysis/plotting and keeps the training forward path
        unchanged.
        """
        h = torch.zeros(self.temporal_hidden, device=seq[0].device, dtype=seq[0].dtype)
        attentions: list[torch.Tensor] = []
        for x in seq:
            r = torch.sigmoid(self.w_r(x) + self.u_r(h))
            z = torch.sigmoid(self.w_z(x) + self.u_z(h))
            a = torch.sigmoid(self.w_a(x))
            h_tilde = torch.tanh(self.w_h(x) + self.u_h(r * h))
            h = (1.0 - z) * h + z * a * h_tilde
            if return_attention:
                attentions.append(a)
        if return_attention:
            return h, torch.stack(attentions, dim=0)
        return h

    def _gru(
        self,
        seq: List[torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(self.temporal_hidden, device=seq[0].device, dtype=seq[0].dtype)
        for x in seq:
            h = self.gru_cell(x.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        if return_attention:
            attention = torch.ones(
                len(seq),
                self.temporal_hidden,
                device=seq[0].device,
                dtype=seq[0].dtype,
            )
            return h, attention
        return h

    def _temporal_mean(
        self,
        seq: List[torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        h = self.temporal_proj(torch.stack(seq, dim=0).mean(dim=0))
        if return_attention:
            attention = torch.ones(
                len(seq),
                self.temporal_hidden,
                device=seq[0].device,
                dtype=seq[0].dtype,
            )
            return h, attention
        return h

    def encode_sequence(
        self,
        seq: List[GraphStep],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode one sample sequence into its final temporal embedding.

        This public helper is intentionally separate from ``forward`` so figure
        scripts can reuse the exact same src/dynaflow model path without copying
        model internals from older scratch scripts.
        """
        embeds = [self._step_embed(step) for step in seq]
        if self.temporal_cell == "garu":
            return self._garu(embeds, return_attention=return_attention)
        if self.temporal_cell == "gru":
            return self._gru(embeds, return_attention=return_attention)
        return self._temporal_mean(embeds, return_attention=return_attention)

    def forward(
        self,
        batch_seq: List[List[GraphStep]],
        return_embeddings: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        states = []
        attention_list: list[torch.Tensor] = []
        for seq in batch_seq:
            if return_attention:
                h_t, alpha_t = self.encode_sequence(seq, return_attention=True)
                attention_list.append(alpha_t)
            else:
                h_t = self.encode_sequence(seq, return_attention=False)
            states.append(h_t)
        h_batch = torch.stack(states, dim=0)
        logits = self.classifier(h_batch).squeeze(-1)
        if return_embeddings and return_attention:
            return logits, h_batch, attention_list
        if return_embeddings:
            return logits, h_batch
        if return_attention:
            return logits, attention_list
        return logits
