"""Model factory for the corrected gene-graph pipeline.

The corrected pipeline keeps the original ``EnhancedBayesianGAT`` architecture (the
methodological problems were in the data protocol, not the layer stack). The only
architectural option added is :class:`EdgeAwareBayesianGAT`, which *actually* feeds
STRING ``combined_score`` into GATv2 attention via ``edge_dim=1`` -- something the
original intended (``use_edge_confidence=True``) but never wired up.
"""

from __future__ import annotations

import torch
from torch_geometric.nn import GATv2Conv

from .legacy_gat import EnhancedBayesianGAT


class EdgeAwareBayesianGAT(EnhancedBayesianGAT):
    """``EnhancedBayesianGAT`` whose GATv2 layers accept a scalar edge attribute."""

    def __init__(self, in_channels, hidden_channels, out_channels, heads_1=4, heads_2=4, dropout=0.5):
        super().__init__(in_channels, hidden_channels, out_channels, heads_1, heads_2, dropout, use_edge_confidence=True)
        h = hidden_channels
        self.gat1 = GATv2Conv(h, h, heads=heads_1, dropout=dropout, concat=True, edge_dim=1)
        self.gat2 = GATv2Conv(h * heads_1, h, heads=heads_2, dropout=dropout, concat=True, edge_dim=1)
        self.gat3 = GATv2Conv(h * heads_2, out_channels, heads=1, dropout=dropout, concat=False, edge_dim=1)


class SparseAwareSequential(torch.nn.Sequential):
    """Drop-in replacement for ``input_mlp`` that also accepts a sparse feature matrix.

    The real node table is ~18.5k x 61k with a very sparse binary GO block (a dense
    float32 copy is ~4.5 GB). State-dict keys (``input_mlp.0.weight`` ...) are unchanged,
    so legacy checkpoints still load strictly.
    """

    def forward(self, x):
        if x.is_sparse or x.layout == torch.sparse_csr:
            lin = self[0]
            h = torch.sparse.mm(x, lin.weight.t()) + lin.bias
            for layer in list(self)[1:]:
                h = layer(h)
            return h
        return super().forward(x)


def enable_sparse_input(model: EnhancedBayesianGAT) -> EnhancedBayesianGAT:
    model.input_mlp = SparseAwareSequential(*model.input_mlp)
    return model


def build_gat(in_channels: int, model_cfg: dict) -> EnhancedBayesianGAT:
    kwargs = dict(
        in_channels=in_channels,
        hidden_channels=int(model_cfg.get("hidden_channels", 32)),
        out_channels=int(model_cfg.get("out_channels", 2)),
        heads_1=int(model_cfg.get("heads_1", 2)),
        heads_2=int(model_cfg.get("heads_2", 2)),
        dropout=float(model_cfg.get("dropout", 0.5)),
    )
    if model_cfg.get("edge_confidence_in_attention", False):
        return EdgeAwareBayesianGAT(**kwargs)
    return EnhancedBayesianGAT(**kwargs, use_edge_confidence=bool(model_cfg.get("use_edge_confidence", True)))
