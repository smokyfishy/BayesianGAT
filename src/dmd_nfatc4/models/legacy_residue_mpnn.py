"""Original NFATC4 residue model from ``reference_notebooks/NFATC4.ipynb`` cell 24.

IMPORTANT naming note
---------------------
The original classes are called ``GVP`` / ``MultiHeadGVPConv`` / ``BayesianGVP_GNN``,
but **they are not Geometric Vector Perceptrons** (Jing et al., 2021). ``GVP`` here is
a two-layer scalar MLP with LayerNorm + GELU; ``MultiHeadGVPConv`` averages four such
MLPs over neighbour features and sums them (``aggr='add'``). There are no vector
channels, no use of coordinates or directions, and no rotation equivariance. The
coordinates (``data.pos``) only influence the model through which edges exist
(Ca-Ca distance < 8 A).

The class and attribute names are preserved verbatim so that
``bayesian_gvp_model.pt`` loads with ``strict=True``. New code should refer to this
model by its honest alias, :data:`LegacyScalarMPNN`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class GVP(nn.Module):
    """Scalar MLP block (named 'GVP' in the original code; not a geometric vector perceptron)."""

    def __init__(self, in_dim, out_dim, act=F.gelu):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.act = act

    def forward(self, x):
        return self.act(self.proj(x))


class MultiHeadGVPConv(MessagePassing):
    """Sum-aggregated scalar message passing with head-averaged MLP messages."""

    def __init__(self, in_dim, out_dim, heads=4, dropout=0.25):
        super().__init__(aggr="add")
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.msg_mlps = nn.ModuleList([GVP(in_dim, out_dim) for _ in range(heads)])
        self.agg_norm = nn.LayerNorm(out_dim)
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        messages = [mlp(x_j) for mlp in self.msg_mlps]
        msg_out = torch.stack(messages).mean(dim=0)
        return self.dropout(msg_out)

    def update(self, aggr_out, x):
        return self.agg_norm(aggr_out + self.residual(x))


class BayesianGVP_GNN(nn.Module):  # noqa: N801 - original name preserved for checkpoint compatibility
    def __init__(self, in_dim, hidden_dim=256, dropout=0.3, heads=4):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.block1 = MultiHeadGVPConv(in_dim, hidden_dim, heads, dropout)
        self.block2 = MultiHeadGVPConv(hidden_dim, hidden_dim, heads, dropout)
        self.block3 = MultiHeadGVPConv(hidden_dim, hidden_dim, heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x, edge_index):
        x = self.input_norm(x)
        x = self.block1(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.block2(x, edge_index)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.block3(x, edge_index)
        x = F.gelu(self.final_norm(x))
        x = self.dropout(x)
        return self.classifier(x).view(-1)


#: Honest alias used throughout the new code base.
LegacyScalarMPNN = BayesianGVP_GNN

# Hyperparameters of the saved checkpoint (NFATC4.ipynb cells 25 and 27).
LEGACY_RESIDUE_HPARAMS = dict(hidden_dim=256, dropout=0.3, heads=4)
LEGACY_RESIDUE_IN_DIM = 1281  # 1280 ESM-2 650M dims + 1 simulated mutation flag (the label!)
