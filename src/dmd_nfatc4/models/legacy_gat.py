"""Original ``EnhancedBayesianGAT`` from ``reference_notebooks/DMD.ipynb`` cell 68.

The class body is kept functionally identical to the notebook so that
``best_enhanced_bayesian_gat.pth`` loads with ``strict=True``. Comments have been
condensed; behaviour has not been changed.

Scientific notes (see docs/LEGACY_VS_CORRECTED.md):

* "Bayesian" here means **MC dropout** at inference time (Gal & Ghahramani style
  approximation). There is no explicit weight posterior.
* ``use_edge_confidence=True`` was set in the notebook, but no ``edge_confidence``
  attribute was ever attached to the graph and the GATv2 layers were built without
  ``edge_dim``, so STRING confidence scores never entered the attention computation.
  The saved checkpoint confirms this: it has no ``lin_edge`` parameters.
* The temperature is a learned scalar trained jointly with the classifier on the
  training loss (not post-hoc calibrated on a held-out set).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch_geometric.nn import GATv2Conv


class EnhancedBayesianGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        heads_1=4,
        heads_2=4,
        dropout=0.5,
        use_edge_confidence=True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_edge_confidence = use_edge_confidence

        # (A) input MLP fusion: [N, in] -> [N, hidden]
        self.input_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(hidden_channels),
        )

        # (B) GATv2 layers with residual (only when dims match) + LayerNorm
        self.gat1 = GATv2Conv(hidden_channels, hidden_channels, heads=heads_1, dropout=dropout, concat=True)
        self.ln1 = LayerNorm(hidden_channels * heads_1)
        self.gat2 = GATv2Conv(hidden_channels * heads_1, hidden_channels, heads=heads_2, dropout=dropout, concat=True)
        self.ln2 = LayerNorm(hidden_channels * heads_2)
        self.gat3 = GATv2Conv(hidden_channels * heads_2, out_channels, heads=1, dropout=dropout, concat=False)
        self.ln3 = LayerNorm(out_channels)

        # (D) learnable temperature, initialised to 1.0
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index, edge_confidence=None):
        h = self.input_mlp(x)

        h_in1 = h
        h = F.dropout(h, p=self.dropout, training=self.training)
        if self.use_edge_confidence and edge_confidence is not None:
            h1 = self.gat1(h, edge_index, edge_attr=edge_confidence)
        else:
            h1 = self.gat1(h, edge_index)
        h1 = F.elu(h1)
        if h1.shape[-1] == h_in1.shape[-1]:
            h1 = h1 + h_in1
        h1 = self.ln1(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        h_in2 = h1
        if self.use_edge_confidence and edge_confidence is not None:
            h2 = self.gat2(h1, edge_index, edge_attr=edge_confidence)
        else:
            h2 = self.gat2(h1, edge_index)
        h2 = F.elu(h2)
        if h2.shape[-1] == h_in2.shape[-1]:
            h2 = h2 + h_in2
        h2 = self.ln2(h2)
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        if self.use_edge_confidence and edge_confidence is not None:
            out = self.gat3(h2, edge_index, edge_attr=edge_confidence)
        else:
            out = self.gat3(h2, edge_index)
        out = self.ln3(out)  # raw logits [N, out_channels]
        return out

    def predict_proba(self, x, edge_index, edge_confidence=None, mc_samples=10):
        """MC dropout + temperature scaling, exactly as in the notebook.

        Note: like the original, this leaves the module in ``train()`` mode.
        """
        self.train()
        all_probs = []
        for _ in range(mc_samples):
            logits = self.forward(x, edge_index, edge_confidence)
            probs = F.softmax(logits / self.temperature, dim=1)
            all_probs.append(probs.unsqueeze(0))
        return torch.cat(all_probs, dim=0).mean(dim=0)


# Hyperparameters of the saved checkpoint (DMD.ipynb cells 70 and 73).
LEGACY_GAT_HPARAMS = dict(
    hidden_channels=32,
    out_channels=2,
    heads_1=2,
    heads_2=2,
    dropout=0.5,
    use_edge_confidence=True,
)
# 6 TPM + 640 ESM + 60 687 GO columns (cell 42 comment; confirmed by checkpoint shape).
LEGACY_GAT_IN_CHANNELS = 61333
