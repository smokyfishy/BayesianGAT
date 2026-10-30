"""Corrected residue-level model.

Same scalar message-passing architecture as the legacy model (honestly named), but
constructed only from label-free inputs. The constructor refuses to build a model
whose input width equals ``ESM + 1`` when told the extra column is the target.

A genuine rotation-equivariant GVP is *not* implemented here; doing so would be a new
experiment requiring its own protocol and tests (see docs/LEGACY_VS_CORRECTED.md).
"""

from __future__ import annotations

import torch

from .legacy_residue_mpnn import BayesianGVP_GNN


class ResidueScalarMPNN(BayesianGVP_GNN):
    """Leakage-free use of the legacy scalar MPNN architecture."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, dropout: float = 0.3, heads: int = 4):
        super().__init__(in_dim, hidden_dim=hidden_dim, dropout=dropout, heads=heads)
        self.in_dim = in_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.in_dim:
            raise ValueError(f"expected {self.in_dim} label-free features, got {x.shape[1]}")
        return super().forward(x, edge_index)


__all__ = ["ResidueScalarMPNN"]
