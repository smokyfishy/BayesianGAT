"""Residue signals of the legacy NFATC4 analysis (NFATC4.ipynb cells 32-36).

Faithfulness notes, established by reading the notebook execution order:

* Saliency (cells 32/33): gradient of ``sum(logits)`` w.r.t. the 1281-dim input
  (including the label column), absolute value summed over features, eval mode.
  Cell 33 re-ran ``backward`` on the same leaf tensor without zeroing ``x.grad``;
  if executed right after cell 32 the saved vector equals **2x** the single-pass
  saliency. :func:`legacy_saliency` exposes ``accumulate_passes`` for that check.
* Prediction std (cell 34): 50 MC-dropout passes, ``np.std`` of sigmoid outputs.
  Cell 34 leaves the model in ``train()`` mode.
* Delta hidden (cell 35): ``||block3(block2(block1(x_dmd))) - (same for x_ctrl)||``
  with ``x_dmd[:, -1] = 1``. This **skips** ``input_norm`` and the inter-block GELU /
  dropout used in ``forward``, and because cell 34 left the model in train mode the
  saved vector was computed with **dropout active** (a single stochastic draw).
* Joint score (cell 36): ``saliency * std_pred * delta_hidden``.
"""

from __future__ import annotations

import numpy as np
import torch


def legacy_saliency(model, x: torch.Tensor, edge_index: torch.Tensor, accumulate_passes: int = 1) -> np.ndarray:
    model.eval()
    xg = x.clone().detach().requires_grad_(True)
    for _ in range(accumulate_passes):
        model(xg, edge_index).sum().backward()
    return xg.grad.abs().sum(dim=1).detach().cpu().numpy()


def legacy_delta_hidden(
    model, x: torch.Tensor, edge_index: torch.Tensor, dropout_active: bool = False, seed: int | None = None
) -> np.ndarray:
    if seed is not None:
        torch.manual_seed(seed)
    model.train(dropout_active)
    x_ctrl = x.clone()
    x_dmd = x.clone()
    x_dmd[:, -1] = 1
    with torch.no_grad():
        h_ctrl = model.block3(model.block2(model.block1(x_ctrl, edge_index), edge_index), edge_index)
        h_dmd = model.block3(model.block2(model.block1(x_dmd, edge_index), edge_index), edge_index)
    model.eval()
    return (h_dmd - h_ctrl).norm(dim=1).cpu().numpy()


def joint_score_raw(saliency: np.ndarray, std_pred: np.ndarray, delta_hidden: np.ndarray) -> np.ndarray:
    a, b, c = (np.asarray(v, dtype=float).ravel() for v in (saliency, std_pred, delta_hidden))
    if not (a.shape == b.shape == c.shape):
        raise ValueError(f"signal lengths differ: {a.shape} {b.shape} {c.shape}")
    return a * b * c
