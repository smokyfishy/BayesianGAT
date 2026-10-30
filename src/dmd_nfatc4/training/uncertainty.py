"""Monte Carlo dropout utilities.

MC dropout (keeping dropout active at inference and averaging T stochastic passes)
is a cheap *approximation* to epistemic uncertainty. It is not a sampled posterior
over all weights, and its standard deviations are not calibrated by construction.
Check calibration on held-out labelled data before interpreting them as reliability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def mc_dropout_gat(model, x, edge_index, edge_attr=None, mc_samples: int = 50, use_temperature: bool = True):
    """Positive-class probability mean/std over MC passes (DMD.ipynb cell 73)."""
    was_training = model.training
    model.train()
    samples = []
    for _ in range(mc_samples):
        logits = model(x, edge_index, edge_attr)
        if use_temperature:
            logits = logits / model.temperature
        samples.append(F.softmax(logits, dim=1)[:, 1].cpu())
    model.train(was_training)
    s = torch.stack(samples)
    return s.mean(0).numpy(), s.std(0).numpy()  # torch.std is unbiased, as in the notebook


@torch.no_grad()
def mc_dropout_binary(model, x, edge_index, mc_samples: int = 50):
    """Sigmoid mean/std over MC passes (NFATC4.ipynb cell 34; population std via numpy)."""
    was_training = model.training
    model.train()
    preds = np.stack([torch.sigmoid(model(x, edge_index)).cpu().numpy() for _ in range(mc_samples)])
    model.train(was_training)
    return preds.mean(axis=0), preds.std(axis=0)


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float | None:
    y_true, prob = np.asarray(y_true).ravel(), np.asarray(prob).ravel()
    if y_true.size == 0:
        return None
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y_true[m].mean())
    return float(ece)
