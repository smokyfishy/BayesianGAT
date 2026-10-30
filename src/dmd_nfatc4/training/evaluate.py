"""Evaluation metrics with guards against invalid scores.

* AUC-type metrics are reported as ``None`` with a reason when only one class is
  present (the original per-domain AUC code silently produced NaN or crashed).
* For the gene task, labels are curated positives vs **unlabeled** genes, so the
  metrics measure "positive vs unlabeled" separability; ranking metrics (recall@k,
  rank of held-out positives) are reported alongside and are easier to interpret.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_metrics(y_true, prob_pos, threshold: float = 0.5, preds=None) -> dict:
    y_true = np.asarray(y_true).astype(int).ravel()
    prob_pos = np.asarray(prob_pos, dtype=float).ravel()
    preds = (prob_pos >= threshold).astype(int) if preds is None else np.asarray(preds).astype(int)
    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    out: dict = {
        "n": int(y_true.size),
        "n_positive": n_pos,
        "n_negative_or_unlabeled": n_neg,
        "prevalence": n_pos / max(y_true.size, 1),
        "confusion_matrix[[tn,fp],[fn,tp]]": confusion_matrix(y_true, preds, labels=[0, 1]).tolist(),
        "accuracy": float((preds == y_true).mean()) if y_true.size else None,
    }
    p, r, f, s = precision_recall_fscore_support(y_true, preds, labels=[0, 1], zero_division=0)
    out["precision_pos"], out["recall_pos_sensitivity"], out["f1_pos"] = float(p[1]), float(r[1]), float(f[1])
    out["specificity"] = float(r[0])
    out["macro_f1"] = float(f1_score(y_true, preds, average="macro", labels=[0, 1], zero_division=0))
    if n_pos == 0 or n_neg == 0:
        reason = "single class present; AUC undefined"
        out.update({"roc_auc": None, "pr_auc": None, "auc_note": reason})
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
        out["pr_auc"] = float(average_precision_score(y_true, prob_pos))
        out["pr_auc_random_baseline"] = out["prevalence"]
    return out


def ranking_metrics(y_true, score, ks=(10, 25, 50, 100)) -> dict:
    """Where do held-out positives land when all evaluated nodes are ranked by score?"""
    y_true = np.asarray(y_true).astype(int).ravel()
    order = np.argsort(-np.asarray(score, dtype=float))
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_ranks = ranks[y_true == 1]
    out = {"n_ranked": int(y_true.size), "n_positive": int(y_true.sum())}
    if pos_ranks.size == 0:
        out["note"] = "no positives in evaluated set"
        return out
    out["positive_ranks"] = sorted(int(r) for r in pos_ranks)
    out["median_positive_rank"] = float(np.median(pos_ranks))
    out["median_positive_percentile"] = float(100 * (1 - (np.median(pos_ranks) - 1) / len(order)))
    for k in ks:
        if k <= len(order):
            out[f"recall_at_{k}"] = float((pos_ranks <= k).sum() / pos_ranks.size)
    return out


def aggregate_seed_metrics(per_seed: list[dict], keys=("roc_auc", "pr_auc", "macro_f1", "recall_pos_sensitivity")):
    agg = {}
    for k in keys:
        vals = [m[k] for m in per_seed if m.get(k) is not None]
        if vals:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}
    return agg
