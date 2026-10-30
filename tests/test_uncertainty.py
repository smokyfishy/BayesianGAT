import numpy as np
import torch

from dmd_nfatc4.models import BayesianGVP_GNN, EnhancedBayesianGAT
from dmd_nfatc4.training.evaluate import binary_metrics, ranking_metrics
from dmd_nfatc4.training.uncertainty import expected_calibration_error, mc_dropout_binary, mc_dropout_gat


def _ring(n):
    s = torch.arange(n)
    return torch.stack([torch.cat([s, (s + 1) % n]), torch.cat([(s + 1) % n, s])])


def test_mc_dropout_gat_has_spread_and_restores_mode():
    torch.manual_seed(0)
    m = EnhancedBayesianGAT(10, 8, 2, heads_1=2, heads_2=2, dropout=0.5).eval()
    mean, std = mc_dropout_gat(m, torch.randn(15, 10), _ring(15), mc_samples=20)
    assert mean.shape == (15,) and (std > 0).any() and not m.training
    assert ((mean >= 0) & (mean <= 1)).all()


def test_mc_dropout_binary_reproducible_with_seed():
    m = BayesianGVP_GNN(6, hidden_dim=8, heads=2)
    x, ei = torch.randn(10, 6), _ring(10)
    torch.manual_seed(5)
    a = mc_dropout_binary(m, x, ei, 10)
    torch.manual_seed(5)
    b = mc_dropout_binary(m, x, ei, 10)
    np.testing.assert_allclose(a[1], b[1])


def test_single_class_auc_not_reported():
    m = binary_metrics([0, 0, 0], [0.1, 0.6, 0.2])
    assert m["roc_auc"] is None and "single class" in m["auc_note"]


def test_binary_metrics_values():
    m = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert m["roc_auc"] == 0.75
    assert m["confusion_matrix[[tn,fp],[fn,tp]]"] == [[2, 0], [1, 1]]


def test_ranking_metrics():
    r = ranking_metrics([0, 1, 0, 1], [0.1, 0.9, 0.5, 0.2], ks=(1, 3))
    assert r["positive_ranks"] == [1, 3]
    assert r["recall_at_1"] == 0.5 and r["recall_at_3"] == 1.0


def test_ece_perfect_calibration_zero():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0.0
