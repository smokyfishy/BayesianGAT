import pytest
import torch

from dmd_nfatc4.models import BayesianGVP_GNN, EdgeAwareBayesianGAT, EnhancedBayesianGAT, ResidueScalarMPNN
from dmd_nfatc4.models.gat import enable_sparse_input


def _ring(n):
    s = torch.arange(n)
    return torch.stack([torch.cat([s, (s + 1) % n]), torch.cat([(s + 1) % n, s])])


def test_legacy_gat_forward_and_predict_proba():
    m = EnhancedBayesianGAT(20, 8, 2, heads_1=2, heads_2=2, dropout=0.5)
    x, ei = torch.randn(10, 20), _ring(10)
    m.eval()
    assert m(x, ei).shape == (10, 2)
    p = m.predict_proba(x, ei, mc_samples=3)
    assert p.shape == (10, 2) and torch.allclose(p.sum(1), torch.ones(10), atol=1e-5)


def test_edge_aware_gat_uses_edge_attr():
    torch.manual_seed(0)
    m = EdgeAwareBayesianGAT(20, 8, 2, heads_1=2, heads_2=2, dropout=0.0).eval()
    x, ei = torch.randn(10, 20), _ring(10)
    a = m(x, ei, torch.rand(ei.shape[1], 1))
    b = m(x, ei, torch.rand(ei.shape[1], 1))
    assert a.shape == (10, 2) and not torch.allclose(a, b)


def test_sparse_input_matches_dense():
    torch.manual_seed(0)
    m = EnhancedBayesianGAT(30, 8, 2, heads_1=2, heads_2=2, dropout=0.5).eval()
    x = (torch.rand(12, 30) > 0.8).float()
    ei = _ring(12)
    dense = m(x, ei)
    keys = set(m.state_dict())
    enable_sparse_input(m)
    assert set(m.state_dict()) == keys  # checkpoint keys preserved
    assert torch.allclose(dense, m(x.to_sparse(), ei), atol=1e-5)


def test_legacy_residue_model_shape():
    m = BayesianGVP_GNN(17, hidden_dim=16, heads=2).eval()
    assert m(torch.randn(9, 17), _ring(9)).shape == (9,)


def test_corrected_residue_model_rejects_wrong_width():
    m = ResidueScalarMPNN(16, hidden_dim=8, heads=2)
    with pytest.raises(ValueError):
        m(torch.randn(5, 17), _ring(5))
