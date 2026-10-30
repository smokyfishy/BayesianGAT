import numpy as np
import pytest
import torch

from dmd_nfatc4.models import (
    LEGACY_GAT_HPARAMS,
    LEGACY_GAT_IN_CHANNELS,
    LEGACY_RESIDUE_HPARAMS,
    LEGACY_RESIDUE_IN_DIM,
    BayesianGVP_GNN,
    EnhancedBayesianGAT,
)
from dmd_nfatc4.utils import load_state_dict


def test_original_gat_checkpoint_strict(assets):
    sd = load_state_dict(assets / "dmd/best_enhanced_bayesian_gat.pth")
    assert not any("lin_edge" in k for k in sd)  # edge confidence never entered attention
    m = EnhancedBayesianGAT(LEGACY_GAT_IN_CHANNELS, **LEGACY_GAT_HPARAMS)
    m.load_state_dict(sd, strict=True)


def test_original_nfatc4_checkpoint_strict_and_saliency_factor(assets):
    from dmd_nfatc4.analysis.residue_saliency import legacy_saliency
    from dmd_nfatc4.data.residue_graph import build_residue_graph

    A = assets / "nfatc4"
    m = BayesianGVP_GNN(LEGACY_RESIDUE_IN_DIM, **LEGACY_RESIDUE_HPARAMS)
    m.load_state_dict(load_state_dict(A / "bayesian_gvp_model.pt"), strict=True)
    g = build_residue_graph(
        A / "AF-Q14934-F1-model_v4.pdb", A / "Q14934_esm.npy", A / "simulated_dmd_mutation_vector.npy", mode="legacy"
    )
    saved = np.load(A / "saliency_vector.npy")
    # saved saliency = 2 x single-pass saliency (x.grad accumulated over cells 32 and 33)
    np.testing.assert_allclose(legacy_saliency(m, g.x, g.edge_index, accumulate_passes=2), saved, rtol=1e-4)


def test_wrong_architecture_is_not_silently_loaded(assets):
    m = EnhancedBayesianGAT(LEGACY_GAT_IN_CHANNELS, 32, 2, heads_1=4, heads_2=4)
    with pytest.raises(RuntimeError):
        m.load_state_dict(load_state_dict(assets / "dmd/best_enhanced_bayesian_gat.pth"), strict=True)


def test_non_state_dict_rejected(tmp_path):
    p = tmp_path / "x.pt"
    torch.save({"a": 1}, p)
    with pytest.raises(TypeError):
        load_state_dict(p)
