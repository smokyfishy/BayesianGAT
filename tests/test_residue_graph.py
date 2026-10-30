import numpy as np
import torch

from dmd_nfatc4.data.residue_graph import build_residue_graph, compare_graphs, contact_edge_index


def test_contact_graph_strict_cutoff_symmetric_no_self():
    coords = np.array([[0, 0, 0], [7.99, 0, 0], [8.0, 0, 0], [30, 0, 0]], dtype=np.float32)
    pairs = set(map(tuple, contact_edge_index(coords).t().tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 2) not in pairs  # d == 8.0 excluded (strict <)
    assert all(i != j for i, j in pairs)
    assert not any(3 in p for p in pairs)


def test_rebuilt_graph_matches_saved(assets):
    from dmd_nfatc4.utils import load_pyg_data

    A = assets / "nfatc4"
    g = build_residue_graph(
        A / "AF-Q14934-F1-model_v4.pdb", A / "Q14934_esm.npy", A / "simulated_dmd_mutation_vector.npy", mode="legacy"
    )
    assert g.num_nodes == 902 and tuple(g.x.shape) == (902, 1281)
    saved = load_pyg_data(A / "nfatc4_graph.pt")  # safe weights_only path, no --trust-pickle needed
    cmp = compare_graphs(g, saved)
    assert cmp["edge_sets_identical"] and cmp["x_max_abs_diff"] == 0.0 and cmp["pos_max_abs_diff"] == 0.0


def test_corrected_graph_has_no_flag_column(assets):
    A = assets / "nfatc4"
    g = build_residue_graph(
        A / "AF-Q14934-F1-model_v4.pdb", A / "Q14934_esm.npy", A / "simulated_dmd_mutation_vector.npy", mode="corrected"
    )
    assert g.x.shape[1] == 1280
    assert torch.equal(g.x, torch.tensor(np.load(A / "Q14934_esm.npy")))
