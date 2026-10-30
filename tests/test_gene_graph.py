import numpy as np
import pandas as pd
import pytest

from dmd_nfatc4.data.gene_graph import (
    audit_ppi_edges,
    build_feature_matrix,
    knn_edge_index,
    labels_from_gene_list,
    parse_feature_blocks,
    ppi_edge_index,
)


def test_blocks_found_by_name_and_positional_rule_checked(synthetic_gene_data):
    df, _, _ = synthetic_gene_data
    b = parse_feature_blocks(list(df.columns), expected_expr=6, expected_esm=8)
    assert (len(b.expr_cols), len(b.go_cols), len(b.esm_cols)) == (6, 10, 8)
    assert b.diagnostics["positional_rule_matches_names"]
    # legacy model input order is expr | esm | go (DMD.ipynb cell 60)
    assert b.legacy_order[:6] == b.expr_cols and b.legacy_order[6:14] == b.esm_cols


def test_positional_rule_violation_is_reported(synthetic_gene_data):
    df, _, _ = synthetic_gene_data
    cols = list(df.columns)
    shuffled = cols[6:16] + cols[:6] + cols[16:]  # GO block first
    b = parse_feature_blocks(shuffled, expected_expr=6, expected_esm=8)
    assert not b.diagnostics["positional_rule_matches_names"]


def test_unknown_prefix_rejected():
    with pytest.raises(ValueError):
        parse_feature_blocks(["TPM_a", "mystery"])


def test_feature_matrix_normalisation(synthetic_gene_data):
    df, _, _ = synthetic_gene_data
    b = parse_feature_blocks(list(df.columns), expected_expr=6, expected_esm=8)
    X = build_feature_matrix(df, b)
    assert X.shape == (len(df), b.n_features) and X.dtype == np.float32
    np.testing.assert_allclose(X[:, :6].mean(0), 0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(X[:, 6:14], axis=1), 1, atol=1e-4)
    assert set(np.unique(X[:, 14:])) <= {0.0, 1.0}


def test_scaler_fit_rows_changes_statistics(synthetic_gene_data):
    df, _, _ = synthetic_gene_data
    b = parse_feature_blocks(list(df.columns), expected_expr=6, expected_esm=8)
    fit = np.arange(30)
    X = build_feature_matrix(df, b, fit_rows=fit)
    np.testing.assert_allclose(X[fit, :6].mean(0), 0, atol=1e-4)
    assert not np.allclose(X[:, :6].mean(0), 0, atol=1e-4)


def test_ppi_edges_symmetric_no_self_loops_unknown_dropped():
    edges = pd.DataFrame(
        {"gene1": ["A", "A", "B", "C", "Z"], "gene2": ["B", "A", "C", "A", "A"], "combined_score": [0.5, 0.9, 0.4, 0.7, 0.3]}
    )
    ei, w = ppi_edge_index(edges, ["A", "B", "C"])
    pairs = set(map(tuple, ei.t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    assert all(i != j for i, j in pairs)
    assert len(pairs) == 6 and w.shape[0] == 6
    assert audit_ppi_edges(edges)["n_self_loops"] == 1


def test_knn_edges_symmetric_and_subset_restriction():
    X = np.random.default_rng(0).normal(size=(40, 3))
    pairs = set(map(tuple, knn_edge_index(X, k=3).t().tolist()))
    assert all((j, i) in pairs for i, j in pairs)
    ei_sub = knn_edge_index(X, k=3, rows=np.arange(20))
    assert int(ei_sub.max()) < 20


def test_labels_unlabeled_semantics():
    y, d = labels_from_gene_list(["DMD", "UTRN", "FOO"], ["DMD", "UTRN", "XIST"])
    assert y.tolist() == [1, 1, 0]
    assert d["positives_missing_from_graph"] == ["XIST"] and d["n_unlabeled"] == 1


def test_label_manifest_matches_notebook_list():
    from dmd_nfatc4.data.labels import KNOWN_DMD_GENES, label_manifest

    m = label_manifest()
    assert len(KNOWN_DMD_GENES) == 79 == len(set(KNOWN_DMD_GENES))
    assert m.notebook_group.notna().all() and m.citation.isna().all()


def test_real_ppi_table(assets):
    from dmd_nfatc4.data.gene_graph import load_ppi_gene_edges

    edges = load_ppi_gene_edges(assets / "dmd/ppi_edges_genes.tsv")
    a = audit_ppi_edges(edges)
    # 12 self-loops exist in the supplied file (paralog UniProt IDs collapsed onto one symbol);
    # the original code removed them with remove_self_loops, and so does ppi_edge_index.
    assert a["n_self_loops"] == 12 and a["n_duplicate_undirected"] == 0
    assert a["canonical_orientation_gene1_lt_gene2_excl_self_loops"]
    assert 0.0 <= a["score_min"] and a["score_max"] <= 1.0
    genes = sorted(set(edges.gene1) | set(edges.gene2))
    ei, _ = ppi_edge_index(edges, genes)
    assert bool((ei[0] != ei[1]).all())
