import numpy as np
import pytest
import torch

from dmd_nfatc4.data.residue_graph import LabelLeakageError, assert_no_label_leakage, build_residue_features


def test_legacy_features_leak_the_label_and_guard_catches_it():
    """Regression test: the original x=[ESM|flag], y=x[:,-1] construction must be caught."""
    esm = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    flags = (np.arange(50) % 7 == 0).astype(float)
    x = build_residue_features(esm, flags, mode="legacy")
    with pytest.raises(LabelLeakageError):
        assert_no_label_leakage(x, torch.tensor(flags))


def test_guard_catches_complement_column():
    y = torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32)
    x = torch.cat([torch.randn(5, 3), (1 - y)[:, None]], dim=1)
    with pytest.raises(LabelLeakageError):
        assert_no_label_leakage(x, y)


def test_corrected_features_never_contain_label():
    esm = np.random.default_rng(1).normal(size=(50, 8)).astype(np.float32)
    flags = (np.arange(50) < 10).astype(float)
    x = build_residue_features(esm, flags, mode="corrected")
    assert x.shape == (50, 8)
    assert_no_label_leakage(x, torch.tensor(flags))


def test_corrected_training_reports_not_evaluated_without_labels(assets, tmp_path):
    from dmd_nfatc4.training.train_residue import train_residue_corrected

    rep = train_residue_corrected({"data_dir": str(assets), "output_dir": str(tmp_path), "labels": {}})
    assert rep["status"] == "NOT EVALUATED"


def test_external_labels_require_provenance(tmp_path):
    from dmd_nfatc4.training.train_residue import load_external_residue_labels

    p = tmp_path / "labels.tsv"
    p.write_text("uniprot_pos\tlabel\tsource\n5\t1\tsome-db:0001\n9\t0\t\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_external_residue_labels(p, 20)
    p.write_text("uniprot_pos\tlabel\tsource\n5\t1\tsome-db:0001\n9\t0\tsome-db:0002\n", encoding="utf-8")
    y, _ = load_external_residue_labels(p, 20)
    assert y[4] == 1 and y[8] == 0 and (y == -1).sum() == 18  # unlisted residues stay unlabeled
