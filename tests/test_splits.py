import numpy as np
import torch

from dmd_nfatc4.data.augmentation import graphsmote_legacy, graphsmote_train_only
from dmd_nfatc4.data.splits import assert_disjoint, audit_augmentation_leakage, stratified_split


def _toy(n=200, n_pos=20, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5)).astype(np.float32)
    y = np.zeros(n, dtype=np.int64)
    y[rng.choice(n, n_pos, replace=False)] = 1
    s, d = rng.integers(0, n, 600), rng.integers(0, n, 600)
    ei = torch.tensor(np.stack([np.r_[s, d], np.r_[d, s]]))
    return X, y, ei


def test_split_deterministic_disjoint_complete_stratified():
    _, y, _ = _toy()
    a, b = stratified_split(y, seed=7), stratified_split(y, seed=7)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)
    assert_disjoint(*a)
    assert sum(len(s) for s in a) == len(y)
    assert y[a[0]].sum() == int(0.7 * y.sum())


def test_different_seed_gives_different_split():
    _, y, _ = _toy()
    assert not np.array_equal(stratified_split(y, seed=1)[0], stratified_split(y, seed=2)[0])


def test_legacy_augment_then_split_leaks():
    X, y, ei = _toy()
    aug = graphsmote_legacy(X, y, ei, ratio=0.25, seed=0)
    assert aug.n_synthetic == int(0.25 * 180) - 20
    tr, va, te = stratified_split(aug.y, seed=42)
    audit = audit_augmentation_leakage(aug.parents, aug.n_original, tr, va, te)
    assert audit["test"]["n_synthetic"] > 0
    assert audit["test"]["n_synthetic_with_parent_in_train"] > 0


def test_corrected_augmentation_is_training_only():
    X, y, ei = _toy()
    tr, va, te = stratified_split(y, seed=3)
    aug, new_tr = graphsmote_train_only(X, y, ei, tr, ratio=0.25, seed=0)
    syn = np.arange(aug.n_original, aug.X.shape[0])
    assert aug.n_synthetic > 0
    assert set(syn) <= set(new_tr) and not (set(syn) & (set(va) | set(te)))
    assert set(aug.parents.ravel()) <= set(tr[y[tr] == 1])
    src, dst = aug.edge_index.numpy()
    touching = (src >= aug.n_original) | (dst >= aug.n_original)
    others = np.where(src[touching] >= aug.n_original, dst[touching], src[touching])
    assert set(others.tolist()) <= set(new_tr.tolist())  # synthetic nodes only touch training nodes
    audit = audit_augmentation_leakage(aug.parents, aug.n_original, new_tr, va, te)
    assert audit["test"]["n_synthetic"] == 0 and audit["val"]["n_synthetic"] == 0


def test_synthetic_edges_none_isolates_synthetic_nodes():
    X, y, ei = _toy()
    tr, _, _ = stratified_split(y, seed=3)
    aug, _ = graphsmote_train_only(X, y, ei, tr, synthetic_edges="none", seed=0)
    assert int(aug.edge_index.max()) < aug.n_original
