"""NFATC4 residue model training.

``legacy``     -- NFATC4.ipynb cell 25: x = [ESM | simulated flag], y = x[:, -1],
                  random 80/20 node split, Adam(lr=4e-3, wd=1e-5), 100 epochs.
                  The model can read the answer from its input; resulting accuracy
                  says nothing about predictive ability.
``corrected``  -- inputs are ESM only (guarded by ``assert_no_label_leakage``) and
                  evaluation uses contiguous sequence blocks held out *inductively*
                  (edges touching the held-out block are removed during training).
                  Targets must be external, provenance-tracked residue labels. If none
                  are supplied, the run reports ``NOT EVALUATED`` unless the user opts
                  into a clearly labelled *simulated-target recovery check*.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..config import resolve_path
from ..data.residue_graph import assert_no_label_leakage, build_residue_graph
from ..models.legacy_residue_mpnn import BayesianGVP_GNN
from ..models.residue_model import ResidueScalarMPNN
from ..utils import set_seed, warn_legacy, write_json
from .evaluate import binary_metrics
from .uncertainty import mc_dropout_binary


def _paths(cfg):
    base = cfg["data_dir"]
    d = cfg.get("data", {})
    return (
        resolve_path(d.get("pdb", "nfatc4/AF-Q14934-F1-model_v4.pdb"), base),
        resolve_path(d.get("esm", "nfatc4/Q14934_esm.npy"), base),
        resolve_path(d.get("mutation_vector", "nfatc4/simulated_dmd_mutation_vector.npy"), base),
    )


def train_residue_legacy(cfg: dict) -> dict:
    warn_legacy("NFATC4 legacy training: the target (mutation flag) is an input feature (label leakage).")
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    tcfg, mcfg = cfg.get("training", {}), cfg.get("model", {})
    pdb, esm, mut = _paths(cfg)
    data = build_residue_graph(pdb, esm, mut, mode="legacy")
    x, y, ei = data.x, data.x[:, -1], data.edge_index
    perm = torch.randperm(data.num_nodes)
    n_tr = int(0.8 * data.num_nodes)
    tr, va = perm[:n_tr], perm[n_tr:]
    model = BayesianGVP_GNN(x.shape[1], **{k: mcfg[k] for k in ("hidden_dim", "dropout", "heads") if k in mcfg})
    opt = torch.optim.Adam(
        model.parameters(), lr=float(tcfg.get("lr", 0.004)), weight_decay=float(tcfg.get("weight_decay", 1e-5))
    )
    log_rows = []
    for epoch in range(1, int(tcfg.get("epochs", 100)) + 1):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, ei)[tr], y[tr])
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(x, ei))
        acc = float(((p[va] > 0.5).float() == y[va]).float().mean())
        if epoch % 10 == 0:
            log_rows.append({"epoch": epoch, "loss": float(loss), "val_acc": acc})
    out_dir = Path(cfg["output_dir"]) / "nfatc4_legacy_retrain"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "bayesian_gvp_model_retrained.pt")
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, ei)).numpy()
    rep = {
        "mode": "legacy",
        "warning": "LABEL LEAKAGE: y = x[:, -1]. Metrics are methodologically invalid as predictive evidence.",
        "seed": seed,
        "log": log_rows,
        "val_metrics_leaky": binary_metrics(y[va].numpy(), p[va.numpy()]),
        "output_dir": str(out_dir),
    }
    write_json(rep, out_dir / "report.json")
    return rep


def load_external_residue_labels(path: Path, n_residues: int) -> tuple[np.ndarray, pd.DataFrame]:
    """TSV with columns ``uniprot_pos`` (1-based), ``label`` (0/1) and ``source`` (provenance).

    Residues not listed are *unlabelled* (mask=False), never assumed benign.
    """
    df = pd.read_csv(path, sep="\t")
    missing = {"uniprot_pos", "label", "source"} - set(df.columns)
    if missing:
        raise ValueError(f"external label file lacks columns {sorted(missing)}")
    if df["source"].isna().any() or (df["source"].astype(str).str.strip() == "").any():
        raise ValueError("every external label needs a non-empty provenance 'source'")
    if df["uniprot_pos"].duplicated().any():
        raise ValueError("duplicate residue positions in external labels")
    if not df["uniprot_pos"].between(1, n_residues).all():
        raise ValueError("uniprot_pos out of range")
    y = np.full(n_residues, -1, dtype=int)
    y[df["uniprot_pos"].to_numpy() - 1] = df["label"].astype(int).to_numpy()
    return y, df


def block_folds(n: int, n_blocks: int) -> list[np.ndarray]:
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:])]


def train_residue_corrected(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    tcfg, mcfg, lcfg = cfg.get("training", {}), cfg.get("model", {}), cfg.get("labels", {})
    pdb, esm, mut = _paths(cfg)
    data = build_residue_graph(pdb, esm, mut, mode="corrected")
    n = data.num_nodes
    out_dir = Path(cfg["output_dir"]) / "nfatc4_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)

    if lcfg.get("path"):
        y, label_df = load_external_residue_labels(resolve_path(lcfg["path"], cfg["data_dir"]), n)
        target_kind = "external_observed"
        provenance = sorted(label_df["source"].astype(str).unique().tolist())
    elif lcfg.get("allow_simulated_target", False):
        y = data.y_simulated.numpy().astype(int)
        target_kind = "SIMULATED_TARGET (not biological labels)"
        provenance = ["NFATC4.ipynb cell 4 simulation (fold-change gate + hand-picked ranges)"]
    else:
        rep = {
            "mode": "corrected",
            "status": "NOT EVALUATED",
            "reason": "No externally sourced, provenance-tracked residue-level labels were supplied. "
            "The only available target is the simulated vector, which is not observed biology. "
            "Set labels.path to a TSV (uniprot_pos, label, source) or labels.allow_simulated_target=true "
            "for an explicitly labelled simulation sanity check.",
        }
        write_json(rep, out_dir / "report.json")
        return rep

    labelled = np.where(y >= 0)[0]
    x = data.x
    assert_no_label_leakage(x[labelled], torch.tensor(y[labelled]))
    oof = np.full(n, np.nan)
    oof_std = np.full(n, np.nan)
    folds = block_folds(n, int(cfg.get("n_blocks", 6)))
    fold_rows = []
    for k, test_block in enumerate(folds):
        set_seed(seed + k)
        test_mask = np.zeros(n, bool)
        test_mask[test_block] = True
        train_nodes = np.array([i for i in labelled if not test_mask[i]])
        keep = ~(torch.as_tensor(test_mask)[data.edge_index[0]] | torch.as_tensor(test_mask)[data.edge_index[1]])
        train_ei = data.edge_index[:, keep]  # inductive: held-out block invisible in training
        model = ResidueScalarMPNN(x.shape[1], **{kk: mcfg[kk] for kk in ("hidden_dim", "dropout", "heads") if kk in mcfg})
        opt = torch.optim.Adam(
            model.parameters(), lr=float(tcfg.get("lr", 0.001)), weight_decay=float(tcfg.get("weight_decay", 1e-5))
        )
        yt = torch.tensor(y, dtype=torch.float32)
        tr_t = torch.as_tensor(train_nodes)
        n_pos = float((y[train_nodes] == 1).sum())
        pos_weight = torch.tensor((len(train_nodes) - n_pos) / max(n_pos, 1.0))
        for _ in range(int(tcfg.get("epochs", 60))):
            model.train()
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x, train_ei)[tr_t], yt[tr_t], pos_weight=pos_weight)
            loss.backward()
            opt.step()
        mean, std = mc_dropout_binary(model, x, data.edge_index, mc_samples=int(cfg.get("mc_samples", 30)))
        eval_idx = np.array([i for i in test_block if y[i] >= 0])
        oof[eval_idx], oof_std[eval_idx] = mean[eval_idx], std[eval_idx]
        fold_rows.append(
            {
                "fold": k,
                "block_uniprot": [int(test_block[0]) + 1, int(test_block[-1]) + 1],
                "n_eval": int(len(eval_idx)),
                "n_pos_eval": int((y[eval_idx] == 1).sum()),
                "n_pos_train": int(n_pos),
            }
        )
    ev = ~np.isnan(oof)
    rep = {
        "mode": "corrected",
        "status": "EVALUATED",
        "target_kind": target_kind,
        "label_provenance": provenance,
        "protocol": f"leave-one-contiguous-block-out ({len(folds)} blocks), inductive edges, ESM-only inputs",
        "folds": fold_rows,
        "pooled_out_of_fold": binary_metrics(y[ev], oof[ev]),
        "caveat": "Simulated targets measure only whether hand-picked ranges are recoverable from ESM "
        "features in unseen sequence blocks; they are not evidence about DMD biology."
        if target_kind.startswith("SIMULATED")
        else "External labels: check their provenance and class balance before interpretation.",
    }
    pd.DataFrame(
        {"index": np.arange(n), "uniprot_pos": np.arange(1, n + 1), "label": y, "oof_mean": oof, "oof_std": oof_std}
    ).to_csv(out_dir / "oof_predictions.csv", index=False)
    write_json(rep, out_dir / "report.json")
    rep["output_dir"] = str(out_dir)
    return rep
