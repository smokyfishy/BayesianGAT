"""Gene-graph GAT training: legacy reproduction and corrected protocols.

Legacy protocol (DMD.ipynb cells 60-72):
    normalise on all nodes -> PPI + all-node kNN -> GraphSMOTE on full graph ->
    stratified 70/15/15 split of the *augmented* nodes -> train (class weights,
    temperature inside the loss) -> select by val F1 -> report test metrics.
    Synthetic interpolations of training positives land in val/test => leakage.

Corrected protocol:
    split ORIGINAL nodes first -> (optionally) fit scaler on training nodes ->
    graph: transductive (all nodes/edges visible, loss on train only) or inductive
    (training graph induced on train nodes) -> GraphSMOTE on training positives only
    -> select on original val nodes -> report original test nodes only, with
    positive-vs-unlabeled caveat, ranking metrics and repeated seeds.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F

from ..config import resolve_path
from ..data.augmentation import graphsmote_legacy, graphsmote_train_only
from ..data.expression import load_gene_tpm, log1p_zscore
from ..data.gene_graph import (
    audit_ppi_edges,
    build_feature_matrix,
    knn_edge_index,
    labels_from_gene_list,
    load_node_table_sparse,
    load_ppi_gene_edges,
    make_synthetic_gene_table,
    merge_edges,
    parse_feature_blocks,
    ppi_edge_index,
)
from ..data.labels import KNOWN_DMD_GENES
from ..data.splits import assert_disjoint, audit_augmentation_leakage, stratified_split
from ..models.gat import build_gat, enable_sparse_input
from ..utils import log, set_seed, warn_legacy, write_json
from .evaluate import aggregate_seed_metrics, binary_metrics, ranking_metrics
from .uncertainty import expected_calibration_error, mc_dropout_gat


@dataclass
class GeneInputs:
    genes: list[str]
    raw: object  # DataFrame (dense node table) | dict for sparse | ndarray
    edges: pd.DataFrame
    positives: list[str]
    source: str
    feature_names: list[str]


def load_gene_inputs(cfg: dict) -> GeneInputs:
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "synthetic")
    base = cfg["data_dir"]
    positives = data_cfg.get("positives") or KNOWN_DMD_GENES
    if source == "synthetic":
        df, edges, pos = make_synthetic_gene_table(**data_cfg.get("synthetic", {}))
        return GeneInputs(list(df.index), df, edges, pos, "synthetic", list(df.columns))
    edges = load_ppi_gene_edges(resolve_path(data_cfg.get("ppi_edges", "dmd/ppi_edges_genes.tsv"), base))
    if source == "tpm_only":
        tpm = load_gene_tpm(resolve_path(data_cfg.get("expression", "dmd/expression_gene_tpm_protein_coding.parquet"), base))
        tpm.columns = [f"TPM_{c}" for c in tpm.columns]
        return GeneInputs(list(tpm.index), tpm, edges, positives, "tpm_only", list(tpm.columns))
    if source == "node_table":
        path = resolve_path(data_cfg.get("node_features", "dmd/node_features.parquet"), base)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. The original node table (~111 MB) is not in the starter archive; "
                "see SOURCES.md / `python -m dmd_nfatc4.cli data-instructions`."
            )
        return GeneInputs([], {"path": path}, edges, positives, "node_table", [])
    raise ValueError(f"unknown data.source {source!r}")


def _peek_genes(path) -> list[str]:
    """Read only the gene index (plus one column) of a large node table."""
    import pyarrow.parquet as pq

    first = next(n for n in pq.ParquetFile(path).schema_arrow.names if n.startswith("TPM_"))
    return list(pd.read_parquet(path, columns=[first]).index)


def featurize(inputs: GeneInputs, fit_rows: np.ndarray | None):
    """Return (X, feature_names) in legacy order; X may be scipy CSR for the real table."""
    if inputs.source == "node_table":
        genes, X, blocks = load_node_table_sparse(inputs.raw["path"], fit_rows)
        inputs.genes = genes
        return X, blocks.legacy_order, blocks.diagnostics
    if inputs.source == "tpm_only":
        X = log1p_zscore(inputs.raw.to_numpy(float), fit_rows).astype(np.float32)
        return X, list(inputs.raw.columns), {"note": "TPM-only features (no GO / ESM available locally)"}
    blocks = parse_feature_blocks(list(inputs.raw.columns), expected_expr=6, expected_esm=None)
    return build_feature_matrix(inputs.raw, blocks, fit_rows), blocks.legacy_order, blocks.diagnostics


def _to_torch_x(X, device):
    if sp.issparse(X):
        coo = X.tocoo()
        t = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data.astype(np.float32), coo.shape).coalesce()
        return t.to(device)
    return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)


def _vstack(X, extra):
    if sp.issparse(X):
        return sp.vstack([X, sp.csr_matrix(extra)]).tocsr()
    return np.vstack([X, extra])


def _train_loop(model, x, y, train_edges, eval_edges, train_idx, val_idx, tcfg, device, select="pr_auc"):
    opt = torch.optim.Adam(
        model.parameters(), lr=float(tcfg.get("lr", 0.005)), weight_decay=float(tcfg.get("weight_decay", 5e-4))
    )
    y_t = torch.as_tensor(y, dtype=torch.long, device=device)
    tr = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    n_pos = int((y[train_idx] == 1).sum())
    n_neg = int((y[train_idx] == 0).sum())
    class_w = torch.tensor([1.0, n_neg / max(n_pos, 1)], device=device)
    best, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, int(tcfg.get("epochs", 100)) + 1):
        model.train()
        opt.zero_grad()
        logits = model(x, train_edges, None) / model.temperature
        loss = F.cross_entropy(logits[tr], y_t[tr], weight=class_w)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            prob = F.softmax(model(x, eval_edges, None) / model.temperature, dim=1)[:, 1].cpu().numpy()
        m = binary_metrics(y[val_idx], prob[val_idx])
        score = m["f1_pos"] if select == "f1_pos" else (m.get(select) if m.get(select) is not None else m["f1_pos"])
        history.append({"epoch": epoch, "loss": float(loss), "val_" + select: score})
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return {"best_val": best, "best_epoch": best_epoch, "class_weights": class_w.tolist(), "history": history}


def run_gat(cfg: dict, seed: int) -> dict:
    set_seed(seed)
    device = torch.device(cfg.get("device", "cpu"))
    mode = cfg.get("mode", "corrected")
    gcfg, acfg, tcfg, mcfg = (cfg.get(k, {}) for k in ("graph", "augmentation", "training", "model"))
    inputs = load_gene_inputs(cfg)
    t0 = time.time()
    report: dict = {"mode": mode, "seed": seed, "data_source": inputs.source}

    if mode == "legacy":
        warn_legacy("GAT legacy protocol: augmentation before split leaks synthetic copies into val/test.")
        X, feat_names, feat_diag = featurize(inputs, fit_rows=None)
        genes = inputs.genes
        y, label_diag = labels_from_gene_list(genes, inputs.positives)
        n = len(genes)
        ppi, _ = ppi_edge_index(inputs.edges, genes)
        edges = merge_edges(ppi, knn_edge_index(X, int(gcfg.get("knn_k", 10))), num_nodes=n) if gcfg.get("knn", True) else ppi
        Xd = X.toarray() if sp.issparse(X) else X
        aug = graphsmote_legacy(Xd, y, edges, ratio=float(acfg.get("ratio", 0.25)), k=int(acfg.get("k", 5)), seed=seed)
        tr, va, te = stratified_split(aug.y, seed=int(cfg.get("split_seed", 42)))
        report["leakage_audit"] = audit_augmentation_leakage(aug.parents, n, tr, va, te)
        x = _to_torch_x(aug.X, device)
        train_edges = eval_edges = aug.edge_index.to(device)
        y_all, n_orig = aug.y, n
        select = "f1_pos"
    else:
        # Split ORIGINAL entities first (labels only need the gene list).
        genes = inputs.genes or _peek_genes(inputs.raw["path"])
        inputs.genes = genes
        y, label_diag = labels_from_gene_list(genes, inputs.positives)
        n = len(genes)
        tr, va, te = stratified_split(y, seed=seed)
        assert_disjoint(tr, va, te)
        fit_rows = tr if cfg.get("features", {}).get("scaler_fit", "train_nodes") == "train_nodes" else None
        X, feat_names, feat_diag = featurize(inputs, fit_rows=fit_rows)
        ppi, _ = ppi_edge_index(inputs.edges, genes)
        protocol = gcfg.get("protocol", "transductive")
        use_knn = gcfg.get("knn", False)
        k = int(gcfg.get("knn_k", 10))
        if protocol == "transductive":
            eval_edges = merge_edges(ppi, knn_edge_index(X, k), num_nodes=n) if use_knn else ppi
            train_edges = eval_edges
        elif protocol == "inductive":
            eval_edges = merge_edges(ppi, knn_edge_index(X, k), num_nodes=n) if use_knn else ppi
            tr_set = torch.zeros(n, dtype=torch.bool)
            tr_set[torch.as_tensor(tr)] = True
            keep = tr_set[ppi[0]] & tr_set[ppi[1]]
            train_edges = ppi[:, keep]
            if use_knn:
                train_edges = merge_edges(train_edges, knn_edge_index(X, k, rows=tr), num_nodes=n)
        else:
            raise ValueError(f"unknown graph.protocol {protocol!r}")
        report["graph_protocol"] = protocol
        y_all, n_orig = y, n
        if acfg.get("enabled", True):
            Xd = X.toarray() if sp.issparse(X) else X
            aug, tr = graphsmote_train_only(
                Xd,
                y,
                train_edges,
                tr,
                ratio=float(acfg.get("ratio", 0.25)),
                k=int(acfg.get("k", 5)),
                synthetic_edges=acfg.get("synthetic_edges", "train_neighbors"),
                seed=seed,
            )
            syn = np.arange(n, aug.X.shape[0])
            if len(set(syn.tolist()) & (set(va.tolist()) | set(te.tolist()))):
                raise AssertionError("synthetic node leaked into evaluation split")
            if aug.parents.size and not set(aug.parents.ravel().tolist()) <= set(tr.tolist()):
                raise AssertionError("synthetic parent outside training split")
            # Eval graph gains the synthetic nodes' (train-only) edges as well.
            train_edges = aug.edge_index
            eval_edges = merge_edges(eval_edges, aug.edge_index, num_nodes=aug.X.shape[0])
            X, y_all = aug.X, aug.y
            report["leakage_audit"] = audit_augmentation_leakage(aug.parents, n, tr, va, te)
        x = _to_torch_x(X, device)
        train_edges, eval_edges = train_edges.to(device), eval_edges.to(device)
        select = tcfg.get("selection_metric", "pr_auc")

    model = build_gat(x.shape[1], mcfg).to(device)
    if x.is_sparse:
        enable_sparse_input(model)
    train_info = _train_loop(model, x, y_all, train_edges, eval_edges, tr, va, tcfg, device, select=select)

    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(x, eval_edges, None) / model.temperature, dim=1)[:, 1].cpu().numpy()
    mc_mean, mc_std = mc_dropout_gat(model, x, eval_edges, mc_samples=int(cfg.get("mc_samples", 30)))

    te_orig = te[te < n_orig]
    metrics = {"test_original_nodes": binary_metrics(y_all[te_orig], prob[te_orig])}
    metrics["test_original_nodes"]["ranking"] = ranking_metrics(y_all[te_orig], mc_mean[te_orig])
    metrics["test_original_nodes"]["ece_mc_mean"] = expected_calibration_error(y_all[te_orig], mc_mean[te_orig])
    if mode == "legacy":
        metrics["test_as_in_notebook_incl_synthetic"] = binary_metrics(y_all[te], prob[te])
        metrics["note"] = (
            "Legacy 'test_as_in_notebook_incl_synthetic' reproduces the original protocol and is "
            "inflated by synthetic nodes interpolated from training positives."
        )
    else:
        metrics["note"] = (
            "Labels are curated positives vs UNLABELED genes; metrics measure positive-vs-unlabeled "
            "separability on held-out original genes, not validated biological discovery."
        )

    report.update(
        {
            "n_nodes_original": n_orig,
            "n_nodes_total": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "feature_diagnostics": feat_diag,
            "label_diagnostics": label_diag,
            "ppi_audit": audit_ppi_edges(inputs.edges),
            "n_train_edges": int(train_edges.shape[1]),
            "n_eval_edges": int(eval_edges.shape[1]),
            "split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
            "train": {k: v for k, v in train_info.items() if k != "history"},
            "metrics": metrics,
            "runtime_s": round(time.time() - t0, 1),
        }
    )
    out_dir = Path(cfg["output_dir"]) / cfg.get("run_name", f"gat_{mode}_{inputs.source}") / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pth")
    write_json(
        {
            "architecture": type(model).__name__,
            "in_channels": int(x.shape[1]),
            "model_cfg": mcfg,
            "feature_names_head": feat_names[:20],
            "n_feature_names": len(feat_names),
            "mode": mode,
            "seed": seed,
            "data_source": inputs.source,
        },
        out_dir / "model_meta.json",
    )
    pd.DataFrame(
        {
            "gene": inputs.genes,
            "label": y_all[:n_orig],
            "split": _split_names(n_orig, tr, va, te),
            "prob_eval_mode": prob[:n_orig],
            "mc_mean": mc_mean[:n_orig],
            "mc_std": mc_std[:n_orig],
        }
    ).to_csv(out_dir / "predictions.csv", index=False)
    write_json(train_info["history"], out_dir / "history.json")
    write_json(report, out_dir / "report.json")
    report["output_dir"] = str(out_dir)
    log.info("GAT %s seed=%d done in %.1fs -> %s", mode, seed, report["runtime_s"], out_dir)
    return report


def _split_names(n, tr, va, te):
    s = np.array(["?"] * n, dtype=object)
    s[tr[tr < n]] = "train"
    s[va[va < n]] = "val"
    s[te[te < n]] = "test"
    return s


def run_gat_seeds(cfg: dict, seeds: list[int]) -> dict:
    reports = [run_gat(cfg, s) for s in seeds]
    per_seed = [r["metrics"]["test_original_nodes"] for r in reports]
    summary = {
        "mode": cfg.get("mode"),
        "seeds": seeds,
        "aggregate_test_original_nodes": aggregate_seed_metrics(per_seed),
        "per_seed_output_dirs": [r["output_dir"] for r in reports],
    }
    out = Path(cfg["output_dir"]) / cfg.get("run_name", f"gat_{cfg.get('mode')}_{reports[0]['data_source']}")
    write_json(summary, out / "summary.json")
    return {"summary": summary, "reports": reports}
