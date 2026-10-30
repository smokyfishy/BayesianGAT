"""Command-line interface: ``python -m dmd_nfatc4.cli <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

from .config import DEFAULT_ASSETS, DEFAULT_OUTPUTS, REPO_ROOT, load_config, resolve_path
from .utils import LegacyMethodWarning, sha256sum, write_json

NFATC4_ARRAYS = {
    "saliency": "saliency_vector.npy",
    "std_pred": "std_pred_vector.npy",
    "delta_hidden": "delta_hidden_vector.npy",
    "rsa": "rsa_vector.npy",
    "flags": "simulated_dmd_mutation_vector.npy",
}


def _common(p: argparse.ArgumentParser, config: bool = True) -> None:
    if config:
        p.add_argument("--config", type=Path, help="YAML config file")
    p.add_argument("--data-dir", type=Path, default=None, help=f"assets root (default {DEFAULT_ASSETS})")
    p.add_argument("--output-dir", type=Path, default=None, help=f"outputs root (default {DEFAULT_OUTPUTS})")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)


def _cfg(args) -> dict:
    over = {
        "data_dir": str(args.data_dir) if args.data_dir else None,
        "output_dir": str(args.output_dir) if args.output_dir else None,
        "seed": args.seed,
        "device": args.device,
    }
    return load_config(getattr(args, "config", None), over)


def _print(obj) -> None:
    from .utils import to_jsonable

    print(json.dumps(to_jsonable(obj), indent=2))


# ----------------------------------------------------------------------------- assets
def cmd_verify_hashes(args) -> int:
    manifest = REPO_ROOT / "FILE_SHA256SUMS.txt"
    ok, bad, missing = [], [], []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(maxsplit=1)
        path = REPO_ROOT / rel.strip()
        if not path.exists():
            missing.append(rel)
        elif sha256sum(path) == digest:
            ok.append(rel)
        else:
            bad.append(rel)
    _print({"ok": len(ok), "mismatch": bad, "missing": missing})
    return 1 if bad else 0


def cmd_inspect_assets(args) -> int:
    import pandas as pd

    from .data.alignment import check_sequence_structure_alignment, read_fasta
    from .data.gene_graph import audit_ppi_edges, load_ppi_gene_edges
    from .data.residue_graph import parse_pdb_ca
    from .utils import load_state_dict

    cfg = _cfg(args)
    A = Path(cfg["data_dir"])
    rep: dict = {"assets_dir": str(A), "files": {}}
    for p in sorted(A.rglob("*")):
        if p.is_file():
            entry = {"bytes": p.stat().st_size}
            if p.suffix == ".npy":
                arr = np.load(p, allow_pickle=False, mmap_mode="r")
                entry.update(shape=list(arr.shape), dtype=str(arr.dtype))
            rep["files"][str(p.relative_to(A)).replace("\\", "/")] = entry
    for rel in ("dmd/best_enhanced_bayesian_gat.pth", "nfatc4/bayesian_gvp_model.pt"):
        if (A / rel).exists():
            sd = load_state_dict(A / rel)
            rep["files"][rel]["state_dict"] = {"n_tensors": len(sd), "n_params": int(sum(v.numel() for v in sd.values()))}
    if (A / "dmd/expression_gene_tpm_protein_coding.parquet").exists():
        tpm = pd.read_parquet(A / "dmd/expression_gene_tpm_protein_coding.parquet")
        rep["expression"] = {"shape": list(tpm.shape), "columns": list(tpm.columns), "unique_genes": bool(tpm.index.is_unique)}
    if (A / "dmd/ppi_edges_genes.tsv").exists():
        rep["ppi_gene_edges"] = audit_ppi_edges(load_ppi_gene_edges(A / "dmd/ppi_edges_genes.tsv"))
    if (A / "nfatc4/AF-Q14934-F1-model_v4.pdb").exists():
        res = parse_pdb_ca(A / "nfatc4/AF-Q14934-F1-model_v4.pdb")
        _, seq = read_fasta(A / "nfatc4/Q14934.fasta")
        rep["nfatc4_alignment"] = check_sequence_structure_alignment(seq, res)
    missing = [
        p for p in ("dmd/node_features.parquet", "dmd/augmented_X_y.pt", "dmd/augmented_edge_index.pt") if not (A / p).exists()
    ]
    rep["missing_for_full_gat_reproduction"] = missing
    out = write_json(rep, Path(cfg["output_dir"]) / "asset_report.json")
    _print({k: v for k, v in rep.items() if k != "files"} | {"n_files": len(rep["files"]), "report": str(out)})
    return 0


def cmd_inspect_checkpoint(args) -> int:
    import torch

    from .models import (
        LEGACY_GAT_HPARAMS,
        LEGACY_GAT_IN_CHANNELS,
        LEGACY_RESIDUE_HPARAMS,
        LEGACY_RESIDUE_IN_DIM,
        BayesianGVP_GNN,
        EnhancedBayesianGAT,
    )
    from .utils import load_state_dict

    cfg = _cfg(args)
    A = Path(cfg["data_dir"])
    out: dict = {}
    specs = {
        "gat": (
            A / "dmd/best_enhanced_bayesian_gat.pth",
            lambda: EnhancedBayesianGAT(LEGACY_GAT_IN_CHANNELS, **LEGACY_GAT_HPARAMS),
        ),
        "nfatc4": (A / "nfatc4/bayesian_gvp_model.pt", lambda: BayesianGVP_GNN(LEGACY_RESIDUE_IN_DIM, **LEGACY_RESIDUE_HPARAMS)),
    }
    for name, (path, build) in specs.items():
        if args.which not in (name, "both"):
            continue
        if not path.exists():
            out[name] = {"status": "missing", "path": str(path)}
            continue
        sd = load_state_dict(path)
        model = build()
        ref = model.state_dict()
        shape_mismatch = {
            k: [list(sd[k].shape), list(ref[k].shape)] for k in sd.keys() & ref.keys() if sd[k].shape != ref[k].shape
        }
        entry = {
            "path": str(path),
            "sha256": sha256sum(path),
            "missing_keys": sorted(ref.keys() - sd.keys()),
            "unexpected_keys": sorted(sd.keys() - ref.keys()),
            "shape_mismatches": shape_mismatch,
        }
        try:
            model.load_state_dict(sd, strict=True)
            entry["strict_load"] = "OK"
            model.eval()
            n = 6
            ei = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
            with torch.no_grad():
                if name == "gat":
                    o = model(torch.randn(n, LEGACY_GAT_IN_CHANNELS), ei)
                    entry["temperature"] = float(sd["temperature"])
                else:
                    o = model(torch.randn(n, LEGACY_RESIDUE_IN_DIM), ei)
            entry["forward_smoke_output_shape"] = list(o.shape)
        except RuntimeError as exc:
            entry["strict_load"] = f"FAILED: {exc}"
        out[name] = entry
    _print(out)
    write_json(out, Path(cfg["output_dir"]) / "checkpoint_report.json")
    return 0 if all(v.get("strict_load") == "OK" for v in out.values()) else 1


def cmd_data_instructions(args) -> int:
    from .data.download import download_public, print_data_instructions

    cfg = _cfg(args)
    print_data_instructions(Path(cfg["data_dir"]))
    if args.download_public:
        written = download_public(Path(cfg["data_dir"]), overwrite=args.overwrite)
        print(f"downloaded: {[str(w) for w in written] or 'nothing (files exist)'}")
    return 0


# ----------------------------------------------------------------------------- NFATC4
def cmd_simulate_mutations(args) -> int:
    from .data.expression import gene_log2_fold_change, load_gene_tpm
    from .data.simulation import describe_flags, simulate_mutation_flags

    cfg = _cfg(args)
    A = Path(cfg["data_dir"])
    fc = gene_log2_fold_change(load_gene_tpm(A / "dmd/expression_gene_tpm_protein_coding.parquet"), "NFATC4")
    flags = simulate_mutation_flags(fc["log2_fc"])
    saved = np.load(A / "nfatc4" / NFATC4_ARRAYS["flags"], allow_pickle=False)
    rep = {
        "WARNING": "SIMULATED vector: hand-picked index ranges gated by one group-level fold change. Not observed mutations.",
        "fold_change_from_gene_level_matrix": fc,
        "simulated": describe_flags(flags),
        "identical_to_saved_vector": bool(np.array_equal(flags, saved)),
    }
    out = Path(cfg["output_dir"]) / "nfatc4" / "simulated_mutation_vector_recomputed.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, flags)
    write_json(rep, out.with_suffix(".json"))
    _print(rep)
    return 0


def cmd_build_residue_graph(args) -> int:
    import torch

    from .data.residue_graph import build_residue_graph, compare_graphs
    from .utils import load_pyg_data

    cfg = _cfg(args)
    A = Path(cfg["data_dir"]) / "nfatc4"
    g = build_residue_graph(A / "AF-Q14934-F1-model_v4.pdb", A / "Q14934_esm.npy", A / NFATC4_ARRAYS["flags"], mode=args.mode)
    rep = {"mode": args.mode, "num_nodes": g.num_nodes, "num_edges": int(g.edge_index.shape[1]), "x_shape": list(g.x.shape)}
    if args.compare_saved:
        saved = load_pyg_data(A / "nfatc4_graph.pt", trust_pickle=args.trust_pickle)
        if args.mode == "legacy":
            rep["comparison_with_saved"] = compare_graphs(g, saved)
        else:
            rep["comparison_with_saved"] = {
                "saved_x_dims": int(saved.x.shape[1]),
                "corrected_x_equals_saved_without_last_column": bool(torch.equal(g.x, saved.x[:, :-1])),
            }
    out = Path(cfg["output_dir"]) / "nfatc4" / f"nfatc4_graph_{args.mode}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"x": g.x, "edge_index": g.edge_index, "pos": g.pos, "resseq": g.resseq}, out)
    rep["saved_tensors"] = str(out)
    _print(rep)
    return 0


def _load_saved_arrays(A: Path) -> dict:
    arrays = {k: np.load(A / v, allow_pickle=False) for k, v in NFATC4_ARRAYS.items()}
    n = {k: v.shape for k, v in arrays.items()}
    if len({s for s in n.values()}) != 1:
        raise ValueError(f"saved arrays have inconsistent shapes: {n}")
    return arrays


def cmd_analyze_nfatc4(args) -> int:
    import torch
    from scipy.stats import spearmanr

    from .analysis.clusters import find_clusters, per_residue_table
    from .analysis.domains import (
        CANONICAL_DOMAINS_CELL41,
        CURATED_DOMAINS_CELL44,
        HISTORICAL_CANDIDATE_WINDOWS,
        HISTORICAL_WINDOWS_NOTE,
        annotate_windows,
        compare_windows,
        distance_to_domains,
        distance_to_simulation_boundaries,
    )
    from .analysis.residue_saliency import joint_score_raw, legacy_delta_hidden, legacy_saliency
    from .analysis.structure import compare_rsa, plddt_from_residues, rsa_dssp, rsa_shrake_rupley
    from .analysis.visualization import plot_joint_score, plot_signals
    from .data.residue_graph import build_residue_graph, parse_pdb_ca
    from .models import LEGACY_RESIDUE_HPARAMS, BayesianGVP_GNN
    from .training.uncertainty import mc_dropout_binary
    from .utils import load_state_dict

    cfg = _cfg(args)
    A = Path(cfg["data_dir"]) / "nfatc4"
    out_dir = Path(cfg["output_dir"]) / "nfatc4"
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = _load_saved_arrays(A)
    pdb = A / "AF-Q14934-F1-model_v4.pdb"
    residues = parse_pdb_ca(pdb)
    rep: dict = {
        "provenance": "HISTORICAL saved arrays from the original Colab run (saliency, MC std, delta-hidden, DSSP RSA). "
        "Model was trained with the simulated label as an input feature; outputs are exploratory.",
    }
    raw = joint_score_raw(arr["saliency"], arr["std_pred"], arr["delta_hidden"])
    clusters, smoothed, thr = find_clusters(raw, sigma=args.sigma, percentile=args.percentile, min_len=args.min_len)
    rsa = arr["rsa"]
    clusters["AvgRSA"] = [float(rsa[s : e + 1].mean()) for s, e in zip(clusters.Start_index0, clusters.End_index0)]
    plddt = plddt_from_residues(residues)
    clusters["AvgPLDDT"] = [float(plddt[s : e + 1].mean()) for s, e in zip(clusters.Start_index0, clusters.End_index0)]
    # Original cell 41 used 0-based Start/End directly against the domain numbers.
    clusters["DistToDomain_cell41_as_original"] = [
        distance_to_domains(s, e, CANONICAL_DOMAINS_CELL41) for s, e in zip(clusters.Start_index0, clusters.End_index0)
    ]
    clusters["DistToDomain_cell41_uniprot"] = [
        distance_to_domains(s, e, CANONICAL_DOMAINS_CELL41) for s, e in zip(clusters.Start_uniprot, clusters.End_uniprot)
    ]
    # Windows that straddle an edge of the hand-picked simulated ranges are simulation artefacts.
    clusters["DistToSimulationBoundary"] = [
        distance_to_simulation_boundaries(s, e) for s, e in zip(clusters.Start_uniprot, clusters.End_uniprot)
    ]
    clusters.to_csv(out_dir / "clusters_from_saved_arrays.csv", index=False)
    per_residue_table(arr["saliency"], arr["std_pred"], arr["delta_hidden"], smoothed, rsa).to_csv(
        out_dir / "per_residue_scores_from_saved_arrays.csv", index=False
    )
    windows0 = list(zip(clusters.Start_index0.astype(int), clusters.End_index0.astype(int)))
    rep["threshold_93pct"] = thr
    rep["clusters"] = clusters.to_dict(orient="records")
    rep["historical_windows"] = {"windows": HISTORICAL_CANDIDATE_WINDOWS, "note": HISTORICAL_WINDOWS_NOTE}
    rep["recomputed_vs_historical"] = compare_windows(windows0)
    uni = list(zip(clusters.Start_uniprot.astype(int), clusters.End_uniprot.astype(int)))
    rep["domain_overlap_uniprot_numbering"] = {
        "cell44_table": annotate_windows(uni, CURATED_DOMAINS_CELL44).to_dict(orient="records"),
        "cell41_table": annotate_windows(uni, CANONICAL_DOMAINS_CELL41).to_dict(orient="records"),
        "caveat": "Both domain tables are historical, uncited and mutually inconsistent.",
    }
    # Structural annotation of RSA
    sr = rsa_shrake_rupley(pdb)
    rep["rsa"] = {"saved_dssp_vs_shrake_rupley": compare_rsa(rsa, sr)}
    d = rsa_dssp(pdb)
    rep["rsa"]["dssp_available"] = d is not None
    if d is not None:
        rep["rsa"]["saved_vs_local_dssp"] = compare_rsa(rsa, d)
    rep["plddt_summary"] = {
        "mean": float(np.nanmean(plddt)),
        "frac_below_50": float(np.mean(plddt < 50)),
        "frac_below_70": float(np.mean(plddt < 70)),
    }
    plot_joint_score(raw, smoothed, thr, clusters, out_dir / "joint_score_saved_arrays.png")
    plot_signals(
        {"saliency": arr["saliency"], "MC std": arr["std_pred"], "delta hidden": arr["delta_hidden"], "RSA": rsa, "pLDDT": plddt},
        arr["flags"],
        out_dir / "signals_saved_arrays.png",
    )

    if args.recompute:
        g = build_residue_graph(pdb, A / "Q14934_esm.npy", A / NFATC4_ARRAYS["flags"], mode="legacy")
        m = BayesianGVP_GNN(g.x.shape[1], **LEGACY_RESIDUE_HPARAMS)
        m.load_state_dict(load_state_dict(A / "bayesian_gvp_model.pt"), strict=True)
        s1 = legacy_saliency(m, g.x, g.edge_index)
        rc: dict = {
            "saliency_saved_over_single_pass_median_ratio": float(np.median(arr["saliency"] / s1)),
            "saliency_two_pass_max_rel_diff": float(
                np.max(np.abs(legacy_saliency(m, g.x, g.edge_index, accumulate_passes=2) - arr["saliency"]) / arr["saliency"])
            ),
        }
        d_eval = legacy_delta_hidden(m, g.x, g.edge_index, dropout_active=False)
        rc["delta_hidden_eval_mode_spearman_vs_saved"] = float(spearmanr(d_eval, arr["delta_hidden"])[0])
        stab = []
        hist = set(HISTORICAL_CANDIDATE_WINDOWS)
        for r in range(args.stability):
            torch.manual_seed(1000 + r)
            _, sd = mc_dropout_binary(m, g.x, g.edge_index, mc_samples=50)
            dh = legacy_delta_hidden(m, g.x, g.edge_index, dropout_active=True)
            cl, _, _ = find_clusters(joint_score_raw(s1 * 2, sd, dh), args.sigma, args.percentile, args.min_len)
            w = set(zip(cl.Start_index0.astype(int), cl.End_index0.astype(int)))
            ov = sum(any(max(a, c) <= min(b, e) for c, e in w) for a, b in hist)
            stab.append(
                {
                    "repeat": r,
                    "std_spearman_vs_saved": float(spearmanr(sd, arr["std_pred"])[0]),
                    "delta_hidden_spearman_vs_saved": float(spearmanr(dh, arr["delta_hidden"])[0]),
                    "n_windows": len(w),
                    "historical_windows_exactly_recovered": len(w & hist),
                    "historical_windows_overlapped": int(ov),
                }
            )
        rc["stability_reruns"] = stab
        rc["interpretation"] = (
            "Saliency is deterministic (saved = 2x single pass due to gradient accumulation in cell 33). "
            "MC std and dropout-active delta-hidden are stochastic; windows depend on the random draw."
        )
        rep["recompute_from_checkpoint"] = rc
    write_json(rep, out_dir / "analysis_report.json")
    _print(
        {
            "clusters": clusters[
                ["Start_index0", "End_index0", "Start_uniprot", "End_uniprot", "Length", "AvgJointScore", "AvgRSA"]
            ].to_dict(orient="records"),
            "recomputed_vs_historical": rep["recomputed_vs_historical"],
            "rsa": rep["rsa"],
            "recompute_from_checkpoint": rep.get("recompute_from_checkpoint", "skipped (use --recompute)"),
            "report": str(out_dir / "analysis_report.json"),
        }
    )
    return 0


def cmd_visualize_nfatc4(args) -> int:
    import pandas as pd

    from .analysis.domains import HISTORICAL_CANDIDATE_WINDOWS
    from .analysis.visualization import write_pml, write_py3dmol_html

    cfg = _cfg(args)
    A = Path(cfg["data_dir"]) / "nfatc4"
    out_dir = Path(cfg["output_dir"]) / "nfatc4"
    csv = out_dir / "clusters_from_saved_arrays.csv"
    if args.windows == "historical-as-index0":
        windows = [(s + 1, e + 1) for s, e in HISTORICAL_CANDIDATE_WINDOWS]
    else:
        if not csv.exists():
            print("run `analyze-nfatc4 --from-saved-arrays` first", file=sys.stderr)
            return 2
        c = pd.read_csv(csv)
        windows = list(zip(c.Start_uniprot.astype(int), c.End_uniprot.astype(int)))
    pdb = (A / "AF-Q14934-F1-model_v4.pdb").resolve()
    written = []
    for fmt in args.format:
        if fmt == "pml":
            written.append(write_pml(pdb, windows, out_dir / "nfatc4_windows.pml", png_name="nfatc4_windows_pymol.png"))
        elif fmt == "html":
            written.append(write_py3dmol_html(pdb, windows, out_dir / "nfatc4_windows.html"))
        elif fmt == "png":
            written += [out_dir / "joint_score_saved_arrays.png", out_dir / "signals_saved_arrays.png"]
            if not written[-1].exists():
                print("PNG figures are produced by analyze-nfatc4", file=sys.stderr)
    _print({"windows_uniprot_1based": windows, "files": [str(w) for w in written]})
    return 0


def cmd_train_residue(args) -> int:
    from .training.train_residue import train_residue_corrected, train_residue_legacy

    cfg = _cfg(args)
    rep = train_residue_legacy(cfg) if cfg.get("mode") == "legacy" else train_residue_corrected(cfg)
    _print(rep)
    return 0


# ----------------------------------------------------------------------------- DMD GAT
def cmd_train_gat(args) -> int:
    from .training.train_gat import run_gat_seeds

    cfg = _cfg(args)
    seeds = args.seeds or [int(cfg["seed"])]
    if args.epochs:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    res = run_gat_seeds(cfg, seeds)
    brief = []
    for r in res["reports"]:
        m = r["metrics"]["test_original_nodes"]
        b = {
            "seed": r["seed"],
            "test_original": {
                k: m.get(k)
                for k in ("n", "n_positive", "roc_auc", "pr_auc", "pr_auc_random_baseline", "macro_f1", "recall_pos_sensitivity")
            },
            "ranking": {k: v for k, v in m["ranking"].items() if k != "positive_ranks"},
            "leakage_audit": r.get("leakage_audit"),
            "output_dir": r["output_dir"],
        }
        if "test_as_in_notebook_incl_synthetic" in r["metrics"]:
            t = r["metrics"]["test_as_in_notebook_incl_synthetic"]
            b["test_as_in_notebook_incl_synthetic"] = {k: t.get(k) for k in ("n", "n_positive", "roc_auc", "macro_f1")}
        brief.append(b)
    _print({"summary": res["summary"], "runs": brief})
    return 0


def cmd_rank_genes(args) -> int:
    import pandas as pd

    from .analysis.gene_ranking import gene_report, zscore_ranking
    from .data.labels import KNOWN_DMD_GENES

    cfg = _cfg(args)
    if args.predictions:
        pred = pd.read_csv(args.predictions)
        positives = KNOWN_DMD_GENES if args.positives == "dmd" else pred.loc[pred.label == 1, "gene"].tolist()
        df = zscore_ranking(pred.gene.tolist(), pred.mc_mean.to_numpy(), pred.mc_std.to_numpy(), positives)
        source = str(args.predictions)
    else:
        from .analysis.gene_ranking import rank_from_checkpoint
        from .models import LEGACY_GAT_HPARAMS

        node_fp = args.node_table or resolve_path("dmd/node_features.parquet", cfg["data_dir"])
        ckpt = args.checkpoint or resolve_path("dmd/best_enhanced_bayesian_gat.pth", cfg["data_dir"])
        if not Path(node_fp).exists():
            _print(
                {
                    "status": "BLOCKED",
                    "reason": f"Ranking with the original checkpoint needs {node_fp} (~111 MB, not in starter) and, "
                    "for an exact reproduction, augmented_X_y.pt + augmented_edge_index.pt (the original ranking ran "
                    "on the augmented graph). Alternatively pass --predictions <run>/predictions.csv from train-gat.",
                }
            )
            return 2
        hp = dict(LEGACY_GAT_HPARAMS)
        if args.meta:
            meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
            hp.update({k: v for k, v in meta.get("model_cfg", {}).items() if k in hp})
        df, info = rank_from_checkpoint(
            node_fp,
            args.ppi_edges or resolve_path("dmd/ppi_edges_genes.tsv", cfg["data_dir"]),
            ckpt,
            hp,
            KNOWN_DMD_GENES if args.positives == "dmd" else args.positive_list or KNOWN_DMD_GENES,
            knn_k=args.knn_k or None,
            seed=int(cfg["seed"]),
        )
        source = f"checkpoint {ckpt} on {node_fp} ({info})"
    out = Path(cfg["output_dir"]) / "gene_ranking.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rep = {
        "source": source,
        "caveat": "Ranking = similarity to curated positives under the model; not causality or validated DMD involvement.",
        f"top_{args.top_k}": df.head(args.top_k)[["rank", "gene", "mean_prob", "std_prob", "z_score"]].to_dict(orient="records"),
        "genes_of_interest": [gene_report(df, g) for g in args.gene],
        "csv": str(out),
    }
    _print(rep)
    return 0


def cmd_rwr_genes(args) -> int:
    from .analysis.gene_ranking import gene_report, random_walk_with_restart, rwr_table
    from .data.expression import load_gene_tpm
    from .data.gene_graph import load_ppi_gene_edges, ppi_edge_index
    from .data.labels import KNOWN_DMD_GENES

    cfg = _cfg(args)
    A = Path(cfg["data_dir"])
    genes = list(load_gene_tpm(A / "dmd/expression_gene_tpm_protein_coding.parquet").index)
    ei, _ = ppi_edge_index(load_ppi_gene_edges(A / "dmd/ppi_edges_genes.tsv"), genes)
    seeds = [i for i, g in enumerate(genes) if g in set(KNOWN_DMD_GENES)]
    p, iters, conv = random_walk_with_restart(ei.numpy(), len(genes), seeds, restart=args.restart)
    df = rwr_table(genes, p)
    out = Path(cfg["output_dir"]) / "rwr_ppi_only.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    rep = {
        "graph": "STRING PPI only (supplied ppi_edges_genes.tsv) over the 19 922 protein-coding genes of the TPM matrix. "
        "The notebook ran RWR on PPI + kNN edges over the 18 479-gene node table, so values differ.",
        "n_nodes": len(genes),
        "n_undirected_edges": int(ei.shape[1] // 2),
        "n_seeds": len(seeds),
        "iterations": iters,
        "converged": conv,
        "top_20": df.head(20).to_dict(orient="records"),
        "genes_of_interest": [gene_report(df, g) for g in args.gene],
        "caveat": "RWR rank reflects network proximity to the curated seed list, which includes many hub genes "
        "(TNF, IL6, TGFB1, ...). It is not evidence of DMD involvement.",
        "csv": str(out),
    }
    _print(rep)
    return 0


def cmd_demo(args) -> int:
    from .training.train_gat import run_gat

    cfg = _cfg(args)
    base = {
        "data_dir": cfg["data_dir"],
        "output_dir": str(Path(cfg["output_dir"]) / "demo_synthetic"),
        "device": cfg["device"],
        "data": {"source": "synthetic", "synthetic": {"n_genes": 300, "n_pos": 24, "seed": 0}},
        "model": {"hidden_channels": 16, "heads_1": 2, "heads_2": 2, "dropout": 0.5},
        "training": {"epochs": args.epochs, "lr": 0.005, "weight_decay": 5e-4},
        "mc_samples": 10,
    }
    legacy = run_gat(dict(base, mode="legacy", graph={"knn": True, "knn_k": 5}, augmentation={"ratio": 0.25}), seed=0)
    corrected = run_gat(
        dict(
            base,
            mode="corrected",
            graph={"protocol": "inductive", "knn": True, "knn_k": 5},
            augmentation={"enabled": True, "ratio": 0.25},
        ),
        seed=0,
    )

    def pick(r):
        m = r["metrics"]["test_original_nodes"]
        return {"roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "n_test": m["n"], "n_pos_test": m["n_positive"]}

    _print(
        {
            "WARNING": "SYNTHETIC fixture -- software demonstration only, no biological meaning.",
            "legacy": {
                "as_in_notebook_incl_synthetic": {
                    k: legacy["metrics"]["test_as_in_notebook_incl_synthetic"][k]
                    for k in ("roc_auc", "macro_f1", "n", "n_positive")
                },
                "original_test_nodes_only": pick(legacy),
                "leakage_audit": legacy["leakage_audit"],
            },
            "corrected_inductive": {"original_test_nodes": pick(corrected), "leakage_audit": corrected.get("leakage_audit")},
        }
    )
    return 0


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dmd_nfatc4", description="DMD Bayesian GAT + NFATC4 residue-graph research CLI")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("verify-hashes", help="check files against FILE_SHA256SUMS.txt")
    s.set_defaults(func=cmd_verify_hashes)

    s = sub.add_parser("inspect-assets", help="shapes, checkpoints, PPI audit, sequence/structure alignment")
    _common(s, config=False)
    s.set_defaults(func=cmd_inspect_assets)

    s = sub.add_parser("inspect-checkpoint", help="strict-load the original checkpoints into the legacy classes")
    _common(s, config=False)
    s.add_argument("--which", choices=["gat", "nfatc4", "both"], default="both")
    s.set_defaults(func=cmd_inspect_checkpoint)

    s = sub.add_parser("data-instructions", help="where to obtain each dataset (opt-in public download)")
    _common(s, config=False)
    s.add_argument("--download-public", action="store_true", help="fetch UniProt FASTA + AlphaFold PDB")
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_data_instructions)

    s = sub.add_parser("simulate-mutations", help="re-derive the SIMULATED NFATC4 mutation vector")
    _common(s, config=False)
    s.set_defaults(func=cmd_simulate_mutations)

    s = sub.add_parser("build-residue-graph", help="rebuild the 8 A Ca contact graph from PDB + ESM")
    _common(s, config=False)
    s.add_argument("--mode", choices=["legacy", "corrected"], default="corrected")
    s.add_argument("--compare-saved", action="store_true", help="compare with assets/nfatc4/nfatc4_graph.pt")
    s.add_argument("--trust-pickle", action="store_true", help="allow full unpickling if safe loading fails")
    s.set_defaults(func=cmd_build_residue_graph)

    s = sub.add_parser("analyze-nfatc4", help="joint score, clusters, RSA, domains from saved arrays")
    _common(s, config=False)
    s.add_argument("--from-saved-arrays", action="store_true", default=True, help="(default) use historical arrays")
    s.add_argument("--recompute", action="store_true", help="also recompute signals from the checkpoint")
    s.add_argument("--stability", type=int, default=5, help="stochastic reruns for --recompute")
    s.add_argument("--sigma", type=float, default=2.0)
    s.add_argument("--percentile", type=float, default=93.0)
    s.add_argument("--min-len", type=int, default=5)
    s.set_defaults(func=cmd_analyze_nfatc4)

    s = sub.add_parser("visualize-nfatc4", help="write PyMOL .pml / py3Dmol .html / list PNGs")
    _common(s, config=False)
    s.add_argument("--format", nargs="+", choices=["png", "pml", "html"], default=["png", "pml"])
    s.add_argument("--windows", choices=["recomputed", "historical-as-index0"], default="recomputed")
    s.set_defaults(func=cmd_visualize_nfatc4)

    s = sub.add_parser("train-residue", help="NFATC4 residue model (legacy reproduction or corrected)")
    _common(s)
    s.set_defaults(func=cmd_train_residue)

    s = sub.add_parser("train-gat", help="gene-graph GAT (legacy reproduction or corrected protocol)")
    _common(s)
    s.add_argument("--seeds", type=int, nargs="+", help="repeat over several seeds")
    s.add_argument("--epochs", type=int, help="override training.epochs")
    s.set_defaults(func=cmd_train_gat)

    s = sub.add_parser("rank-genes", help="z-score prioritisation of unlabeled genes (cell 73/79)")
    _common(s, config=False)
    s.add_argument("--predictions", type=Path, help="predictions.csv written by train-gat")
    s.add_argument("--checkpoint", type=Path, help="GAT state_dict (default: original checkpoint in assets)")
    s.add_argument("--meta", type=Path, help="model_meta.json for non-legacy hyperparameters")
    s.add_argument("--node-table", type=Path, help="node_features.parquet (default assets/dmd/)")
    s.add_argument("--ppi-edges", type=Path, help="ppi_edges_genes.tsv (default assets/dmd/)")
    s.add_argument("--knn-k", type=int, default=10, help="kNN edges added to PPI (0 = PPI only)")
    s.add_argument("--positive-list", nargs="+", help=argparse.SUPPRESS)
    s.add_argument("--positives", choices=["dmd", "file"], default="dmd", help="exclusion set: DMD list or run labels")
    s.add_argument("--top-k", type=int, default=25)
    s.add_argument("--gene", nargs="+", default=["NFATC4", "NFATC2"])
    s.set_defaults(func=cmd_rank_genes)

    s = sub.add_parser("rwr-genes", help="random walk with restart from curated DMD seeds on the STRING PPI graph")
    _common(s, config=False)
    s.add_argument("--restart", type=float, default=0.3)
    s.add_argument("--gene", nargs="+", default=["NFATC4", "NFATC2"])
    s.set_defaults(func=cmd_rwr_genes)

    s = sub.add_parser("demo", help="end-to-end legacy vs corrected run on a SYNTHETIC fixture (CPU, <1 min)")
    _common(s, config=False)
    s.add_argument("--epochs", type=int, default=40)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if not args.verbose:
        warnings.filterwarnings("once", category=LegacyMethodWarning)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
