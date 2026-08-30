"""Reproducibility entrypoint for the ICML 2026 Bonbon phenomics paper.

Runs the full evaluation pipeline from frozen Bonbon embeddings through all
benchmark results reported in the paper. Each stage can be run independently.

Prerequisites:
    - Python environment with: torch, numpy, pandas, scipy, sklearn, xgboost,
      efaar_benchmarking, bonbontoken
    - Frozen Bonbon checkpoint: dark-snowball-245
    - RxRx3-core data at /opt/dlami/nvme/rxrx3_phenomics/data/
    - Bonbon embeddings at /opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245/

Usage:
    source ~/output/bin/activate

    # Run everything
    python analysis/reproduce.py --stage all

    # Run individual stages
    python analysis/reproduce.py --stage embed      # Extract embeddings
    python analysis/reproduce.py --stage phenomics  # CC AUROC + per-target
    python analysis/reproduce.py --stage efaar      # JUMP-CP benchmark
    python analysis/reproduce.py --stage attention   # Active moiety analysis

    # Just print the summary from existing results
    python analysis/reproduce.py --stage summary
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent.resolve()
EVAL_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation")
JUMP_DIR = Path("/opt/dlami/nvme/jump_cp/data")


def run_script(name, description):
    script = ANALYSIS_DIR / name
    if not script.exists():
        print(f"  [SKIP] {name} not found at {script}")
        return False
    print(f"\n{'─' * 60}")
    print(f"  Running: {description}")
    print(f"  Script:  {name}")
    print(f"{'─' * 60}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ANALYSIS_DIR.parent),
    )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"  [{status}] {name} — {elapsed:.1f}s")
    return result.returncode == 0


def stage_embed():
    print("\n" + "=" * 60)
    print("STAGE 1: EMBEDDING EXTRACTION")
    print("=" * 60)
    ok = True
    ok &= run_script("download_human_proteome.py", "Download human proteome from UniProt")
    ok &= run_script("build_interaction_prints.py", "Build interaction-print matrices")
    ok &= run_script("extract_fusion_cls.py", "Extract fusion CLS tokens for all pairs")
    ok &= run_script("proteome_zero_shot.py", "Compute proteome-scale zero-shot CG scores")
    return ok


def stage_phenomics():
    print("\n" + "=" * 60)
    print("STAGE 2: PHENOMICS BENCHMARKS (RxRx3)")
    print("=" * 60)
    ok = True
    ok &= run_script("comprehensive_pipeline.py", "CC AUROC: pair-level CV with XGBoost + MLP")
    ok &= run_script("clean_compound_cv.py", "CC AUROC: compound-level CV with per-fold signatures")
    ok &= run_script("per_target_global.py", "Per-target AUROC: zero-shot and trained models")
    return ok


def stage_efaar():
    print("\n" + "=" * 60)
    print("STAGE 3: EFAAR KNOWN-RELATIONSHIP BENCHMARK (JUMP-CP)")
    print("=" * 60)
    return run_script("efaar_bonbon_benchmark.py", "EFAAR benchmark on Bonbon proteome embeddings")


def stage_attention():
    print("\n" + "=" * 60)
    print("STAGE 4: ACTIVE MOIETY ATTENTION ANALYSIS")
    print("=" * 60)
    return run_script("active_moiety_attention.py", "Active moiety attention (Phase 16)")


def load_json(filename):
    path = EVAL_DIR / filename
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def fmt(v):
    return f"{v:.3f}" if isinstance(v, (int, float)) else str(v)


def print_summary():
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY — Bonbon Phenomics (dark-snowball-245)")
    print("=" * 70)

    # ── CC AUROC (pair-level CV) ──
    mlp_imp = load_json("mlp_improvements_results.json")
    if mlp_imp:
        print("\n  CC AUROC (pair-level CV, @0.6)")
        print("  " + "─" * 50)
        best_mlp = max(
            ((k, v["mean"]) for k, v in mlp_imp.items()
             if isinstance(v, dict) and "mean" in v),
            key=lambda x: x[1], default=None,
        )
        if best_mlp:
            print(f"    MLP ({best_mlp[0]}):  {best_mlp[1]*100:.1f}%")
        print(f"    Beaini et al. (no CV):       76.0%")

    # ── CC AUROC (compound-level CV) ──
    cv = load_json("clean_compound_cv_results.json")
    if cv:
        print("\n  CC AUROC (compound-level CV, @0.6)")
        print("  " + "─" * 50)
        for k, v in cv.items():
            if "0.6" in k and isinstance(v, dict) and "mean" in v:
                label = "Per-fold sig" if "clean" in k else "Global sig"
                model = "XGB" if "xgb" in k and "reg" not in k else "Ensemble" if "ensemble" in k else "MLP" if "mlp" in k else k
                print(f"    {label} {model:15s}  {v['mean']*100:.1f}%")

    # ── Per-target AUROC ──
    cb_concat = load_json("codebook_concat_results.json")
    pt_cls = load_json("per_target_fusion_cls_results.json")
    pt_auroc = load_json("per_target_auroc.json")
    if cb_concat or pt_cls or pt_auroc:
        print("\n  Per-target AUROC (median)")
        print("  " + "─" * 50)
        # Zero-shot from per_target_auroc.json (list of dicts)
        if pt_auroc and isinstance(pt_auroc, list):
            best_zs = max(pt_auroc, key=lambda x: x.get("median_auroc", 0))
            print(f"    {'Zero-shot (' + best_zs['method'] + ')':35s}  {best_zs['median_auroc']*100:.1f}%")
        # Trained: fusion CLS MLP
        if pt_cls and isinstance(pt_cls, dict):
            for k in ["global_mlp_1024", "cls_cg_combined_mlp"]:
                if k in pt_cls and isinstance(pt_cls[k], (int, float)):
                    print(f"    {'Trained fusion CLS (' + k + ')':35s}  {pt_cls[k]*100:.1f}%")
        # Trained: codebook tokens (the 89.9% headline)
        if cb_concat and isinstance(cb_concat, dict):
            for k in ["mol_tokens_1280", "tokens_concat_2560"]:
                if k in cb_concat and isinstance(cb_concat[k], (int, float)):
                    print(f"    {'Trained codebook (' + k + ')':35s}  {cb_concat[k]*100:.1f}%")
        print(f"    {'Beaini et al. (Boltz-2)':35s}  53.9%")

    # ── EFAAR ──
    efaar_path = JUMP_DIR / "efaar_results.json"
    efaar = None
    if efaar_path.exists():
        with open(efaar_path) as f:
            efaar = json.load(f)
    if efaar and isinstance(efaar, dict):
        print("\n  EFAAR Known-Relationship Recall@0.05/0.95 (JUMP-CP, 7,976 genes)")
        print("  " + "─" * 66)
        print(f"    {'Method':25s} {'CORUM':>7s} {'HuMAP':>7s} {'React.':>7s} {'SIGNOR':>7s} {'StrDB':>7s}")
        print("    " + "─" * 62)
        order = ["cellprofiler", "projector", "cb_tokens", "cb_pa", "fusion"]
        labels = {"cellprofiler": "CellProfiler", "projector": "Bonbon projector",
                  "cb_tokens": "Bonbon cb_tokens", "cb_pa": "Bonbon cb_pa",
                  "fusion": "Bonbon fusion"}
        for key in order:
            if key in efaar:
                r = efaar[key]
                print(f"    {labels.get(key, key):25s} {fmt(r.get('CORUM','-')):>7s} "
                      f"{fmt(r.get('HuMAP','-')):>7s} {fmt(r.get('Reactome','-')):>7s} "
                      f"{fmt(r.get('SIGNOR','-')):>7s} {fmt(r.get('StringDB','-')):>7s}")
        print(f"    {'MAE-G/8 (published)':25s} {'0.264':>7s} {'0.215':>7s} {'0.165':>7s} {'—':>7s} {'0.235':>7s}")

    # ── Active moiety ──
    attn = load_json("active_moiety_attention.json")
    if attn and "metadata" in attn:
        meta = attn["metadata"]
        n_analyzed = meta.get("n_targets_analyzed", 1)
        n_sig = meta.get("n_significant_p005", 0)
        print("\n  Active Moiety Attention")
        print("  " + "─" * 50)
        print(f"    Targets analyzed:        {n_analyzed}")
        print(f"    Significant (p<0.05):    {n_sig} ({n_sig/max(n_analyzed,1)*100:.0f}%)")

    has_results = any([mlp_imp, cv, efaar, attn])
    if not has_results:
        print("\n  No result files found.")
        print(f"  Expected at: {EVAL_DIR}")
        print("  Run --stage all to generate results.")

    print("\n" + "=" * 70)
    print(f"  Eval dir:  {EVAL_DIR}")
    print(f"  JUMP dir:  {JUMP_DIR}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce all results from the ICML 2026 Bonbon phenomics paper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage",
        choices=["embed", "phenomics", "efaar", "attention", "summary", "all"],
        default="summary",
        help="Pipeline stage to run (default: summary)",
    )
    args = parser.parse_args()

    t0 = time.time()

    if args.stage in ("embed", "all"):
        stage_embed()
    if args.stage in ("phenomics", "all"):
        stage_phenomics()
    if args.stage in ("efaar", "all"):
        stage_efaar()
    if args.stage in ("attention", "all"):
        stage_attention()

    print_summary()

    if args.stage != "summary":
        print(f"  Total wall time: {time.time() - t0:.1f}s\n")


if __name__ == "__main__":
    main()
