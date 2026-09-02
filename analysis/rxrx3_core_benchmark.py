"""RxRx3-core EFAAR benchmark: Bonbon vs. Kraus et al. 2025 baselines.

Runs EFAAR known-relationship benchmarking on the RxRx3-core 735-gene
CRISPR subset using Bonbon protein embeddings and the OpenPhenom baseline
embeddings provided with the dataset.

Published baselines from Kraus et al. (LMRL Workshop, ICLR 2025):
  Table 6 — Multivariate known biological relationship recall (10% extremes)
    Method         CORUM  hu.MAP  Reactome  StringDB
    Random          .107   .111     .107      .115
    CellProfiler    .361   .444     .160      .330
    OpenPhenom-S/16 .300   .352     .158      .281
    Phenom-1        .395   .482     .188      .349
    Phenom-2        .486   .553     .197      .415

Usage:
    source ~/output/bin/activate
    python analysis/rxrx3_core_benchmark.py
"""

import os
import sys
import time
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import Bunch
from efaar_benchmarking.benchmarking import known_relationship_benchmark
from efaar_benchmarking.constants import BENCHMARK_SOURCES

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
PROTEOME_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/human_proteome")
DATA_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/data")
RESULTS_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation")

EMB_PATHS = {
    "projector": PROTEOME_DIR / "human_proteome_pairs_protein_protein_mean_dark-snowball-245",
    "cb_tokens": PROTEOME_DIR / "human_proteome_pairs_protein_codebook_dark-snowball-245" / "tokens",
    "cb_pa": PROTEOME_DIR / "human_proteome_pairs_protein_codebook_dark-snowball-245" / "pooled_attention",
}

# Two RxRx3 gene symbols map to alternate names in our proteome embeddings
GENE_ALIASES = {
    "C7orf26": "INTS15",
    "MPP5": "PALS1",
}

# Published baselines from Kraus et al. 2025, Table 6
# recall at 10% extremes (top 5% + bottom 5%)
PUBLISHED_BASELINES = {
    "Random": {"CORUM": 0.107, "HuMAP": 0.111, "Reactome": 0.107, "StringDB": 0.115},
    "CellProfiler": {"CORUM": 0.361, "HuMAP": 0.444, "Reactome": 0.160, "StringDB": 0.330},
    "OpenPhenom-S/16": {"CORUM": 0.300, "HuMAP": 0.352, "Reactome": 0.158, "StringDB": 0.281},
    "Phenom-1 (MAE-G/8)": {"CORUM": 0.395, "HuMAP": 0.482, "Reactome": 0.188, "StringDB": 0.349},
    "Phenom-2": {"CORUM": 0.486, "HuMAP": 0.553, "Reactome": 0.197, "StringDB": 0.415},
}


def p(*a, **k):
    print(*a, **k, flush=True)


# ---------------------------------------------------------------------------
# embedding loaders
# ---------------------------------------------------------------------------
def load_bonbon_embeddings(path, gene_filter=None, aliases=None):
    """Load .pt embeddings, applying gene_filter and aliases."""
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    genes, embs = [], []

    # Build reverse alias map: embedding filename -> rxrx3 gene name
    reverse_alias = {}
    if aliases:
        reverse_alias = {v: k for k, v in aliases.items()}

    for f in files:
        emb_name = f.replace(".pt", "")
        # Determine gene name: use original rxrx3 name if this is an alias
        gene = reverse_alias.get(emb_name, emb_name)

        if gene_filter is not None and gene not in gene_filter:
            continue

        e = torch.load(os.path.join(path, f), map_location="cpu", weights_only=True)
        if e.dim() > 1:
            e = e.float().mean(dim=0)
        else:
            e = e.float()
        genes.append(gene)
        embs.append(e.numpy())

    return genes, np.stack(embs)


def load_openphenom_baseline(meta):
    """Load and process OpenPhenom embeddings following Kraus et al. pipeline."""
    emb_df = pd.read_parquet(DATA_DIR / "OpenPhenom_rxrx3_core_embeddings.parquet")
    merged = meta.merge(emb_df, on="well_id")
    crispr = merged[merged["perturbation_type"] == "CRISPR"].copy()
    fcols = [c for c in merged.columns if c.startswith("feature_")]
    features = crispr[fcols].values.astype(np.float32)
    md = crispr[["gene", "plate"]].reset_index(drop=True)

    # TVN normalization (same as Kraus et al.)
    features_tvn = tvn_on_controls(features, md, "gene", "EMPTY_control", "plate")

    # Aggregate per gene
    md_reset = md.reset_index(drop=True)
    agg_embs, agg_genes = [], []
    for gene, group in md_reset.groupby("gene"):
        if gene in ["EMPTY_control", "CRISPR_control"]:
            continue
        agg_embs.append(np.mean(features_tvn[group.index.values], axis=0))
        agg_genes.append(gene)

    return agg_genes, np.vstack(agg_embs)


# ---------------------------------------------------------------------------
# normalization (TVN, matching EFAAR standard)
# ---------------------------------------------------------------------------
def centerscale_on_controls(embeddings, metadata, pert_col, control_key, batch_col=None):
    embeddings = embeddings.copy()
    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            bi = metadata[batch_col] == batch
            ci = bi & (metadata[pert_col] == control_key)
            if ci.sum() > 1:
                embeddings[bi] = StandardScaler().fit(embeddings[ci]).transform(embeddings[bi])
        return embeddings
    ci = metadata[pert_col] == control_key
    return StandardScaler().fit(embeddings[ci]).transform(embeddings)


def tvn_on_controls(embeddings, metadata, pert_col, control_key, batch_col=None):
    from scipy import linalg

    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key, batch_col)
    ci = metadata[pert_col] == control_key
    embeddings = PCA().fit(embeddings[ci]).transform(embeddings)
    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            bi = metadata[batch_col] == batch
            bci = bi & (metadata[pert_col] == control_key)
            if bci.sum() > 2:
                cov = np.cov(embeddings[bci], rowvar=False, ddof=1) + 0.5 * np.eye(embeddings.shape[1])
                embeddings[bi] = embeddings[bi] @ linalg.fractional_matrix_power(cov, -0.5)
    return embeddings


# ---------------------------------------------------------------------------
# benchmarking
# ---------------------------------------------------------------------------
def run_benchmark(genes, embeddings, label, sources=None):
    """Run EFAAR known-relationship benchmark."""
    if sources is None:
        sources = BENCHMARK_SOURCES
    map_data = Bunch(
        features=pd.DataFrame(embeddings),
        metadata=pd.DataFrame({"gene": genes}),
    )
    results = known_relationship_benchmark(
        map_data,
        pert_col="gene",
        benchmark_sources=sources,
        log_stats=True,
    )
    p(f"\n  {label}:")
    p(results[["source", "recall_0.05_0.95", "recall_0.1_0.9"]].to_string(index=False))
    return results


def format_comparison_table(all_results):
    """Pretty-print comparison table against published baselines."""
    # EFAAR uses CORUM, HuMAP, Reactome, SIGNOR, StringDB
    # Kraus et al. reports CORUM, hu.MAP, Reactome, StringDB (no SIGNOR)
    sources_compare = ["CORUM", "HuMAP", "Reactome", "StringDB"]

    p("\n" + "=" * 90)
    p("COMPARISON TABLE: recall at 10% extremes (5%/95% thresholds)")
    p("=" * 90)
    header = f"{'Method':<35} {'CORUM':>8} {'HuMAP':>8} {'Reactome':>10} {'StringDB':>10} {'Mean':>8}"
    p(header)
    p("-" * 90)

    # Published baselines
    for method, vals in PUBLISHED_BASELINES.items():
        scores = [vals.get(s, float("nan")) for s in sources_compare]
        mean_score = np.nanmean(scores)
        p(f"{method:<35} {scores[0]:>8.3f} {scores[1]:>8.3f} {scores[2]:>10.3f} {scores[3]:>10.3f} {mean_score:>8.3f}")

    p("-" * 90)

    # Our results
    for method, df in all_results.items():
        scores = []
        for src in sources_compare:
            row = df[df["source"] == src]
            if len(row) > 0:
                scores.append(row.iloc[0]["recall_0.05_0.95"])
            else:
                scores.append(float("nan"))
        mean_score = np.nanmean(scores)
        p(f"{method:<35} {scores[0]:>8.3f} {scores[1]:>8.3f} {scores[2]:>10.3f} {scores[3]:>10.3f} {mean_score:>8.3f}")

    p("=" * 90)

    # Also show SIGNOR (not in Kraus et al.)
    p("\nAdditional: SIGNOR recall (not in Kraus et al.)")
    for method, df in all_results.items():
        row = df[df["source"] == "SIGNOR"]
        if len(row) > 0:
            p(f"  {method:<35} {row.iloc[0]['recall_0.05_0.95']:.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    p("=" * 70)
    p("RxRx3-core EFAAR BENCHMARK: Bonbon vs. Kraus et al. 2025")
    p("=" * 70)

    # Load metadata
    meta = pd.read_csv(DATA_DIR / "metadata_rxrx3_core.csv", low_memory=False)
    crispr = meta[meta["perturbation_type"] == "CRISPR"]
    rxrx3_genes = set(
        g for g in crispr["gene"].unique()
        if g not in ["EMPTY_control", "CRISPR_control"]
    )
    p(f"RxRx3-core CRISPR genes: {len(rxrx3_genes)}")

    # Build filter set including aliases
    gene_filter = set()
    for g in rxrx3_genes:
        gene_filter.add(GENE_ALIASES.get(g, g))
    # But we also keep original names in the filter for the reverse lookup
    gene_filter |= rxrx3_genes

    all_results = {}

    # -----------------------------------------------------------------------
    # 1. OpenPhenom baseline (our reproduction)
    # -----------------------------------------------------------------------
    p("\n" + "-" * 60)
    p("1. OpenPhenom-S/16 baseline (TVN-normalized, our reproduction)")
    p("-" * 60)
    op_genes, op_embs = load_openphenom_baseline(meta)
    p(f"   Genes after aggregation: {len(op_genes)}")
    r = run_benchmark(op_genes, op_embs, "OpenPhenom-S/16 (reproduced)")
    all_results["OpenPhenom-S/16 (repro)"] = r

    # -----------------------------------------------------------------------
    # 2. Bonbon individual embeddings (735-gene subset)
    # -----------------------------------------------------------------------
    for name, path in EMB_PATHS.items():
        p(f"\n" + "-" * 60)
        p(f"2. Bonbon {name} ({len(rxrx3_genes)}-gene subset)")
        p("-" * 60)
        genes, embs = load_bonbon_embeddings(path, gene_filter=rxrx3_genes, aliases=GENE_ALIASES)
        p(f"   Loaded {len(genes)} genes (dim={embs.shape[1]})")
        r = run_benchmark(genes, embs, f"Bonbon {name}")
        all_results[f"Bonbon {name} (735)"] = r

    # -----------------------------------------------------------------------
    # 3. Bonbon fusion: L2-norm + concatenate all three
    # -----------------------------------------------------------------------
    p(f"\n" + "-" * 60)
    p(f"3. Bonbon fusion (proj + tok + pa, L2-norm concat)")
    p("-" * 60)

    # Find common genes across all embedding types
    common = set(rxrx3_genes)
    for path in EMB_PATHS.values():
        available_raw = set(f.replace(".pt", "") for f in os.listdir(path) if f.endswith(".pt"))
        # Map back through aliases
        available = set()
        reverse = {v: k for k, v in GENE_ALIASES.items()}
        for name_raw in available_raw:
            available.add(reverse.get(name_raw, name_raw))
        common &= available
    common = sorted(common)
    p(f"   Common genes across all 3 embedding types: {len(common)}")

    parts = []
    for path in EMB_PATHS.values():
        genes, embs = load_bonbon_embeddings(path, gene_filter=set(common), aliases=GENE_ALIASES)
        # L2 normalize
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        parts.append(embs / norms)

    fused = np.hstack(parts)
    p(f"   Fused dim: {fused.shape[1]}")
    r = run_benchmark(common, fused, "Bonbon fusion (proj+tok+pa)")
    all_results["Bonbon fusion (735)"] = r

    # -----------------------------------------------------------------------
    # 4. Bonbon full proteome (20,337 genes)
    # -----------------------------------------------------------------------
    for name, path in EMB_PATHS.items():
        p(f"\n" + "-" * 60)
        p(f"4. Bonbon {name} (full proteome)")
        p("-" * 60)
        genes, embs = load_bonbon_embeddings(path)
        p(f"   Loaded {len(genes)} genes (dim={embs.shape[1]})")
        r = run_benchmark(genes, embs, f"Bonbon {name} (full)")
        all_results[f"Bonbon {name} (full)"] = r

    # -----------------------------------------------------------------------
    # 5. Full proteome fusion
    # -----------------------------------------------------------------------
    p(f"\n" + "-" * 60)
    p(f"5. Bonbon fusion (full proteome)")
    p("-" * 60)
    full_parts = []
    full_common = None
    for path in EMB_PATHS.values():
        available = set(f.replace(".pt", "") for f in os.listdir(path) if f.endswith(".pt"))
        if full_common is None:
            full_common = available
        else:
            full_common &= available
    full_common = sorted(full_common)
    p(f"   Common genes (full proteome): {len(full_common)}")

    for path in EMB_PATHS.values():
        genes, embs = load_bonbon_embeddings(path, gene_filter=set(full_common))
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        full_parts.append(embs / norms)

    fused_full = np.hstack(full_parts)
    p(f"   Fused dim: {fused_full.shape[1]}")
    r = run_benchmark(full_common, fused_full, "Bonbon fusion (full)")
    all_results["Bonbon fusion (full)"] = r

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    format_comparison_table(all_results)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "rxrx3_core_results.json"

    out = {"published_baselines": PUBLISHED_BASELINES}
    for key, df in all_results.items():
        out[key] = {}
        for _, row in df.iterrows():
            out[key][row["source"]] = {
                "recall_5_95": float(row["recall_0.05_0.95"]),
                "recall_10_90": float(row["recall_0.1_0.9"]),
            }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    p(f"\nTotal time: {time.time() - t0:.0f}s")
    p(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
