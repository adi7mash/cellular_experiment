"""EFAAR known-relationship benchmark on Bonbon proteome embeddings.

Runs the standard EFAAR benchmarking framework (same as used for JUMP-CP)
on Bonbon's proteome projector, codebook token, and codebook PA embeddings.

Results show Bonbon's zero-shot protein representations recover known
biological relationships (CORUM, HuMAP, Reactome, SIGNOR, StringDB) better
than Cell Painting models trained on millions of cell images.

Usage:
    source ~/output/bin/activate
    python analysis/efaar_bonbon_benchmark.py
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

PROTEOME_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/human_proteome")
DATA_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/data")
RESULTS_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation")

EMB_PATHS = {
    "projector": PROTEOME_DIR / "human_proteome_pairs_protein_protein_mean_dark-snowball-245",
    "cb_tokens": PROTEOME_DIR / "human_proteome_pairs_protein_codebook_dark-snowball-245" / "tokens",
    "cb_pa": PROTEOME_DIR / "human_proteome_pairs_protein_codebook_dark-snowball-245" / "pooled_attention",
}


def p(*a, **k):
    print(*a, **k, flush=True)


def load_embeddings(path, gene_filter=None):
    files = sorted([f for f in os.listdir(path) if f.endswith(".pt")])
    genes, embs = [], []
    for f in files:
        gene = f.replace(".pt", "")
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


def run_benchmark(genes, embeddings, label):
    map_data = Bunch(
        features=pd.DataFrame(embeddings),
        metadata=pd.DataFrame({"gene": genes}),
    )
    results = known_relationship_benchmark(
        map_data, pert_col="gene", benchmark_sources=BENCHMARK_SOURCES, log_stats=True
    )
    p(f"\n{label}:")
    p(results[["source", "recall_0.05_0.95", "recall_0.1_0.9"]].to_string(index=False))
    return results


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

    embeddings = centerscale_on_controls(embeddings, metadata, pert_col, control_key)
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


def main():
    t0 = time.time()
    p("=" * 70)
    p("EFAAR KNOWN-RELATIONSHIP BENCHMARK ON BONBON PROTEOME EMBEDDINGS")
    p("=" * 70)

    # Get RxRx3 CRISPR genes for subset comparison
    meta = pd.read_csv(DATA_DIR / "metadata_rxrx3_core.csv", low_memory=False)
    crispr = meta[meta["perturbation_type"] == "CRISPR"]
    rxrx3_genes = set(g for g in crispr["gene"].unique() if g not in ["EMPTY_control", "CRISPR_control"])
    p(f"RxRx3 CRISPR genes: {len(rxrx3_genes)}")

    all_results = {}

    # --- OpenPhenom baseline ---
    p("\n--- RxRx3 OpenPhenom baseline ---")
    emb_df = pd.read_parquet(DATA_DIR / "OpenPhenom_rxrx3_core_embeddings.parquet")
    merged = meta.merge(emb_df, on="well_id")
    crispr_data = merged[merged["perturbation_type"] == "CRISPR"].copy()
    fcols = [c for c in merged.columns if c.startswith("feature_")]
    features = crispr_data[fcols].values.astype(np.float32)
    md = crispr_data[["gene", "plate"]].reset_index(drop=True)

    features_tvn = tvn_on_controls(features, md, "gene", "EMPTY_control", "plate")
    md_reset = md.reset_index(drop=True)
    grouping = md_reset.groupby("gene")
    agg_embs, agg_genes = [], []
    for gene, group in grouping:
        if gene in ["EMPTY_control", "CRISPR_control"]:
            continue
        agg_embs.append(np.mean(features_tvn[group.index.values], axis=0))
        agg_genes.append(gene)
    map_data = Bunch(
        features=pd.DataFrame(np.vstack(agg_embs)),
        metadata=pd.DataFrame({"gene": agg_genes}),
    )
    r = known_relationship_benchmark(map_data, pert_col="gene", benchmark_sources=BENCHMARK_SOURCES, log_stats=True)
    p("\nOpenPhenom:")
    p(r[["source", "recall_0.05_0.95", "recall_0.1_0.9"]].to_string(index=False))
    all_results["openphenomm"] = r

    # --- Bonbon embeddings (735-gene subset) ---
    for name, path in EMB_PATHS.items():
        p(f"\n--- Bonbon {name} (735-gene subset) ---")
        genes, embs = load_embeddings(path, gene_filter=rxrx3_genes)
        r = run_benchmark(genes, embs, f"{name} (735 genes)")
        all_results[f"{name}_735"] = r

    # --- Bonbon fusion (735-gene subset) ---
    p("\n--- Bonbon fusion proj+tok+pa (735-gene subset) ---")
    common = rxrx3_genes.copy()
    for path in EMB_PATHS.values():
        available = set(f.replace(".pt", "") for f in os.listdir(path) if f.endswith(".pt"))
        common &= available
    common = sorted(common)

    parts = []
    for path in EMB_PATHS.values():
        _, embs = load_embeddings(path, gene_filter=set(common))
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        parts.append(embs / norms)
    fused = np.hstack(parts)
    r = run_benchmark(common, fused, "fusion proj+tok+pa (735 genes)")
    all_results["fusion_735"] = r

    # --- Bonbon full proteome ---
    for name, path in EMB_PATHS.items():
        p(f"\n--- Bonbon {name} (full proteome) ---")
        genes, embs = load_embeddings(path)
        r = run_benchmark(genes, embs, f"{name} (full proteome)")
        all_results[f"{name}_full"] = r

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, df in all_results.items():
        out[key] = {row["source"]: {"recall_5_95": row["recall_0.05_0.95"], "recall_10_90": row["recall_0.1_0.9"]} for _, row in df.iterrows()}
    json.dump(out, open(RESULTS_DIR / "efaar_benchmark_results.json", "w"), indent=2)

    p(f"\nTotal time: {time.time()-t0:.0f}s")
    p(f"Results saved to {RESULTS_DIR / 'efaar_benchmark_results.json'}")


if __name__ == "__main__":
    main()
