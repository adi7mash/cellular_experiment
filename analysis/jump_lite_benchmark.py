#!/usr/bin/env python3
"""
JUMP-lite CG→CRISPR cross-modality retrieval benchmark.

Evaluates Bonbon's frozen interaction embeddings on the JUMP-lite benchmark
(Muñoz et al. 2026, arxiv 2608.07632). Uses pre-computed cosine similarity
between compound and protein embeddings as the retrieval score, matching the
JUMP-lite per-query recall metric exactly.

Usage:
    python jump_lite_benchmark.py [--checkpoint dark-snowball-245]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl


JUMP_LITE_METADATA = Path("/tmp/JUMP_lite/metadata")
CRISPR_META = Path("/opt/dlami/nvme/jump_cp/data/crispr.csv.gz")


def normalize_name(s: str) -> str:
    s = s.lower().strip()
    for suffix in [
        " hcl", " hydrochloride", " sodium", " potassium", " mesylate",
        " maleate", " fumarate", " tosylate", " citrate", " tartrate",
        " sulfate", " phosphate", " acetate", " bromide", " chloride",
        " dihydrochloride", " monohydrochloride", " calcium",
    ]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    s = re.sub(r"\s*\(.*?\)\s*", "", s)
    return s.strip(" .,;-")


def build_compound_mapping(our_compounds, annotations):
    ann_cpds = annotations.select(["Metadata_JCP2022", "name"]).unique()
    jump_name_to_jcp, jump_norm_to_jcp = {}, {}
    for row in ann_cpds.iter_rows():
        jcp, name = row
        jump_name_to_jcp[name] = jcp
        jump_name_to_jcp[name.lower()] = jcp
        jump_norm_to_jcp[normalize_name(name)] = jcp

    mapping = {}
    for cpd in our_compounds:
        if cpd in jump_name_to_jcp:
            mapping[cpd] = jump_name_to_jcp[cpd]
        elif cpd.lower() in jump_name_to_jcp:
            mapping[cpd] = jump_name_to_jcp[cpd.lower()]
        elif normalize_name(cpd) in jump_norm_to_jcp:
            mapping[cpd] = jump_norm_to_jcp[normalize_name(cpd)]
    return mapping


def calculate_recall_at_k(sim, positive_mask, k_percentages=(1.0, 5.0, 10.0)):
    """JUMP-lite exact recall: per-query hits/positives, averaged."""
    n_queries, n_references = sim.shape
    n_pos = positive_mask.sum(axis=1)
    valid = n_pos > 0
    results = {}
    for k_pct in k_percentages:
        k = max(1, int(n_references * k_pct / 100))
        top_k = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        hits = positive_mask[rows, top_k].sum(axis=1)
        recall_per_query = hits[valid] / n_pos[valid]
        results[f"recall@{k_pct:.0f}%"] = float(recall_per_query.mean()) if recall_per_query.size else 0.0
    return results


def run_benchmark(checkpoint: str = "dark-snowball-245"):
    base = Path(f"/opt/dlami/nvme/rxrx3_phenomics/results/{checkpoint}")
    proteome = base / "human_proteome"

    ann = pl.read_parquet(JUMP_LITE_METADATA / "jump_lite_v1_refchem_annotations.parquet")
    jump_meta = pl.read_parquet(JUMP_LITE_METADATA / "jump_lite_v1_perturbation_metadata.parquet")

    ct_data = np.load(proteome / "cg_codebook_tokens.npz", allow_pickle=True)
    pa_data = np.load(proteome / "cg_pooled_attention.npz", allow_pickle=True)

    our_compounds = ct_data["compound_ids"].tolist()
    our_proteins = ct_data["protein_ids"].tolist()
    ct_scores = ct_data["score_cosine"]
    pa_scores = pa_data["score_cosine"]

    compound_to_idx = {c: i for i, c in enumerate(our_compounds)}
    protein_to_idx = {p: i for i, p in enumerate(our_proteins)}

    cpd_name_to_jcp = build_compound_mapping(our_compounds, ann)
    jcp_to_cpd_name = {v: k for k, v in cpd_name_to_jcp.items()}

    crispr_trt = jump_meta.filter(
        (pl.col("Metadata_Perturbation_Type") == "crispr")
        & (pl.col("Metadata_pert_type") == "trt")
    )
    ref_genes = sorted(
        g for g in crispr_trt["Metadata_Symbol"].unique().to_list() if g in protein_to_idx
    )
    ref_gene_indices = np.array([protein_to_idx[g] for g in ref_genes])
    n_ref = len(ref_genes)
    ref_gene_to_ri = {g: i for i, g in enumerate(ref_genes)}

    cg_ann = ann.filter(
        (pl.col("modality") == "crispr") & pl.col("CrossModalityTier").is_not_null()
    )

    def eval_tier(tiers, scores_matrix, label):
        filtered = cg_ann.filter(pl.col("CrossModalityTier").is_in(tiers))
        cpd_targets = defaultdict(set)
        for row in filtered.iter_rows(named=True):
            cpd_jcp = row["Metadata_JCP2022"]
            gene = row["target"]
            if cpd_jcp in jcp_to_cpd_name and gene in ref_gene_to_ri:
                cpd_targets[cpd_jcp].add(gene)

        eval_cpds = sorted(cpd_targets.keys())
        n_q = len(eval_cpds)
        sim = np.zeros((n_q, n_ref), dtype=np.float32)
        mask = np.zeros((n_q, n_ref), dtype=bool)

        for qi, jcp in enumerate(eval_cpds):
            cpd_idx = compound_to_idx[jcp_to_cpd_name[jcp]]
            sim[qi] = scores_matrix[cpd_idx, ref_gene_indices]
            for t in cpd_targets[jcp]:
                mask[qi, ref_gene_to_ri[t]] = True

        recall = calculate_recall_at_k(sim, mask)
        n_valid = int((mask.sum(axis=1) > 0).sum())
        n_pairs = int(mask.sum())
        print(f"  {label}: {n_valid} queries, {n_pairs} pairs — " +
              ", ".join(f"{k}: {v*100:.2f}%" for k, v in recall.items()))
        return {"n_queries": n_valid, "n_pairs": n_pairs, **recall}

    print(f"JUMP-lite CG→CRISPR | checkpoint: {checkpoint} | ref genes: {n_ref}")
    print(f"Mapped compounds: {len(cpd_name_to_jcp)}/{len(our_compounds)}")

    results = {}
    tier_sets = {
        "all_tiers": ["Tier1", "Tier2", "Tier3"],
        "Tier2": ["Tier2"],
        "Tier3": ["Tier3"],
    }

    for tier_name, tiers in tier_sets.items():
        print(f"\n--- {tier_name} ---")
        results[tier_name] = {
            "codebook_tokens": eval_tier(tiers, ct_scores, "Codebook 1280d"),
            "pooled_attention": eval_tier(tiers, pa_scores, "Attention 8192d"),
            "ensemble": eval_tier(tiers, (ct_scores + pa_scores) / 2, "Ensemble"),
        }

    out = {
        "benchmark": "JUMP-lite CG→CRISPR",
        "reference": "Muñoz et al. 2026 (arxiv 2608.07632)",
        "checkpoint": checkpoint,
        "metric": "per-query recall (hits_in_top_k / n_positives), averaged",
        "reference_genes": n_ref,
        "mapped_compounds": len(cpd_name_to_jcp),
        "random_baseline": {
            f"recall@{k}%": max(1, int(n_ref * k / 100)) / n_ref for k in [1, 5, 10]
        },
        "results": results,
    }

    out_path = base / "evaluation" / "jump_lite_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="dark-snowball-245")
    args = parser.parse_args()
    run_benchmark(args.checkpoint)
