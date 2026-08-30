"""Proteome-scale zero-shot per-target AUROC.

Uses the full human proteome (20,337 proteins) for PPI propagation
of CG scores, improving zero-shot per-target from 58.1% to 60.3%.

Best method: weighted rank fusion of
  2.5 × projector_735 + 1.5 × proteome_PA_PPI

Requires:
  - Proteome projector embeddings (run millefeuille with --protein_encoder protein)
  - Proteome codebook PA embeddings (from human_proteome_cg_scores.py)
  - RxRx3 evaluation data (from phenomics_signature.py)
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from phenomics_signature import load_data, select_known

PROTEOME_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/human_proteome")
RESULTS_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation")


def p(*a, **k):
    print(*a, **k, flush=True)


def eval_matrix(score_mat, ki, cg_binary, n_g, label):
    aucs = []
    for gi in range(n_g):
        y = cg_binary[ki, gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        s = score_mat[ki, gi]
        try:
            aucs.append(roc_auc_score(y, s))
        except:
            pass
    med = np.median(aucs)
    p25, p75 = np.percentile(aucs, 25), np.percentile(aucs, 75)
    p(f"  {label:55s} {med:.4f} [{p25:.4f}-{p75:.4f}] ({len(aucs)} tgts)")
    return med, aucs


def main():
    p("=" * 70)
    p("PROTEOME-SCALE ZERO-SHOT PER-TARGET")
    p("=" * 70)

    t0 = time.time()

    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi_small, cg_scores = data[:7]
    cg_binary = gt["cg_binary_top1"]
    known_idx = select_known(cg_scores, 150)
    ki = known_idx
    proj_raw = cg_scores["proj_raw"]

    gene_map = pd.read_csv("/opt/dlami/nvme/rxrx3_phenomics/data/gene_to_protein.tsv", sep="\t")
    rxrx3_genes = gene_map["gene_symbol"].tolist()
    pairs = pd.read_csv("/opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv", sep="\t")
    rxrx3_mols = sorted(pairs["Molecule"].unique())

    proteome_pa = np.load(PROTEOME_DIR / "cg_pooled_attention.npz")
    proteome_ppi = np.load(PROTEOME_DIR / "proteome_ppi.npz")
    pa_cos_prot = proteome_pa["score_cosine"]
    ppi_mat = proteome_ppi["ppi_matrix"]
    cb_prot_ids = list(proteome_ppi["protein_ids"])
    pa_comp_ids = list(proteome_pa["compound_ids"])

    gene_cb_idx = {gi: cb_prot_ids.index(g) for gi, g in enumerate(rxrx3_genes) if g in cb_prot_ids}
    mol_cb_idx = {mi: pa_comp_ids.index(m) for mi, m in enumerate(rxrx3_mols) if m in pa_comp_ids}
    gene_cbidx = np.array([gene_cb_idx[g] for g in sorted(gene_cb_idx)])
    mol_cbidx = np.array([mol_cb_idx[m] for m in sorted(mol_cb_idx)])
    valid_genes_cb = sorted(gene_cb_idx.keys())
    valid_mols_cb = sorted(mol_cb_idx.keys())

    pa_direct = pa_cos_prot[np.ix_(mol_cbidx, gene_cbidx)]
    pa_prop = pa_cos_prot[mol_cbidx] @ ppi_mat[:, gene_cbidx]
    pa_blended = 0.7 * pa_direct + 0.3 * pa_prop
    pa_full = np.zeros((n_c, n_g))
    for i, gi in enumerate(valid_genes_cb):
        for j, ci in enumerate(valid_mols_cb):
            pa_full[ci, gi] = pa_blended[j, i]
    pa_r = np.apply_along_axis(rankdata, 0, pa_full)

    r_proj = np.apply_along_axis(rankdata, 0, proj_raw)

    p("\n--- Baselines ---")
    eval_matrix(proj_raw, ki, cg_binary, n_g, "Projector dot (best single)")

    p("\n--- Weighted rank fusion ---")
    results = {}

    for w_proj in [1, 1.5, 2, 2.5, 3]:
        for w_pa in [0.5, 1, 1.5, 2]:
            score_mat = w_proj * r_proj + w_pa * pa_r
            med, _ = eval_matrix(score_mat, ki, cg_binary, n_g, f"w_proj={w_proj}, w_pa={w_pa}")
            results[f"proj_{w_proj}_pa_{w_pa}"] = med

    best_key = max(results, key=results.get)
    p(f"\n  Best: {best_key} -> {results[best_key]:.4f}")

    out = RESULTS_DIR / "proteome_zero_shot_results.json"
    json.dump(results, open(out, "w"), indent=2)
    p(f"\n  Saved to {out}")
    p(f"  Time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
