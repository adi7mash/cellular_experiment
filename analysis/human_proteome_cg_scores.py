"""Compute CG interaction scores for the human proteome.

Uses the existing molecule codebook embeddings (1,674 compounds) and the
newly computed human proteome protein codebook embeddings (20K proteins)
to build the full compound × protein interaction score matrix.
"""

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROT_EMB_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/human_proteome/"
                     "human_proteome_pairs_protein_codebook_dark-snowball-245")
MOL_CB_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245/"
                   "rxrx3_pairs_molecule_codebook_dark-snowball-245")
PROTEOME_TSV = "/opt/dlami/nvme/rxrx3_phenomics/data/human_proteome.tsv"
PAIRS_TSV = "/opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv"
OUTPUT_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/human_proteome")


def p(*a, **k):
    print(*a, **k, flush=True)


def load_embeddings(emb_dir, entity_ids, emb_type="tokens"):
    """Load and mean-pool embeddings for a list of entity IDs."""
    pt_dir = emb_dir / emb_type
    loaded = {}
    for f in pt_dir.glob("*.pt"):
        name = f.stem
        t = torch.load(f, map_location="cpu", weights_only=True).float()
        if t.dim() > 1:
            t = t.mean(dim=0)
        loaded[name] = t.numpy()

    ordered = []
    found_ids = []
    for eid in entity_ids:
        if eid in loaded:
            ordered.append(loaded[eid])
            found_ids.append(eid)

    return np.array(ordered, dtype=np.float32), found_ids


def cosine_sim(A, B):
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (An @ Bn.T).astype(np.float32)


def main():
    p("=" * 70)
    p("HUMAN PROTEOME CG INTERACTION SCORES")
    p("=" * 70)

    t0 = time.time()

    # Load proteome gene list
    import pandas as pd
    proteome = pd.read_csv(PROTEOME_TSV, sep="\t")
    prot_ids = proteome["gene_symbol"].fillna(proteome["uniprot_id"]).tolist()
    # Deduplicate while preserving order
    seen = set()
    unique_prot_ids = []
    for pid in prot_ids:
        if pid not in seen:
            seen.add(pid)
            unique_prot_ids.append(pid)
    prot_ids = unique_prot_ids
    p(f"  Proteome: {len(prot_ids)} unique proteins")

    # Load molecule list from original pairs
    pairs = pd.read_csv(PAIRS_TSV, sep="\t")
    mol_ids = pairs["Molecule"].unique().tolist()
    p(f"  Molecules: {len(mol_ids)} compounds")

    # Load protein codebook embeddings (tokens + pooled_attention)
    p("\nLoading protein embeddings...")
    prot_tok, prot_tok_ids = load_embeddings(PROT_EMB_DIR, prot_ids, "tokens")
    prot_pa, prot_pa_ids = load_embeddings(PROT_EMB_DIR, prot_ids, "pooled_attention")
    p(f"  Protein tokens: {prot_tok.shape} ({len(prot_tok_ids)} matched)")
    p(f"  Protein pooled_attn: {prot_pa.shape} ({len(prot_pa_ids)} matched)")

    # Load molecule codebook embeddings
    p("\nLoading molecule embeddings...")
    mol_tok, mol_tok_ids = load_embeddings(MOL_CB_DIR, mol_ids, "tokens")
    mol_pa, mol_pa_ids = load_embeddings(MOL_CB_DIR, mol_ids, "pooled_attention")
    p(f"  Molecule tokens: {mol_tok.shape} ({len(mol_tok_ids)} matched)")
    p(f"  Molecule pooled_attn: {mol_pa.shape} ({len(mol_pa_ids)} matched)")

    # Compute CG scores
    p("\nComputing CG interaction scores...")

    # Codebook tokens: mol × prot
    cg_tok_raw = mol_tok @ prot_tok.T  # [n_mol, n_prot]
    cg_tok_cos = cosine_sim(mol_tok, prot_tok)
    p(f"  CG codebook tokens: {cg_tok_raw.shape}")
    p(f"    Raw: mean={cg_tok_raw.mean():.4f}, std={cg_tok_raw.std():.4f}")
    p(f"    Cos: mean={cg_tok_cos.mean():.4f}, std={cg_tok_cos.std():.4f}")

    np.savez_compressed(OUTPUT_DIR / "cg_codebook_tokens.npz",
        score_raw=cg_tok_raw, score_cosine=cg_tok_cos,
        compound_ids=np.array(mol_tok_ids), protein_ids=np.array(prot_tok_ids))

    # Pooled attention: mol × prot
    cg_pa_raw = mol_pa @ prot_pa.T
    cg_pa_cos = cosine_sim(mol_pa, prot_pa)
    p(f"  CG pooled attention: {cg_pa_raw.shape}")
    p(f"    Raw: mean={cg_pa_raw.mean():.4f}, std={cg_pa_raw.std():.4f}")
    p(f"    Cos: mean={cg_pa_cos.mean():.4f}, std={cg_pa_cos.std():.4f}")

    np.savez_compressed(OUTPUT_DIR / "cg_pooled_attention.npz",
        score_raw=cg_pa_raw, score_cosine=cg_pa_cos,
        compound_ids=np.array(mol_pa_ids), protein_ids=np.array(prot_pa_ids))

    # Also compute PPI matrix from protein codebook cosine similarity
    p("\nComputing PPI matrix (protein-protein cosine similarity)...")
    ppi = cosine_sim(prot_tok, prot_tok)
    p(f"  PPI matrix: {ppi.shape}")
    p(f"    Mean: {ppi[np.triu_indices(len(ppi), k=1)].mean():.4f}")

    np.savez_compressed(OUTPUT_DIR / "proteome_ppi.npz",
        ppi_matrix=ppi, protein_ids=np.array(prot_tok_ids))

    # Stats
    elapsed = time.time() - t0
    p(f"\n{'='*70}")
    p("SUMMARY")
    p(f"{'='*70}")
    p(f"  Proteins embedded: {len(prot_tok_ids)}")
    p(f"  Molecules: {len(mol_tok_ids)}")
    p(f"  CG matrix shape: {cg_tok_raw.shape}")
    p(f"  Total pairs: {cg_tok_raw.shape[0] * cg_tok_raw.shape[1]:,}")
    p(f"  PPI matrix: {ppi.shape}")
    p(f"  Time: {elapsed:.0f}s")

    # Check overlap with RxRx3 genes
    rxrx3_genes = pd.read_csv("/opt/dlami/nvme/rxrx3_phenomics/data/gene_to_protein.tsv",
                               sep="\t")["gene_symbol"].tolist()
    overlap = set(prot_tok_ids) & set(rxrx3_genes)
    p(f"  RxRx3 gene overlap: {len(overlap)}/{len(rxrx3_genes)}")

    p(f"\n  Output: {OUTPUT_DIR}")
    p(f"  Files:")
    for f in sorted(OUTPUT_DIR.glob("*.npz")):
        size_mb = f.stat().st_size / 1e6
        p(f"    {f.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
