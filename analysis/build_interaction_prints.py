"""Build interaction-print matrices from Bonbon embeddings (Phase 4).

Mathematical optimization for compound-compound similarity:

CONCAT interaction-print for compound j: IP_j = [prot[1];mol[j], prot[2];mol[j], ..., prot[N];mol[j]]
CC dot product: IP_j · IP_k = N*mol[j]·mol[k] + Σ_i ||prot[i]||²
The protein term is CONSTANT across all (j,k) pairs, so concat CC cosine similarity
is a monotonic rescaling of molecule-molecule cosine similarity. We compute this
analytically instead of brute-force.

PRODUCT interaction-print: IP_j = [prot[1]*mol[j], prot[2]*mol[j], ..., prot[N]*mol[j]]
CC dot product: IP_j · IP_k = Σ_d W_d * mol[j,d] * mol[k,d] where W_d = Σ_i prot[i,d]²
This IS a protein-weighted molecule similarity (different from plain molecule sim).
Compute as: mol @ diag(W) @ mol.T
"""
import json
import glob
import os
import sys
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from norm_diagnostic import find_embedding_dirs


def load_ordered_embeddings(pt_dir: str, entity_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    available = {}
    for f in glob.glob(os.path.join(pt_dir, "*.pt")):
        name = os.path.basename(f).replace(".pt", "")
        t = torch.load(f, map_location="cpu", weights_only=True).float()
        if t.dim() > 1:
            t = t.mean(dim=0)
        available[name] = t.numpy()

    ordered = []
    found_ids = []
    for eid in entity_ids:
        if eid in available:
            ordered.append(available[eid])
            found_ids.append(eid)

    if len(found_ids) < len(entity_ids):
        missing = [e for e in entity_ids if e not in available]
        print(f"  WARNING: {len(missing)} entities not found: {missing[:3]}...")

    return np.array(ordered, dtype=np.float32), found_ids


def cosine_sim(A, B=None):
    if B is None:
        B = A
    An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (An @ Bn.T).astype(np.float32)


def main(checkpoint_dir: str, results_dir: str, ground_truth_dir: str):
    ip_dir = os.path.join(results_dir, "interaction_prints")
    os.makedirs(ip_dir, exist_ok=True)

    gt_cc = np.load(os.path.join(ground_truth_dir, "phenomics_compound_compound_sim.npz"))
    gt_cg = np.load(os.path.join(ground_truth_dir, "phenomics_compound_gene_sim.npz"))
    compound_ids = gt_cc["compound_ids"].tolist()
    gene_ids = gt_cg["gene_ids"].tolist()
    print(f"Ground truth: {len(compound_ids)} compounds, {len(gene_ids)} genes")

    norm_path = os.path.join(results_dir, "norm_diagnostics.json")
    with open(norm_path) as f:
        norm_reports = json.load(f)

    emb_dirs = find_embedding_dirs(checkpoint_dir)
    print(f"Embedding directories: {list(emb_dirs.keys())}")

    embeddings = {}
    for key, path in emb_dirs.items():
        ids = compound_ids if "molecule" in key else gene_ids
        matrix, found = load_ordered_embeddings(path, ids)
        print(f"  {key}: {matrix.shape} ({len(found)}/{len(ids)})")
        embeddings[key] = (matrix, found)

    # ===================================================================
    # 4c: Binary interaction-prints (protein×molecule score matrices)
    # ===================================================================
    print("\n=== Binary interaction-prints (4c) ===")

    if "molecule_projection" in embeddings and "protein_projection" in embeddings:
        mol_proj, mol_ids = embeddings["molecule_projection"]
        prot_proj, prot_ids = embeddings["protein_projection"]
        print(f"\nProjection: mol {mol_proj.shape}, prot {prot_proj.shape}")

        # (n_prot, n_mol) score matrix
        score_raw = prot_proj @ mol_proj.T
        score_cos = cosine_sim(prot_proj, mol_proj)

        np.savez_compressed(os.path.join(ip_dir, "binary_projection.npz"),
            score_raw=score_raw, score_cosine=score_cos,
            protein_ids=np.array(prot_ids), compound_ids=np.array(mol_ids))
        print(f"  Raw dot: mean={score_raw.mean():.4f}, std={score_raw.std():.4f}")
        print(f"  Cosine: mean={score_cos.mean():.4f}, std={score_cos.std():.4f}")

    if "molecule_codebook_tokens" in embeddings and "protein_codebook_tokens" in embeddings:
        mol_tok, mol_ids = embeddings["molecule_codebook_tokens"]
        prot_tok, prot_ids = embeddings["protein_codebook_tokens"]
        print(f"\nCodebook tokens: mol {mol_tok.shape}, prot {prot_tok.shape}")

        score_raw = prot_tok @ mol_tok.T
        score_cos = cosine_sim(prot_tok, mol_tok)

        np.savez_compressed(os.path.join(ip_dir, "binary_codebook_tokens.npz"),
            score_raw=score_raw, score_cosine=score_cos,
            protein_ids=np.array(prot_ids), compound_ids=np.array(mol_ids))
        print(f"  Raw dot: mean={score_raw.mean():.4f}, std={score_raw.std():.4f}")
        print(f"  Cosine: mean={score_cos.mean():.4f}, std={score_cos.std():.4f}")

    # ===================================================================
    # 4d/4e: Codebook interaction-prints (CC similarity)
    # ===================================================================
    for emb_type in ["pooled_attention", "tokens"]:
        mol_key = f"molecule_codebook_{emb_type}"
        prot_key = f"protein_codebook_{emb_type}"
        if mol_key not in embeddings or prot_key not in embeddings:
            continue

        mol_emb, mol_ids = embeddings[mol_key]
        prot_emb, prot_ids = embeddings[prot_key]
        n_prot = len(prot_ids)
        n_mol = len(mol_ids)
        d_mol = mol_emb.shape[1]
        d_prot = prot_emb.shape[1]

        print(f"\n=== Codebook {emb_type} interaction-prints ===")
        print(f"  Molecules: {n_mol}×{d_mol}, Proteins: {n_prot}×{d_prot}")

        # --- CONCAT (analytical) ---
        # cos(IP_j, IP_k) = (C + N*mol[j]·mol[k]) / sqrt((C + N*||mol[j]||²)(C + N*||mol[k]||²))
        C = np.sum(np.linalg.norm(prot_emb, axis=1) ** 2)
        N = n_prot
        mol_dot = mol_emb @ mol_emb.T
        mol_norms_sq = np.diag(mol_dot)

        numerator = C + N * mol_dot
        denom_j = np.sqrt(C + N * mol_norms_sq)
        denominator = np.outer(denom_j, denom_j)
        cc_sim_concat = (numerator / (denominator + 1e-8)).astype(np.float32)

        np.savez_compressed(os.path.join(ip_dir, f"{emb_type}_concat.npz"),
            similarity=cc_sim_concat, compound_ids=np.array(mol_ids))
        ut = cc_sim_concat[np.triu_indices(n_mol, k=1)]
        print(f"  Concat CC sim: mean={ut.mean():.4f}, std={ut.std():.4f}")

        # --- PRODUCT (protein-weighted molecule similarity) ---
        if d_prot == d_mol:
            # W_d = Σ_i prot[i,d]²  — per-dimension weight from all proteins
            W = np.sum(prot_emb ** 2, axis=0)  # (d,)

            # Weighted molecule embeddings
            mol_weighted = mol_emb * np.sqrt(W)[np.newaxis, :]  # (n_mol, d)
            cc_sim_product = cosine_sim(mol_weighted)

            np.savez_compressed(os.path.join(ip_dir, f"{emb_type}_product.npz"),
                similarity=cc_sim_product, compound_ids=np.array(mol_ids))
            ut = cc_sim_product[np.triu_indices(n_mol, k=1)]
            print(f"  Product CC sim: mean={ut.mean():.4f}, std={ut.std():.4f}")
        else:
            print(f"  PRODUCT skipped: dimension mismatch (prot={d_prot}, mol={d_mol})")

    # ===================================================================
    # Compound-gene score matrices for per-target evaluation
    # ===================================================================
    print("\n=== Compound-gene score matrices ===")

    for mol_key, prot_key, label in [
        ("molecule_projection", "protein_projection", "cg_projection"),
        ("molecule_codebook_tokens", "protein_codebook_tokens", "cg_codebook_tokens"),
        ("molecule_codebook_pooled_attention", "protein_codebook_pooled_attention", "cg_pooled_attention"),
    ]:
        if mol_key not in embeddings or prot_key not in embeddings:
            continue

        mol, mol_ids = embeddings[mol_key]
        prot, prot_ids = embeddings[prot_key]

        cg_raw = mol @ prot.T
        cg_cos = cosine_sim(mol, prot)

        np.savez_compressed(os.path.join(ip_dir, f"{label}.npz"),
            score_raw=cg_raw, score_cosine=cg_cos,
            compound_ids=np.array(mol_ids), protein_ids=np.array(prot_ids))
        print(f"  {label}: {cg_cos.shape}")

    print("\nDone building interaction-prints!")


if __name__ == "__main__":
    checkpoint_dir = sys.argv[1]
    results_dir = sys.argv[2]
    gt_dir = sys.argv[3] if len(sys.argv) > 3 else "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
    main(checkpoint_dir, results_dir, gt_dir)
