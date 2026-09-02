"""
LINCS L1000 Evaluation — Bonbon CG scores predict compound-gene connectivity.
"""

import numpy as np
import pandas as pd
import torch
import h5py
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score
import os

BASE = "/opt/dlami/nvme/lincs_l1000"

# ── Load metadata ──
print("Loading metadata...")
sig_info = pd.read_csv(f"{BASE}/sig_info.txt.gz", sep="\t", low_memory=False)
gene_info = pd.read_csv(f"{BASE}/gene_info.txt.gz", sep="\t", low_memory=False)
compound_df = pd.read_csv(f"{BASE}/compound_to_safe.tsv", sep="\t")
landmark_genes = pd.read_csv(f"{BASE}/landmark_genes.tsv", sep="\t")

lm_gene_ids = gene_info[gene_info['pr_is_lm'] == 1]['pr_gene_id'].tolist()
lm_gene_symbols = gene_info[gene_info['pr_is_lm'] == 1]['pr_gene_symbol'].tolist()

print(f"Landmark genes: {len(lm_gene_ids)}")
print(f"Compounds: {len(compound_df)}")

# ── Load Level 5 signatures ──
print("\nLoading Level 5 GCTX...")
gctx_path = f"{BASE}/level5.gctx"

with h5py.File(gctx_path, 'r') as f:
    row_ids = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f['0']['META']['ROW']['id'][:]]
    col_ids = [x.decode() if isinstance(x, bytes) else str(x)
               for x in f['0']['META']['COL']['id'][:]]

    # matrix[sig_idx, gene_idx], shape (473647, 12328)
    row_id_to_idx = {rid: i for i, rid in enumerate(row_ids)}
    col_id_to_idx = {cid: i for i, cid in enumerate(col_ids)}
    lm_indices = np.array([row_id_to_idx[str(gid)] for gid in lm_gene_ids
                           if str(gid) in row_id_to_idx])
    lm_sorted_order = np.argsort(lm_indices)
    lm_sorted_indices = lm_indices[lm_sorted_order]
    print(f"Landmark gene indices found: {len(lm_indices)}")

# ── Collect all signature indices we need ──
print("\nCollecting signature indices...")
ts_compounds = set(compound_df['compound_id'].tolist())
mapped_lm_symbols = set(landmark_genes['gene'].tolist())

cp_sigs = sig_info[(sig_info['pert_type'] == 'trt_cp') &
                   (sig_info['pert_iname'].isin(ts_compounds))]
sh_sigs = sig_info[(sig_info['pert_type'].isin(['trt_sh', 'trt_sh.cgs'])) &
                   (sig_info['pert_iname'].isin(mapped_lm_symbols))]

compound_sig_groups = cp_sigs.groupby('pert_iname')['sig_id'].apply(list).to_dict()
gene_sig_groups = sh_sigs.groupby('pert_iname')['sig_id'].apply(list).to_dict()

# Collect all unique sig indices we need to read (cap at 50 per entity)
all_sig_indices = set()
compound_sig_idx = {}
for cname, sids in compound_sig_groups.items():
    indices = [col_id_to_idx[sid] for sid in sids if sid in col_id_to_idx][:50]
    if indices:
        compound_sig_idx[cname] = indices
        all_sig_indices.update(indices)

gene_sig_idx = {}
for gname, sids in gene_sig_groups.items():
    indices = [col_id_to_idx[sid] for sid in sids if sid in col_id_to_idx][:50]
    if indices:
        gene_sig_idx[gname] = indices
        all_sig_indices.update(indices)

all_sig_list = sorted(all_sig_indices)
sig_to_local = {s: i for i, s in enumerate(all_sig_list)}
print(f"Total unique signatures to read: {len(all_sig_list)}")
print(f"Compounds with signatures: {len(compound_sig_idx)}")
print(f"Genes with shRNA signatures: {len(gene_sig_idx)}")

# ── Batch read all needed signatures for landmark genes ──
print(f"\nBatch reading {len(all_sig_list)} signatures × {len(lm_indices)} landmark genes...")

with h5py.File(gctx_path, 'r') as f:
    matrix = f['0']['DATA']['0']['matrix']
    # Read in chunks to manage memory
    chunk_size = 5000
    sig_data = np.zeros((len(all_sig_list), len(lm_indices)), dtype=np.float32)

    for start in range(0, len(all_sig_list), chunk_size):
        end = min(start + chunk_size, len(all_sig_list))
        chunk_indices = all_sig_list[start:end]
        # h5py fancy indexing: read selected rows, selected columns
        # Must sort indices for h5py efficiency
        chunk_sorted = np.sort(chunk_indices)
        chunk_data = matrix[chunk_sorted][:, lm_sorted_indices]
        # Map back to our ordering
        for i, ci in enumerate(chunk_indices):
            local_i = sig_to_local[ci]
            # Find position of ci in chunk_sorted
            pos = np.searchsorted(chunk_sorted, ci)
            sig_data[local_i] = chunk_data[pos][lm_sorted_order]

        if (start // chunk_size) % 5 == 0:
            print(f"  Read {end}/{len(all_sig_list)} signatures...")

print(f"Loaded signature data: {sig_data.shape}")

# ── Build consensus signatures ──
print("\nBuilding consensus compound signatures...")
compound_consensus = {}
for cname, indices in compound_sig_idx.items():
    local_indices = [sig_to_local[i] for i in indices]
    compound_consensus[cname] = sig_data[local_indices].mean(axis=0)
print(f"Consensus compound signatures: {len(compound_consensus)}")

print("Building consensus gene knockdown signatures...")
gene_consensus = {}
for gname, indices in gene_sig_idx.items():
    local_indices = [sig_to_local[i] for i in indices]
    gene_consensus[gname] = sig_data[local_indices].mean(axis=0)
print(f"Consensus gene knockdown signatures: {len(gene_consensus)}")

del sig_data  # free memory

# ── Compute L1000 connectivity matrix ──
print("\nComputing L1000 connectivity...")

eval_compounds = sorted(compound_consensus.keys())
eval_genes = sorted(set(gene_consensus.keys()) & mapped_lm_symbols)
print(f"Evaluation: {len(eval_compounds)} compounds × {len(eval_genes)} genes")

C_mat = np.array([compound_consensus[c] for c in eval_compounds])
G_mat = np.array([gene_consensus[g] for g in eval_genes])

C_norm = C_mat / (np.linalg.norm(C_mat, axis=1, keepdims=True) + 1e-10)
G_norm = G_mat / (np.linalg.norm(G_mat, axis=1, keepdims=True) + 1e-10)

connectivity = np.abs(C_norm @ G_norm.T)
signed_conn = C_norm @ G_norm.T
print(f"Connectivity stats: mean={connectivity.mean():.4f}, std={connectivity.std():.4f}")

# ── Load Bonbon CG scores ──
print("\nLoading Bonbon CG scores...")

mol_dir = f"{BASE}/lincs_pairs_molecule_codebook_dark-snowball-245"
prot_dir = f"{BASE}/lincs_pairs_protein_codebook_dark-snowball-245"

mol_tokens = {}
prot_tokens = {}

for c in eval_compounds:
    for loc in [f"{mol_dir}/tokens/{c}.pt", f"{mol_dir}/{c}.pt"]:
        if os.path.exists(loc):
            mol_tokens[c] = torch.load(loc, map_location='cpu', weights_only=True)
            break

for g in eval_genes:
    for loc in [f"{prot_dir}/tokens/{g}.pt", f"{prot_dir}/{g}.pt"]:
        if os.path.exists(loc):
            prot_tokens[g] = torch.load(loc, map_location='cpu', weights_only=True)
            break

print(f"Loaded mol tokens: {len(mol_tokens)}, prot tokens: {len(prot_tokens)}")

shared_compounds = sorted(set(eval_compounds) & set(mol_tokens.keys()))
shared_genes = sorted(set(eval_genes) & set(prot_tokens.keys()))
print(f"Shared: {len(shared_compounds)} compounds × {len(shared_genes)} genes")

# Build Bonbon CG score matrix
cg_scores = np.zeros((len(shared_compounds), len(shared_genes)))
for ci, c in enumerate(shared_compounds):
    for gi, g in enumerate(shared_genes):
        cg_scores[ci, gi] = (mol_tokens[c] @ prot_tokens[g]).item()

c_idx = [eval_compounds.index(c) for c in shared_compounds]
g_idx = [eval_genes.index(g) for g in shared_genes]
conn_sub = connectivity[np.ix_(c_idx, g_idx)]
signed_sub = signed_conn[np.ix_(c_idx, g_idx)]

# ── Evaluate ──
print(f"\n{'='*70}")
print(f"LINCS L1000 EVALUATION — Bonbon CG scores vs transcriptomic connectivity")
print(f"{'='*70}")
print(f"Matrix: {len(shared_compounds)} compounds × {len(shared_genes)} genes")

flat_cg = cg_scores.ravel()
flat_conn = conn_sub.ravel()

# 1. Global correlation
sp_r, sp_p = spearmanr(flat_cg, flat_conn)
pe_r, pe_p = pearsonr(flat_cg, flat_conn)
print(f"\n1. Global correlation (CG score vs |L1000 cosine|):")
print(f"   Spearman r = {sp_r:.4f} (p = {sp_p:.2e})")
print(f"   Pearson r  = {pe_r:.4f} (p = {pe_p:.2e})")

# 1b. Signed connectivity
sp_signed, _ = spearmanr(flat_cg, -signed_sub.ravel())
print(f"\n   CG vs -signed_connectivity (binding → inhibition):")
print(f"   Spearman r = {sp_signed:.4f}")

# 2. Per-gene correlation
print(f"\n2. Per-gene Spearman (CG vs |connectivity|):")
gene_sp = []
for gi in range(len(shared_genes)):
    r, _ = spearmanr(cg_scores[:, gi], conn_sub[:, gi])
    if not np.isnan(r):
        gene_sp.append(r)
print(f"   Median = {np.median(gene_sp):.4f}, Mean = {np.mean(gene_sp):.4f}")
print(f"   Genes evaluated: {len(gene_sp)}")
print(f"   >0: {sum(1 for r in gene_sp if r > 0)}, <0: {sum(1 for r in gene_sp if r < 0)}")

# 3. Per-compound correlation
print(f"\n3. Per-compound Spearman (CG vs |connectivity|):")
comp_sp = []
for ci in range(len(shared_compounds)):
    r, _ = spearmanr(cg_scores[ci, :], conn_sub[ci, :])
    if not np.isnan(r):
        comp_sp.append(r)
print(f"   Median = {np.median(comp_sp):.4f}, Mean = {np.mean(comp_sp):.4f}")
print(f"   Compounds evaluated: {len(comp_sp)}")
print(f"   >0: {sum(1 for r in comp_sp if r > 0)}, <0: {sum(1 for r in comp_sp if r < 0)}")

# 4. AUROC on top-connected pairs
print(f"\n4. AUROC — can CG scores identify highly connected pairs?")
for thr_pct in [1, 5, 10]:
    thr = np.percentile(flat_conn, 100 - thr_pct)
    labels = (flat_conn >= thr).astype(int)
    if 0 < labels.sum() < len(labels):
        auroc = roc_auc_score(labels, flat_cg)
        print(f"   Top {thr_pct}% connected: AUROC = {auroc:.4f} ({labels.sum()} positives)")

# 5. Recall@k
print(f"\n5. Recall@k — per gene, overlap of top CG-scored and top L1000-connected compounds:")
for k_pct in [1, 5, 10]:
    hits, total = 0, 0
    for gi in range(len(shared_genes)):
        nc = len(shared_compounds)
        k = max(1, int(nc * k_pct / 100))
        top_conn_set = set(np.argsort(-conn_sub[:, gi])[:k])
        top_cg_set = set(np.argsort(-cg_scores[:, gi])[:k])
        hits += len(top_conn_set & top_cg_set)
        total += k
    recall = 100 * hits / total if total else 0
    print(f"   Recall@{k_pct}%: {recall:.2f}% (random: {k_pct}%)")

# 6. Gene-level AUROC: for each gene, treat top-10% L1000-connected compounds as positives
print(f"\n6. Per-gene AUROC — CG scores predict top-connected compounds:")
gene_aurocs = []
for gi in range(len(shared_genes)):
    thr = np.percentile(conn_sub[:, gi], 90)
    labels = (conn_sub[:, gi] >= thr).astype(int)
    if 0 < labels.sum() < len(labels):
        auc = roc_auc_score(labels, cg_scores[:, gi])
        gene_aurocs.append(auc)

print(f"   Median AUROC = {np.median(gene_aurocs):.4f}")
print(f"   Mean AUROC   = {np.mean(gene_aurocs):.4f}")
print(f"   Genes evaluated: {len(gene_aurocs)}")
print(f"   Random baseline: 0.5000")

print(f"\n{'='*70}")
print("Done.")
