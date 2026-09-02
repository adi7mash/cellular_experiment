"""
LINCS L1000 Full Evaluation v2
- All embedding types: codebook tokens, pooled_attention, contrastive projection
- All scoring: dot product, cosine
- Multi-input Ridge probe per gene
- Multi-output MLP on GPU
- PPI propagation
- 10µM/24h filtered L1000
"""
import sys, os, gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import h5py
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix, diags

BASE = "/opt/dlami/nvme/lincs_l1000"
V2 = f"{BASE}/v2"
CONN_CACHE = f"{BASE}/connectivity_10uM_24h.npz"

def p(*a, **k): print(*a, **k, flush=True)

# ──────────────────────────────────────────────────────────────
# Step 1: Precompute L1000 connectivity (or load from cache)
# ──────────────────────────────────────────────────────────────
if os.path.exists(CONN_CACHE):
    p("Loading cached connectivity...")
    cache = np.load(CONN_CACHE, allow_pickle=True)
    connectivity = cache['connectivity']
    eval_compounds = list(cache['compounds'])
    eval_genes = list(cache['genes'])
    p(f"  {len(eval_compounds)} compounds × {len(eval_genes)} genes")
else:
    p("Building L1000 connectivity from GCTX (10µM, 24h)...")
    sig_info = pd.read_csv(f"{BASE}/sig_info.txt.gz", sep="\t", low_memory=False)
    gene_info = pd.read_csv(f"{BASE}/gene_info.txt.gz", sep="\t", low_memory=False)
    compound_df = pd.read_csv(f"{BASE}/compound_to_safe.tsv", sep="\t")
    landmark_genes = pd.read_csv(f"{BASE}/landmark_genes.tsv", sep="\t")

    lm_gene_ids = gene_info[gene_info['pr_is_lm'] == 1]['pr_gene_id'].tolist()
    ts_compounds = set(compound_df['compound_id'].tolist())
    mapped_lm_symbols = set(landmark_genes['gene'].tolist())

    with h5py.File(f"{BASE}/level5.gctx", 'r') as f:
        row_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0']['META']['ROW']['id'][:]]
        col_ids = [x.decode() if isinstance(x, bytes) else str(x) for x in f['0']['META']['COL']['id'][:]]

    row_id_to_idx = {rid: i for i, rid in enumerate(row_ids)}
    col_id_to_idx = {cid: i for i, cid in enumerate(col_ids)}
    lm_indices = np.array([row_id_to_idx[str(gid)] for gid in lm_gene_ids if str(gid) in row_id_to_idx])
    lm_sorted_order = np.argsort(lm_indices)
    lm_sorted_indices = lm_indices[lm_sorted_order]

    cp_filtered = sig_info[(sig_info['pert_type'] == 'trt_cp') &
                           (sig_info['pert_iname'].isin(ts_compounds)) &
                           (sig_info['pert_idose'] == '10 µM') &
                           (sig_info['pert_itime'] == '24 h')]
    sh_sigs = sig_info[(sig_info['pert_type'].isin(['trt_sh', 'trt_sh.cgs'])) &
                       (sig_info['pert_iname'].isin(mapped_lm_symbols))]

    compound_sig_groups = cp_filtered.groupby('pert_iname')['sig_id'].apply(list).to_dict()
    gene_sig_groups = sh_sigs.groupby('pert_iname')['sig_id'].apply(list).to_dict()

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
    p(f"  Reading {len(all_sig_list)} signatures...")

    with h5py.File(f"{BASE}/level5.gctx", 'r') as f:
        matrix = f['0']['DATA']['0']['matrix']
        sig_data = np.zeros((len(all_sig_list), len(lm_indices)), dtype=np.float32)
        for start in range(0, len(all_sig_list), 5000):
            end = min(start + 5000, len(all_sig_list))
            chunk_indices = all_sig_list[start:end]
            chunk_sorted = np.sort(chunk_indices)
            chunk_data = matrix[chunk_sorted][:, lm_sorted_indices]
            for i, ci in enumerate(chunk_indices):
                pos = np.searchsorted(chunk_sorted, ci)
                sig_data[sig_to_local[ci]] = chunk_data[pos][lm_sorted_order]

    compound_consensus = {}
    for cname, indices in compound_sig_idx.items():
        compound_consensus[cname] = sig_data[[sig_to_local[i] for i in indices]].mean(axis=0)
    gene_consensus = {}
    for gname, indices in gene_sig_idx.items():
        gene_consensus[gname] = sig_data[[sig_to_local[i] for i in indices]].mean(axis=0)
    del sig_data

    eval_compounds = sorted(compound_consensus.keys())
    eval_genes = sorted(set(gene_consensus.keys()) & mapped_lm_symbols)

    C_mat = np.array([compound_consensus[c] for c in eval_compounds])
    G_mat = np.array([gene_consensus[g] for g in eval_genes])
    C_norm = C_mat / (np.linalg.norm(C_mat, axis=1, keepdims=True) + 1e-10)
    G_norm = G_mat / (np.linalg.norm(G_mat, axis=1, keepdims=True) + 1e-10)
    connectivity = np.abs(C_norm @ G_norm.T).astype(np.float32)

    np.savez_compressed(CONN_CACHE, connectivity=connectivity,
                        compounds=eval_compounds, genes=eval_genes)
    p(f"  Saved: {len(eval_compounds)} compounds × {len(eval_genes)} genes")
    del C_mat, G_mat, C_norm, G_norm, compound_consensus, gene_consensus

n_c, n_g = connectivity.shape
p(f"\nEvaluation matrix: {n_c} × {n_g}")

# ──────────────────────────────────────────────────────────────
# Step 2: Load all embedding types
# ──────────────────────────────────────────────────────────────
p("\nLoading embeddings...")

def load_emb(directory, names, subdirs=None):
    loaded = {}
    for name in names:
        if subdirs:
            for sub in subdirs:
                path = os.path.join(directory, sub, f"{name}.pt")
                if os.path.exists(path):
                    loaded.setdefault(sub, {})[name] = torch.load(path, map_location='cpu', weights_only=True).float()
        else:
            path = os.path.join(directory, f"{name}.pt")
            if os.path.exists(path):
                loaded[name] = torch.load(path, map_location='cpu', weights_only=True).float()
    return loaded

# v2 codebook with subdirs
mol_cb = load_emb(f"{V2}/lincs_pairs_molecule_codebook_dark-snowball-245", eval_compounds, ['tokens', 'pooled_attention'])
prot_cb = load_emb(f"{V2}/lincs_pairs_protein_codebook_dark-snowball-245", eval_genes, ['tokens', 'pooled_attention'])

# v2 contrastive projection (flat)
mol_proj = load_emb(f"{V2}/lincs_pairs_molecule_molecule_mean_dark-snowball-245", eval_compounds)
prot_proj = load_emb(f"{V2}/lincs_pairs_protein_protein_mean_dark-snowball-245", eval_genes)

for label, d in [("mol_cb_tokens", mol_cb.get('tokens', {})),
                 ("mol_cb_pa", mol_cb.get('pooled_attention', {})),
                 ("prot_cb_tokens", prot_cb.get('tokens', {})),
                 ("prot_cb_pa", prot_cb.get('pooled_attention', {})),
                 ("mol_proj", mol_proj), ("prot_proj", prot_proj)]:
    p(f"  {label}: {len(d)} loaded")

# Find common entities across all types
common_compounds = set(eval_compounds)
common_genes = set(eval_genes)
for d in [mol_cb.get('tokens', {}), mol_cb.get('pooled_attention', {}), mol_proj]:
    common_compounds &= set(d.keys())
for d in [prot_cb.get('tokens', {}), prot_cb.get('pooled_attention', {}), prot_proj]:
    common_genes &= set(d.keys())

compounds = sorted(common_compounds)
genes = sorted(common_genes)
p(f"\nCommon: {len(compounds)} compounds × {len(genes)} genes")

# Build index maps
c_idx = [eval_compounds.index(c) for c in compounds]
g_idx = [eval_genes.index(g) for g in genes]
conn_sub = connectivity[np.ix_(c_idx, g_idx)]

# Build matrices for each embedding type
def stack(d, keys):
    return torch.stack([d[k] for k in keys])

mol_tok_mat = stack(mol_cb['tokens'], compounds)
prot_tok_mat = stack(prot_cb['tokens'], genes)
mol_pa_mat = stack(mol_cb['pooled_attention'], compounds)
prot_pa_mat = stack(prot_cb['pooled_attention'], genes)
mol_proj_mat = stack(mol_proj, compounds)
prot_proj_mat = stack(prot_proj, compounds if False else genes)

emb_dim = mol_tok_mat.shape[1]
proj_dim = mol_proj_mat.shape[1]
p(f"Codebook dim: {emb_dim}, Projection dim: {proj_dim}")

# ──────────────────────────────────────────────────────────────
# Step 3: Direct scoring evaluation
# ──────────────────────────────────────────────────────────────
def cosine_mat(A, B):
    An = A / (A.norm(dim=1, keepdim=True) + 1e-8)
    Bn = B / (B.norm(dim=1, keepdim=True) + 1e-8)
    return (An @ Bn.T).numpy()

scoring_methods = {
    "CB tokens dot": (mol_tok_mat @ prot_tok_mat.T).numpy(),
    "CB tokens cosine": cosine_mat(mol_tok_mat, prot_tok_mat),
    "CB pooled_attn dot": (mol_pa_mat @ prot_pa_mat.T).numpy(),
    "CB pooled_attn cosine": cosine_mat(mol_pa_mat, prot_pa_mat),
    "Contrastive dot": (mol_proj_mat @ prot_proj_mat.T).numpy(),
    "Contrastive cosine": cosine_mat(mol_proj_mat, prot_proj_mat),
}

p(f"\n{'='*70}")
p(f"LINCS L1000 FULL EVALUATION v2 — {len(compounds)} cpds × {len(genes)} genes")
p(f"{'='*70}")

p(f"\n{'Method':<28} {'Spearman':>9} {'AUROC-5%':>9} {'GeneSp':>9} {'GeneAUC':>9}")
p(f"{'-'*64}")

results = {}
for name, scores in scoring_methods.items():
    flat_s, flat_c = scores.ravel(), conn_sub.ravel()
    sp_r, _ = spearmanr(flat_s, flat_c)
    thr = np.percentile(flat_c, 95)
    labels = (flat_c >= thr).astype(int)
    auroc5 = roc_auc_score(labels, flat_s)

    gene_sp = []
    gene_auc = []
    for gi in range(len(genes)):
        r, _ = spearmanr(scores[:, gi], conn_sub[:, gi])
        if not np.isnan(r): gene_sp.append(r)
        t90 = np.percentile(conn_sub[:, gi], 90)
        lab = (conn_sub[:, gi] >= t90).astype(int)
        if 0 < lab.sum() < len(lab):
            gene_auc.append(roc_auc_score(lab, scores[:, gi]))

    p(f"  {name:<26} {sp_r:>9.4f} {auroc5:>9.4f} {np.median(gene_sp):>9.4f} {np.median(gene_auc):>9.4f}")
    results[name] = scores

# ──────────────────────────────────────────────────────────────
# Step 4: Multi-input Ridge probes (CG profile → connectivity)
# ──────────────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("MULTI-INPUT RIDGE PROBE (5-fold CV per gene)")
p(f"{'='*70}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def ridge_probe(X, Y, name, alpha=100.0):
    gene_sp, gene_auc = [], []
    nc = X.shape[0]
    for gi in range(Y.shape[1]):
        y = Y[:, gi]
        preds = np.zeros(nc)
        for tr, te in kf.split(X):
            sc = StandardScaler()
            model = Ridge(alpha=alpha)
            model.fit(sc.fit_transform(X[tr]), y[tr])
            preds[te] = model.predict(sc.transform(X[te]))
        r, _ = spearmanr(preds, y)
        if not np.isnan(r): gene_sp.append(r)
        t90 = np.percentile(y, 90)
        lab = (y >= t90).astype(int)
        if 0 < lab.sum() < len(lab):
            gene_auc.append(roc_auc_score(lab, preds))
    p(f"  {name:<40} Sp={np.median(gene_sp):.4f} AUC={np.median(gene_auc):.4f}")
    return gene_sp, gene_auc

# Score-based probes
for name, scores in scoring_methods.items():
    ridge_probe(scores, conn_sub, f"Ridge({name})")

# Raw embedding probes
ridge_probe(mol_tok_mat.numpy(), conn_sub, "Ridge(mol_tok 1280d)")
ridge_probe(mol_pa_mat.numpy(), conn_sub, "Ridge(mol_pa 1280d)")
ridge_probe(mol_proj_mat.numpy(), conn_sub, f"Ridge(mol_proj {proj_dim}d)")

# Combined probes
combined_scores = np.hstack([results["CB tokens dot"], results["CB pooled_attn dot"], results["Contrastive dot"]])
ridge_probe(combined_scores, conn_sub, "Ridge(all 3 scores concat)", alpha=200.0)

combined_emb = np.hstack([mol_tok_mat.numpy(), mol_pa_mat.numpy(), mol_proj_mat.numpy()])
ridge_probe(combined_emb, conn_sub, f"Ridge(all 3 embs concat {combined_emb.shape[1]}d)", alpha=500.0)

# ──────────────────────────────────────────────────────────────
# Step 5: Multi-output MLP on GPU
# ──────────────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("MULTI-OUTPUT MLP (compound → all gene connectivities)")
p(f"{'='*70}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p(f"Device: {device}")

def mlp_probe(X_np, Y_np, name, hidden=512, epochs=200, lr=1e-3):
    X_t = torch.tensor(X_np, dtype=torch.float32)
    Y_t = torch.tensor(Y_np, dtype=torch.float32)
    n_in = X_t.shape[1]
    n_out = Y_t.shape[1]

    all_preds = torch.zeros_like(Y_t)

    for fold, (tr, te) in enumerate(kf.split(X_t)):
        x_tr, y_tr = X_t[tr].to(device), Y_t[tr].to(device)
        x_te = X_t[te].to(device)

        mu, std = x_tr.mean(0), x_tr.std(0) + 1e-8
        x_tr = (x_tr - mu) / std
        x_te = (x_te - mu) / std

        model = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden // 2, n_out)
        ).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for ep in range(epochs):
            pred = model(x_tr)
            loss = nn.MSELoss()(pred, y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            all_preds[te] = model(x_te).cpu()

    preds_np = all_preds.numpy()
    gene_sp, gene_auc = [], []
    for gi in range(n_out):
        r, _ = spearmanr(preds_np[:, gi], Y_np[:, gi])
        if not np.isnan(r): gene_sp.append(r)
        t90 = np.percentile(Y_np[:, gi], 90)
        lab = (Y_np[:, gi] >= t90).astype(int)
        if 0 < lab.sum() < len(lab):
            gene_auc.append(roc_auc_score(lab, preds_np[:, gi]))

    p(f"  {name:<40} Sp={np.median(gene_sp):.4f} AUC={np.median(gene_auc):.4f} (>{sum(1 for a in gene_auc if a>0.5)}/{len(gene_auc)} genes)")
    return gene_sp, gene_auc

# CG score inputs
mlp_probe(results["CB tokens dot"], conn_sub, "MLP(CB_tok_dot → conn)")
mlp_probe(results["Contrastive cosine"], conn_sub, "MLP(contr_cos → conn)")
mlp_probe(combined_scores, conn_sub, "MLP(all_scores → conn)")

# Raw embedding inputs
mlp_probe(mol_tok_mat.numpy(), conn_sub, "MLP(mol_tok_1280d → conn)")
mlp_probe(mol_proj_mat.numpy(), conn_sub, f"MLP(mol_proj_{proj_dim}d → conn)")
mlp_probe(combined_emb, conn_sub, f"MLP(all_emb_{combined_emb.shape[1]}d → conn)", hidden=1024, epochs=300)

# ──────────────────────────────────────────────────────────────
# Step 6: PPI-propagated scores + Ridge
# ──────────────────────────────────────────────────────────────
p(f"\n{'='*70}")
p("PPI-PROPAGATED SCORES")
p(f"{'='*70}")

info = pd.read_csv(f"{BASE}/string_info.txt.gz", sep="\t")
string_to_gene = dict(zip(info['#string_protein_id'], info['preferred_name']))
ppi = pd.read_csv(f"{BASE}/string_ppi.txt.gz", sep=" ")
ppi_hc = ppi[ppi['combined_score'] >= 700].copy()
ppi_hc['gene1'] = ppi_hc['protein1'].map(string_to_gene)
ppi_hc['gene2'] = ppi_hc['protein2'].map(string_to_gene)
ppi_hc = ppi_hc.dropna(subset=['gene1', 'gene2'])

gene_to_idx_ppi = {g: i for i, g in enumerate(genes)}
rows, cols, weights = [], [], []
for _, r in ppi_hc.iterrows():
    g1, g2 = r['gene1'], r['gene2']
    if g1 in gene_to_idx_ppi and g2 in gene_to_idx_ppi:
        i, j = gene_to_idx_ppi[g1], gene_to_idx_ppi[g2]
        w = r['combined_score'] / 1000.0
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([w, w])

A = csr_matrix((weights, (rows, cols)), shape=(len(genes), len(genes)))
degrees = np.array(A.sum(axis=1)).ravel()
D_inv = 1.0 / (degrees + 1e-10)
D_inv[degrees == 0] = 0
W = diags(D_inv) @ A

p(f"PPI: {len(genes)} genes, {A.nnz} edges, {(degrees > 0).sum()} connected")

for alpha in [0.1, 0.3, 0.5]:
    for score_name in ["CB tokens dot", "Contrastive cosine"]:
        scores = results[score_name].copy()
        for _ in range(3):
            scores = alpha * results[score_name] + (1 - alpha) * (scores @ W.T.toarray())
        sp_r, _ = spearmanr(scores.ravel(), conn_sub.ravel())
        thr = np.percentile(conn_sub.ravel(), 95)
        auroc5 = roc_auc_score((conn_sub.ravel() >= thr).astype(int), scores.ravel())
        p(f"  PPI({score_name}, α={alpha}): Sp={sp_r:.4f} AUC5={auroc5:.4f}")

# PPI-propagated Ridge
best_scores = results["CB tokens dot"].copy()
for _ in range(3):
    best_scores = 0.1 * results["CB tokens dot"] + 0.9 * (best_scores @ W.T.toarray())
ridge_probe(best_scores, conn_sub, "Ridge(PPI-CB_tok α=0.1)")

p(f"\n{'='*70}")
p("Done.")
