"""
Evaluate Bonbon CG profiles on Tahoe-100M pseudobulk expression prediction.

Input:
  - pseudobulk_means.npz: pseudobulk expression per (drug, cell_line)
  - tahoe_cg_profiles.npz: CG profiles for drugs (357 × 945)

Protocol:
  - Select HVGs from pseudobulk variance
  - Compute expression deltas (perturbed - control per cell line)
  - Ridge regression: CG profile → expression delta
  - Evaluate Pearson delta, direction accuracy, discrimination score
  - Data efficiency: train on 1%/5%/10%/100% of data
  - Compare: zero-shot CG score (direct), Ridge, MLP
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import defaultdict
import time
import os

BASE = "/opt/dlami/nvme/tahoe"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def p(*a, **k):
    print(*a, **k, flush=True)

# ─── Load data ───
p("Loading pseudobulk data...")
pb = np.load(f"{BASE}/pseudobulk_means.npz", allow_pickle=True)
means = pb['means']  # [N_conditions, N_genes]
drug_names = pb['drug_names']
cell_line_ids = pb['cell_line_ids']
cell_counts = pb['cell_counts']
gene_indices = pb['gene_indices']
control_means = pb['control_means']  # [N_cell_lines, N_genes]
control_cell_lines = pb['control_cell_lines']
control_cell_counts = pb['control_cell_counts']

p(f"  Conditions: {means.shape[0]}")
p(f"  Genes: {means.shape[1]}")
p(f"  Control cell lines: {len(control_cell_lines)}")

# Map cell_line -> control mean
ctrl_map = {cl: control_means[i] for i, cl in enumerate(control_cell_lines)}

# ─── Load CG profiles ───
p("\nLoading CG profiles...")
cg = np.load(f"{BASE}/tahoe_cg_profiles.npz", allow_pickle=True)
cg_profiles = cg['cg_profiles']  # [N_drugs, 945]
cg_drugs = list(cg['drugs'])
cg_proteins = list(cg['proteins'])
p(f"  CG profiles: {cg_profiles.shape}")

# ─── Match drugs between Tahoe and CG profiles ───
# Tahoe drug names may differ slightly (case, suffixes)
drug_name_map = {}
for td in np.unique(drug_names):
    td_lower = td.lower().strip()
    for i, cd in enumerate(cg_drugs):
        if cd.lower().strip() == td_lower:
            drug_name_map[td] = i
            break

p(f"  Matched: {len(drug_name_map)} / {len(np.unique(drug_names))} Tahoe drugs")

# Filter to conditions where we have both CG profiles and control means
valid_mask = np.array([
    (dn in drug_name_map) and (cl in ctrl_map)
    for dn, cl in zip(drug_names, cell_line_ids)
])
# Also require minimum cells
min_cells = 10
valid_mask &= (cell_counts >= min_cells)

valid_idx = np.where(valid_mask)[0]
p(f"  Valid conditions: {len(valid_idx)} (of {len(drug_names)})")
p(f"  Drugs covered: {len(set(drug_names[valid_idx]))}")
p(f"  Cell lines covered: {len(set(cell_line_ids[valid_idx]))}")

if len(valid_idx) < 50:
    p("ERROR: Too few valid conditions. Check drug name matching.")
    import sys; sys.exit(1)

# Build matrices
X_cg = np.array([cg_profiles[drug_name_map[drug_names[i]]] for i in valid_idx])
Y_expr = means[valid_idx]
Y_ctrl = np.array([ctrl_map[cell_line_ids[i]] for i in valid_idx])
Y_delta = Y_expr - Y_ctrl  # expression deltas

valid_drugs = drug_names[valid_idx]
valid_cls = cell_line_ids[valid_idx]

p(f"\n  X (CG profiles): {X_cg.shape}")
p(f"  Y (expression delta): {Y_delta.shape}")

# ─── Select HVGs ───
p("\nSelecting HVGs...")
delta_var = np.var(Y_delta, axis=0)
delta_var_nonzero = delta_var > 1e-10
n_nonzero = delta_var_nonzero.sum()
p(f"  Non-zero variance genes: {n_nonzero}")

# Top 2000 most variable genes
n_hvg = min(2000, n_nonzero)
hvg_idx = np.argsort(-delta_var)[:n_hvg]
Y_delta_hvg = Y_delta[:, hvg_idx]
p(f"  Selected {n_hvg} HVGs")

# ─── Evaluation metrics ───
def eval_pearson_delta(pred, true):
    corrs = []
    for i in range(pred.shape[0]):
        if np.std(pred[i]) < 1e-8 or np.std(true[i]) < 1e-8:
            continue
        r, _ = pearsonr(pred[i], true[i])
        if not np.isnan(r):
            corrs.append(r)
    return np.mean(corrs) if corrs else float('nan'), corrs

def eval_direction_accuracy(pred, true):
    accs = []
    for i in range(pred.shape[0]):
        sign_t = np.sign(true[i])
        sign_p = np.sign(pred[i])
        acc = (sign_t == sign_p).mean()
        accs.append(acc)
    return np.mean(accs), accs

def eval_top_deg(pred, true, top_k=50):
    corrs = []
    for i in range(pred.shape[0]):
        abs_delta = np.abs(true[i])
        top_idx = np.argsort(-abs_delta)[:top_k]
        p_top = pred[i, top_idx]
        t_top = true[i, top_idx]
        if np.std(p_top) < 1e-8 or np.std(t_top) < 1e-8:
            continue
        r, _ = pearsonr(p_top, t_top)
        if not np.isnan(r):
            corrs.append(r)
    return np.mean(corrs) if corrs else float('nan'), corrs

def eval_discrimination(pred_means, true_means, metric='cosine'):
    from scipy.spatial.distance import cdist
    if pred_means.shape[0] == 0:
        return float('nan')
    dist = cdist(pred_means, true_means, metric=metric)
    N = dist.shape[0]
    ranks = []
    for i in range(N):
        d_true = dist[i, i]
        rank = np.sum(dist[i] < d_true)
        ranks.append(rank)
    mean_rank = np.mean(ranks)
    return 1.0 - 2.0 * (mean_rank / N)

def full_eval(pred_delta, true_delta, name=""):
    pd_mean, pd_corrs = eval_pearson_delta(pred_delta, true_delta)
    da_mean, _ = eval_direction_accuracy(pred_delta, true_delta)
    deg50_mean, deg50_corrs = eval_top_deg(pred_delta, true_delta, 50)
    disc = eval_discrimination(pred_delta, true_delta)
    mse = np.mean((pred_delta - true_delta) ** 2)

    p(f"  {name:40s} | Pearson Δ: {pd_mean:.4f} | DEG-50: {deg50_mean:.4f} | "
      f"Dir Acc: {da_mean:.4f} | Disc: {disc:.4f} | MSE: {mse:.6f}")
    return {
        'pearson_delta': pd_mean, 'deg50': deg50_mean,
        'dir_acc': da_mean, 'discrimination': disc, 'mse': mse,
        'pearson_corrs': pd_corrs, 'deg50_corrs': deg50_corrs
    }

# ─── Evaluation ───
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n = len(valid_idx)

p(f"\n{'='*90}")
p(f"TAHOE-100M EVALUATION: {n} conditions, {n_hvg} HVGs")
p(f"{'='*90}")

# ─── 0. Pure zero-shot: direct CG scores for overlapping genes ───
p("\n--- Zero-shot: Direct CG scores (no training) ---")
import json
with open(f"{BASE}/protein_to_gene_tid.json") as f:
    protein_to_tid = json.load(f)

# Find which HVG positions correspond to our 945 proteins
hvg_gene_indices = gene_indices[hvg_idx]
prot_to_cg_col = {p: i for i, p in enumerate(cg_proteins)}
tid_to_protein = {int(tid): p for p, tid in protein_to_tid.items()}

# For each HVG, check if it's one of our 945 proteins
hvg_to_cg = {}
for hi, gidx in enumerate(hvg_gene_indices):
    if int(gidx) in tid_to_protein:
        prot = tid_to_protein[int(gidx)]
        if prot in prot_to_cg_col:
            hvg_to_cg[hi] = prot_to_cg_col[prot]

p(f"  HVGs overlapping with our 945 proteins: {len(hvg_to_cg)}/{n_hvg}")

if len(hvg_to_cg) >= 10:
    # For overlapping genes only: CG score IS the predicted delta
    overlap_hvg = sorted(hvg_to_cg.keys())
    direct_preds = np.zeros((n, len(overlap_hvg)))
    for i, hi in enumerate(overlap_hvg):
        direct_preds[:, i] = X_cg[:, hvg_to_cg[hi]]
    true_overlap = Y_delta_hvg[:, overlap_hvg]

    pd_mean, _ = eval_pearson_delta(direct_preds, true_overlap)
    da_mean, _ = eval_direction_accuracy(direct_preds, true_overlap)
    p(f"  {'Direct CG (overlap genes only)':40s} | Pearson Δ: {pd_mean:.4f} | "
      f"Dir Acc: {da_mean:.4f} | ({len(overlap_hvg)} genes)")

# ─── 1. Zero-shot baseline: mean prediction ───
p("\n--- Baseline: predict mean delta (train mean) ---")
mean_preds = np.zeros_like(Y_delta_hvg)
for tr, te in kf.split(X_cg):
    mean_preds[te] = Y_delta_hvg[tr].mean(axis=0)
full_eval(mean_preds, Y_delta_hvg, "Train Mean")

# ─── 2. Ridge regression: CG profiles → HVG deltas ───
p("\n--- Ridge: CG profiles → HVG deltas ---")
for alpha in [1.0, 10.0, 100.0, 1000.0]:
    ridge_preds = np.zeros_like(Y_delta_hvg)
    for tr, te in kf.split(X_cg):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cg[tr])
        X_te = sc.transform(X_cg[te])
        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_delta_hvg[tr])
        ridge_preds[te] = model.predict(X_te)
    full_eval(ridge_preds, Y_delta_hvg, f"Ridge(α={alpha})")

# ─── 4. Cell-line conditioned Ridge ───
p("\n--- Cell-line conditioned Ridge ---")
unique_cls = sorted(set(valid_cls))
cl_to_idx = {cl: i for i, cl in enumerate(unique_cls)}

# One-hot encode cell lines
cl_onehot = np.zeros((n, len(unique_cls)))
for i, cl in enumerate(valid_cls):
    cl_onehot[i, cl_to_idx[cl]] = 1.0

X_cg_cl = np.hstack([X_cg, cl_onehot])
p(f"  CG + cell-line features: {X_cg_cl.shape}")

for alpha in [10.0, 100.0]:
    cl_preds = np.zeros_like(Y_delta_hvg)
    for tr, te in kf.split(X_cg_cl):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cg_cl[tr])
        X_te = sc.transform(X_cg_cl[te])
        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_delta_hvg[tr])
        cl_preds[te] = model.predict(X_te)
    full_eval(cl_preds, Y_delta_hvg, f"Ridge+CellLine(α={alpha})")

# ─── 4b. PCA-reduced CG profiles ───
p("\n--- PCA-reduced CG profiles → Ridge ---")
from sklearn.decomposition import PCA
for n_comp in [50, 100, 200]:
    pca_preds = np.zeros_like(Y_delta_hvg)
    for tr, te in kf.split(X_cg):
        pca = PCA(n_components=n_comp)
        sc = StandardScaler()
        X_tr = pca.fit_transform(sc.fit_transform(X_cg[tr]))
        X_te = pca.transform(sc.transform(X_cg[te]))
        model = Ridge(alpha=100.0)
        model.fit(X_tr, Y_delta_hvg[tr])
        pca_preds[te] = model.predict(X_te)
    full_eval(pca_preds, Y_delta_hvg, f"PCA({n_comp})+Ridge(α=100)")

# ─── 4c. Leave-one-drug-out CV (unseen drug evaluation) ───
p("\n--- Leave-one-drug-out CV (unseen drugs) ---")
unique_drugs_valid = sorted(set(valid_drugs))
p(f"  {len(unique_drugs_valid)} unique drugs")

lodo_preds = np.zeros_like(Y_delta_hvg)
lodo_mask = np.zeros(n, dtype=bool)

for di, drug in enumerate(unique_drugs_valid):
    te_mask = valid_drugs == drug
    tr_mask = ~te_mask
    if te_mask.sum() == 0 or tr_mask.sum() < 10:
        continue

    sc = StandardScaler()
    X_tr = sc.fit_transform(X_cg[tr_mask])
    X_te = sc.transform(X_cg[te_mask])
    model = Ridge(alpha=100.0)
    model.fit(X_tr, Y_delta_hvg[tr_mask])
    lodo_preds[te_mask] = model.predict(X_te)
    lodo_mask |= te_mask

if lodo_mask.sum() > 0:
    full_eval(lodo_preds[lodo_mask], Y_delta_hvg[lodo_mask], "Leave-one-drug-out Ridge(α=100)")

# ─── 5. MLP ───
p("\n--- MLP: CG profiles → HVG deltas ---")
X_t = torch.tensor(X_cg, dtype=torch.float32)
Y_t = torch.tensor(Y_delta_hvg, dtype=torch.float32)

def train_mlp(X_feat, hidden_dims, epochs, lr, dropout, wd, name="MLP"):
    X_in = torch.tensor(X_feat, dtype=torch.float32) if isinstance(X_feat, np.ndarray) else X_feat
    all_preds = torch.zeros_like(Y_t)
    t0 = time.time()

    for fold, (tr, te) in enumerate(kf.split(X_feat)):
        x_tr, y_tr = X_in[tr].to(device), Y_t[tr].to(device)
        x_te = X_in[te].to(device)

        mu, std = x_tr.mean(0), x_tr.std(0).clamp(min=1e-8)
        x_tr = (x_tr - mu) / std
        x_te = (x_te - mu) / std

        layers = []
        in_dim = x_tr.shape[1]
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, n_hvg))
        model = nn.Sequential(*layers).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
    return full_eval(preds_np, Y_delta_hvg, name), preds_np

train_mlp(X_cg, [512], 300, 1e-3, 0.3, 1e-3, "MLP [512] dp=0.3")
train_mlp(X_cg, [512, 256], 400, 5e-4, 0.4, 5e-3, "MLP [512,256] dp=0.4")
train_mlp(X_cg_cl, [512], 300, 1e-3, 0.3, 1e-3, "MLP+CL [512] dp=0.3")

# ─── 6. Data efficiency: 1%, 5%, 10%, 25%, 50%, 100% ───
p(f"\n{'='*90}")
p("DATA EFFICIENCY: Ridge(α=100) at varying training fractions")
p(f"{'='*90}")

best_alpha = 100.0  # from above

for frac in [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]:
    frac_preds = np.zeros_like(Y_delta_hvg)
    n_train_list = []

    for fold, (tr, te) in enumerate(kf.split(X_cg)):
        n_sub = max(1, int(len(tr) * frac))
        np.random.seed(fold + 42)
        sub = np.random.choice(tr, size=n_sub, replace=False)
        n_train_list.append(n_sub)

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cg[sub])
        X_te = sc.transform(X_cg[te])

        model = Ridge(alpha=best_alpha)
        model.fit(X_tr, Y_delta_hvg[sub])
        frac_preds[te] = model.predict(X_te)

    avg_n = int(np.mean(n_train_list))
    full_eval(frac_preds, Y_delta_hvg, f"Ridge {frac*100:.0f}% ({avg_n} train)")

# ─── 7. Per cell-line breakdown ───
p(f"\n{'='*90}")
p("PER CELL-LINE BREAKDOWN (Ridge α=100)")
p(f"{'='*90}")

ridge_preds_full = np.zeros_like(Y_delta_hvg)
for tr, te in kf.split(X_cg):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_cg[tr])
    X_te = sc.transform(X_cg[te])
    model = Ridge(alpha=best_alpha)
    model.fit(X_tr, Y_delta_hvg[tr])
    ridge_preds_full[te] = model.predict(X_te)

for cl in sorted(set(valid_cls)):
    cl_mask = valid_cls == cl
    if cl_mask.sum() < 10:
        continue
    pd_mean, _ = eval_pearson_delta(ridge_preds_full[cl_mask], Y_delta_hvg[cl_mask])
    deg50, _ = eval_top_deg(ridge_preds_full[cl_mask], Y_delta_hvg[cl_mask], 50)
    da, _ = eval_direction_accuracy(ridge_preds_full[cl_mask], Y_delta_hvg[cl_mask])
    p(f"  {cl:12s} (n={cl_mask.sum():3d}): Pearson Δ={pd_mean:.4f} DEG-50={deg50:.4f} DirAcc={da:.4f}")

# ─── Save results ───
np.savez(f"{BASE}/tahoe_eval_results.npz",
    ridge_preds_full=ridge_preds_full,
    Y_delta_hvg=Y_delta_hvg,
    X_cg=X_cg,
    valid_drugs=valid_drugs,
    valid_cls=valid_cls,
    hvg_idx=hvg_idx,
    gene_indices=gene_indices)
p(f"\nResults saved to {BASE}/tahoe_eval_results.npz")

p(f"\n{'='*90}")
p("Done.")
