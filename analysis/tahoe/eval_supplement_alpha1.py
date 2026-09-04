"""Supplementary eval: data efficiency and LODO with α=1.0 (optimal)."""
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import time

BASE = "/opt/dlami/nvme/tahoe"

def p(*a, **k):
    print(*a, **k, flush=True)

# Load data (same as main eval)
pb = np.load(f"{BASE}/pseudobulk_means.npz", allow_pickle=True)
means = pb['means']
drug_names = pb['drug_names']
cell_line_ids = pb['cell_line_ids']
cell_counts = pb['cell_counts']
gene_indices = pb['gene_indices']
control_means = pb['control_means']
control_cell_lines = pb['control_cell_lines']

ctrl_map = {cl: control_means[i] for i, cl in enumerate(control_cell_lines)}

cg = np.load(f"{BASE}/tahoe_cg_profiles.npz", allow_pickle=True)
cg_profiles = cg['cg_profiles']
cg_drugs = list(cg['drugs'])

drug_name_map = {}
for td in np.unique(drug_names):
    td_lower = td.lower().strip()
    for i, cd in enumerate(cg_drugs):
        if cd.lower().strip() == td_lower:
            drug_name_map[td] = i
            break

valid_mask = np.array([
    (dn in drug_name_map) and (cl in ctrl_map)
    for dn, cl in zip(drug_names, cell_line_ids)
])
valid_mask &= (cell_counts >= 10)
valid_idx = np.where(valid_mask)[0]

X_cg = np.array([cg_profiles[drug_name_map[drug_names[i]]] for i in valid_idx])
Y_expr = means[valid_idx]
Y_ctrl = np.array([ctrl_map[cell_line_ids[i]] for i in valid_idx])
Y_delta = Y_expr - Y_ctrl
valid_drugs = drug_names[valid_idx]
valid_cls = cell_line_ids[valid_idx]

n = len(valid_idx)
delta_var = np.var(Y_delta, axis=0)
n_hvg = min(2000, (delta_var > 1e-10).sum())
hvg_idx = np.argsort(-delta_var)[:n_hvg]
Y_delta_hvg = Y_delta[:, hvg_idx]

def eval_pearson_delta(pred, true):
    corrs = []
    for i in range(pred.shape[0]):
        if np.std(pred[i]) < 1e-8 or np.std(true[i]) < 1e-8:
            continue
        r, _ = pearsonr(pred[i], true[i])
        if not np.isnan(r):
            corrs.append(r)
    return np.mean(corrs) if corrs else float('nan'), corrs

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

def eval_direction_accuracy(pred, true):
    accs = []
    for i in range(pred.shape[0]):
        acc = (np.sign(true[i]) == np.sign(pred[i])).mean()
        accs.append(acc)
    return np.mean(accs), accs

def eval_discrimination(pred_means, true_means):
    from scipy.spatial.distance import cdist
    if pred_means.shape[0] == 0:
        return float('nan')
    dist = cdist(pred_means, true_means, metric='cosine')
    N = dist.shape[0]
    ranks = []
    for i in range(N):
        rank = np.sum(dist[i] < dist[i, i])
        ranks.append(rank)
    return 1.0 - 2.0 * (np.mean(ranks) / N)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

p(f"{'='*90}")
p(f"SUPPLEMENTARY: α=1.0 — {n} conditions, {n_hvg} HVGs")
p(f"{'='*90}")

# --- Data efficiency with α=1.0 ---
p("\nDATA EFFICIENCY: Ridge(α=1.0)")
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
        model = Ridge(alpha=1.0)
        model.fit(X_tr, Y_delta_hvg[sub])
        frac_preds[te] = model.predict(X_te)
    avg_n = int(np.mean(n_train_list))
    pd_mean, _ = eval_pearson_delta(frac_preds, Y_delta_hvg)
    deg50, _ = eval_top_deg(frac_preds, Y_delta_hvg, 50)
    da, _ = eval_direction_accuracy(frac_preds, Y_delta_hvg)
    disc = eval_discrimination(frac_preds, Y_delta_hvg)
    mse = np.mean((frac_preds - Y_delta_hvg) ** 2)
    p(f"  Ridge α=1 {frac*100:5.0f}% ({avg_n:4d} train) | Pearson Δ: {pd_mean:.4f} | "
      f"DEG-50: {deg50:.4f} | Dir Acc: {da:.4f} | Disc: {disc:.4f} | MSE: {mse:.6f}")

# --- Leave-one-drug-out with multiple alphas ---
p(f"\nLEAVE-ONE-DRUG-OUT CV")
unique_drugs = sorted(set(valid_drugs))
p(f"  {len(unique_drugs)} unique drugs")

for alpha in [1.0, 10.0]:
    lodo_preds = np.zeros_like(Y_delta_hvg)
    lodo_mask = np.zeros(n, dtype=bool)
    for drug in unique_drugs:
        te_mask = valid_drugs == drug
        tr_mask = ~te_mask
        if te_mask.sum() == 0 or tr_mask.sum() < 10:
            continue
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_cg[tr_mask])
        X_te = sc.transform(X_cg[te_mask])
        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_delta_hvg[tr_mask])
        lodo_preds[te_mask] = model.predict(X_te)
        lodo_mask |= te_mask

    if lodo_mask.sum() > 0:
        pd_mean, _ = eval_pearson_delta(lodo_preds[lodo_mask], Y_delta_hvg[lodo_mask])
        deg50, _ = eval_top_deg(lodo_preds[lodo_mask], Y_delta_hvg[lodo_mask], 50)
        da, _ = eval_direction_accuracy(lodo_preds[lodo_mask], Y_delta_hvg[lodo_mask])
        disc = eval_discrimination(lodo_preds[lodo_mask], Y_delta_hvg[lodo_mask])
        mse = np.mean((lodo_preds[lodo_mask] - Y_delta_hvg[lodo_mask]) ** 2)
        p(f"  LODO Ridge(α={alpha:5.1f}) | Pearson Δ: {pd_mean:.4f} | DEG-50: {deg50:.4f} | "
          f"Dir Acc: {da:.4f} | Disc: {disc:.4f} | MSE: {mse:.6f}")

# --- Per cell-line with α=1.0 ---
p(f"\nPER CELL-LINE (Ridge α=1.0)")
ridge_preds = np.zeros_like(Y_delta_hvg)
for tr, te in kf.split(X_cg):
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_cg[tr])
    X_te = sc.transform(X_cg[te])
    model = Ridge(alpha=1.0)
    model.fit(X_tr, Y_delta_hvg[tr])
    ridge_preds[te] = model.predict(X_te)

for cl in sorted(set(valid_cls)):
    cl_mask = valid_cls == cl
    if cl_mask.sum() < 10:
        continue
    pd_mean, _ = eval_pearson_delta(ridge_preds[cl_mask], Y_delta_hvg[cl_mask])
    deg50, _ = eval_top_deg(ridge_preds[cl_mask], Y_delta_hvg[cl_mask], 50)
    da, _ = eval_direction_accuracy(ridge_preds[cl_mask], Y_delta_hvg[cl_mask])
    p(f"  {cl:12s} (n={cl_mask.sum():3d}): Pearson Δ={pd_mean:.4f} DEG-50={deg50:.4f} DirAcc={da:.4f}")

p("\nDone.")
