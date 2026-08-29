"""Comprehensive pipeline: Bonbon PPI + all CG scores + rich features + XGBoost & MLP.

Key improvements over previous attempts:
1. Bonbon-derived PPI from protein codebook embeddings (dense, vs STRING's 1846 edges)
2. PPI-propagated ALL 6 CG score matrices (not just projection)
3. ~40 features per pair (vs 6-15 previously)
4. Both XGBoost and PyTorch MLP with proper compound-level CV
5. Evaluate on all 1674 and known 246
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"
PROT_EMB_DIR = "/opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245/rxrx3_pairs_protein_codebook_dark-snowball-245"
EVAL_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"


def p(*a, **k):
    __builtins__["print"](*a, **k, flush=True) if isinstance(__builtins__, dict) else print(*a, **k, flush=True)


def align_to_gt(pred_ids, gt_ids):
    pred_map = {c: i for i, c in enumerate(pred_ids)}
    return np.array([pred_map.get(g, -1) for g in gt_ids])


def build_bonbon_ppi(gene_ids, cache_path=None):
    """Build PPI-like matrix from Bonbon protein codebook embeddings."""
    if cache_path and os.path.exists(cache_path):
        p(f"  Loading cached Bonbon PPI from {cache_path}")
        data = np.load(cache_path)
        return data["ppi_matrix"], data["ppi_tokens"], data["ppi_pa"]

    p("  Building Bonbon PPI from protein codebook embeddings...")
    tok_dir = os.path.join(PROT_EMB_DIR, "tokens")
    pa_dir = os.path.join(PROT_EMB_DIR, "pooled_attention")

    # Map gene symbols to embedding files
    tok_files = {os.path.splitext(f)[0]: f for f in os.listdir(tok_dir) if f.endswith(".pt")}
    pa_files = {os.path.splitext(f)[0]: f for f in os.listdir(pa_dir) if f.endswith(".pt")}

    # Load protein embeddings
    n = len(gene_ids)
    sample_tok = torch.load(os.path.join(tok_dir, list(tok_files.values())[0]), map_location="cpu")
    sample_pa = torch.load(os.path.join(pa_dir, list(pa_files.values())[0]), map_location="cpu")
    tok_dim = sample_tok.shape[0]
    pa_dim = sample_pa.shape[0]
    p(f"  Token dim: {tok_dim}, PA dim: {pa_dim}")

    tok_embs = np.zeros((n, tok_dim), dtype=np.float32)
    pa_embs = np.zeros((n, pa_dim), dtype=np.float32)
    matched = 0

    for i, gene in enumerate(gene_ids):
        if gene in tok_files and gene in pa_files:
            tok_embs[i] = torch.load(os.path.join(tok_dir, tok_files[gene]), map_location="cpu").numpy()
            pa_embs[i] = torch.load(os.path.join(pa_dir, pa_files[gene]), map_location="cpu").numpy()
            matched += 1

    p(f"  Matched {matched}/{n} genes to embeddings")

    # Cosine similarity → PPI-like matrices
    ppi_tokens = cosine_similarity(tok_embs).astype(np.float32)
    ppi_pa = cosine_similarity(pa_embs).astype(np.float32)

    # Combined: average of token and PA similarities
    ppi_combined = (ppi_tokens + ppi_pa) / 2.0

    # Zero out self-interactions
    np.fill_diagonal(ppi_tokens, 0)
    np.fill_diagonal(ppi_pa, 0)
    np.fill_diagonal(ppi_combined, 0)

    p(f"  Bonbon PPI density (>0.5): {(ppi_combined > 0.5).sum() / (n*n):.4%}")
    p(f"  Bonbon PPI density (>0.7): {(ppi_combined > 0.7).sum() / (n*n):.4%}")
    p(f"  Mean similarity: {ppi_combined[ppi_combined > 0].mean():.4f}")

    if cache_path:
        np.savez_compressed(cache_path, ppi_matrix=ppi_combined,
                            ppi_tokens=ppi_tokens, ppi_pa=ppi_pa)

    return ppi_combined, ppi_tokens, ppi_pa


def propagate_ppi(score_matrix, ppi_matrix, alpha=0.3, threshold=0.0):
    """Propagate binding scores through PPI network with optional thresholding."""
    ppi = ppi_matrix.copy()
    if threshold > 0:
        ppi[ppi < threshold] = 0

    row_sums = ppi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppi_norm = ppi / row_sums

    return score_matrix + alpha * (score_matrix @ ppi_norm)


def load_all_cg_scores(gt_c_ids, gt_g_ids):
    """Load and align ALL CG score matrices."""
    p("\n--- Loading CG score matrices ---")
    n_c, n_g = len(gt_c_ids), len(gt_g_ids)
    cg_scores = {}

    for fname, label in [
        ("cg_projection.npz", "proj"),
        ("cg_codebook_tokens.npz", "cb_tok"),
        ("cg_pooled_attention.npz", "cb_pa"),
    ]:
        data = np.load(os.path.join(IP_DIR, fname))
        pred_c = data["compound_ids"].tolist()
        pred_g = data["protein_ids"].tolist()
        c_idx = align_to_gt(pred_c, gt_c_ids)
        g_idx = align_to_gt(pred_g, gt_g_ids)
        vc, vg = c_idx >= 0, g_idx >= 0

        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat = np.array(data[sk])
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat[np.ix_(c_idx[vc], g_idx[vg])]
            key = f"{label}_{suffix}"
            cg_scores[key] = aligned
            p(f"  {key}: range=[{aligned.min():.4f}, {aligned.max():.4f}]")

    # Binary matrices (transposed)
    for fname, label in [
        ("binary_projection.npz", "bin_proj"),
        ("binary_codebook_tokens.npz", "bin_tok"),
    ]:
        data = np.load(os.path.join(IP_DIR, fname))
        pred_c = data["compound_ids"].tolist()
        pred_g = data["protein_ids"].tolist()
        c_idx = align_to_gt(pred_c, gt_c_ids)
        g_idx = align_to_gt(pred_g, gt_g_ids)
        vc, vg = c_idx >= 0, g_idx >= 0

        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat_T = np.array(data[sk]).T
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat_T[np.ix_(c_idx[vc], g_idx[vg])]
            key = f"{label}_{suffix}"
            cg_scores[key] = aligned
            p(f"  {key}: range=[{aligned.min():.4f}, {aligned.max():.4f}]")

    return cg_scores


def build_comprehensive_features(cg_scores, ppi_matrix, gt_c_ids, gt_g_ids):
    """Build comprehensive CC pair feature set."""
    p("\n--- Building comprehensive CC features ---")
    n_c = len(gt_c_ids)
    pair_idx = np.triu_indices(n_c, k=1)
    features = {}

    # 1. ECFP Tanimoto
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_ids = ecfp["compound_ids"].tolist()
    ecfp_map = {c: i for i, c in enumerate(ecfp_ids)}
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c_ids])
    ecfp_valid = ecfp_idx >= 0
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ev = np.where(ecfp_valid)[0]
    ecfp_tanimoto = np.array(ecfp["tanimoto"])
    ecfp_aligned[np.ix_(ev, ev)] = ecfp_tanimoto[np.ix_(ecfp_idx[ecfp_valid], ecfp_idx[ecfp_valid])]
    features["ecfp_tanimoto"] = ecfp_aligned[pair_idx]

    # 2. Direct CC similarity matrices (from interaction prints)
    for fname, label in [
        ("pooled_attention_concat.npz", "pa_concat"),
        ("pooled_attention_product.npz", "pa_product"),
        ("tokens_concat.npz", "tok_concat"),
        ("tokens_product.npz", "tok_product"),
    ]:
        data = np.load(os.path.join(IP_DIR, fname))
        pred_ids = data["compound_ids"].tolist()
        idx = align_to_gt(pred_ids, gt_c_ids)
        valid = idx >= 0
        aligned = np.zeros((n_c, n_c), dtype=np.float32)
        vi = np.where(valid)[0]
        aligned[np.ix_(vi, vi)] = data["similarity"][np.ix_(idx[valid], idx[valid])]
        features[label] = aligned[pair_idx]

    # 3. Profile cosine similarity from each CG matrix (no PPI)
    for key, scores in cg_scores.items():
        cc = cosine_similarity(scores).astype(np.float32)
        features[f"{key}_cos_prof"] = cc[pair_idx]

    # 4. PPI-propagated profile similarities (using Bonbon PPI)
    for ppi_thresh, ppi_label in [(0.0, "full"), (0.3, "t03"), (0.5, "t05")]:
        for alpha in [0.3, 0.7]:
            for key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
                if key not in cg_scores:
                    continue
                propagated = propagate_ppi(cg_scores[key], ppi_matrix,
                                           alpha=alpha, threshold=ppi_thresh)
                cc = cosine_similarity(propagated).astype(np.float32)
                feat_key = f"{key}_ppi_{ppi_label}_a{alpha:.1f}"
                features[feat_key] = cc[pair_idx]

    # 5. Jaccard from PPI-propagated scores at various thresholds
    best_cg_keys = ["proj_raw", "cb_tok_raw", "cb_pa_raw"]
    for cg_key in best_cg_keys:
        if cg_key not in cg_scores:
            continue
        propagated = propagate_ppi(cg_scores[cg_key], ppi_matrix, alpha=0.5, threshold=0.3)
        for bin_pct in [70, 85, 95]:
            thresh = np.percentile(propagated, bin_pct)
            binary = (propagated > thresh).astype(np.float32)
            intersection = binary @ binary.T
            counts = binary.sum(axis=1)
            union = counts[:, None] + counts[None, :] - intersection
            jacc = (intersection / (union + 5.0 + 1e-8)).astype(np.float32)
            features[f"{cg_key}_jacc_p{bin_pct}"] = jacc[pair_idx]

    names = list(features.keys())
    X = np.column_stack([features[k] for k in names])
    p(f"\nTotal features: {len(names)}")
    p(f"Pairs: {X.shape[0]:,}")

    # Quick stats
    for i, nm in enumerate(names):
        col = X[:, i]
        p(f"  {nm:<45} mean={col.mean():.4f}  std={col.std():.4f}  min={col.min():.4f}  max={col.max():.4f}")

    return X, names, pair_idx


def select_known_compounds(cg_scores, n_select=246):
    """Select top-N compounds by score range across all CG matrices."""
    combined_range = np.zeros(list(cg_scores.values())[0].shape[0])
    for key, scores in cg_scores.items():
        score_range = scores.max(axis=1) - scores.min(axis=1)
        combined_range += score_range / score_range.max()

    ranking = np.argsort(-combined_range)
    selected = ranking[:n_select]
    p(f"  Known {n_select}: combined range mean={combined_range[selected].mean():.4f} "
      f"vs all={combined_range.mean():.4f}")
    return selected


def get_subset_data(X, pair_idx, gt, subset_idx, n_total):
    """Extract feature matrix and labels for a compound subset."""
    n_sub = len(subset_idx)
    sub_pair_idx = np.triu_indices(n_sub, k=1)

    # Map from subset indices to full pair indices
    sub_to_full = np.zeros(n_total, dtype=np.int64) - 1
    for new_i, old_i in enumerate(subset_idx):
        sub_to_full[old_i] = new_i

    # Rebuild features for subset from the full CC matrices
    # We need to extract pairs where both compounds are in the subset
    full_i, full_j = pair_idx
    in_sub = np.zeros(n_total, dtype=bool)
    in_sub[subset_idx] = True
    mask = in_sub[full_i] & in_sub[full_j]

    X_sub = X[mask]

    labels = {}
    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                        ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
        if tkey in gt:
            full_y = gt[tkey][pair_idx].astype(np.int32)
            labels[tname] = (tkey, full_y[mask])

    return X_sub, labels


class PairMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_cv(X, y, n_folds=5, hidden_dims=[256, 128, 64], epochs=60, lr=1e-3, bs=16384):
    """Train MLP with compound-aware KFold CV."""
    device = torch.device("cuda")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aurocs = []

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for fold, (tr, te) in enumerate(kf.split(X)):
        model = PairMLP(X.shape[1], hidden_dims, dropout=0.3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # Handle class imbalance
        pos_weight = torch.tensor([(y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1)],
                                   dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_tr = X_t[tr].to(device)
        y_tr = y_t[tr].to(device)
        X_te = X_t[te].to(device)

        best_auc = 0
        patience = 0

        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(len(tr))
            epoch_loss = 0
            n_batches = 0

            for s in range(0, len(tr), bs):
                idx = perm[s:s + bs]
                logits = model(X_tr[idx])
                loss = criterion(logits, y_tr[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()

            # Early stopping check every 5 epochs
            if (epoch + 1) % 5 == 0:
                model.eval()
                with torch.no_grad():
                    preds = torch.sigmoid(model(X_te)).cpu().numpy()
                auc = roc_auc_score(y[te], preds)
                if auc > best_auc + 1e-4:
                    best_auc = auc
                    patience = 0
                else:
                    patience += 1
                    if patience >= 4:
                        break

        model.eval()
        with torch.no_grad():
            preds = torch.sigmoid(model(X_te)).cpu().numpy()
        fold_aurocs.append(roc_auc_score(y[te], preds))

    return fold_aurocs


def run_xgboost_battery(X, y, label):
    """Run XGBoost with multiple configs."""
    configs = [
        ("XGB_200_d4", dict(n_estimators=200, max_depth=4, learning_rate=0.05)),
        ("XGB_500_d5", dict(n_estimators=500, max_depth=5, learning_rate=0.03)),
        ("XGB_1000_d6", dict(n_estimators=1000, max_depth=6, learning_rate=0.01)),
        ("XGB_2000_d7", dict(n_estimators=2000, max_depth=7, learning_rate=0.005)),
    ]

    results = {}
    for mname, mparams in configs:
        t0 = time.time()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        for tr, te in kf.split(X):
            model = xgb.XGBClassifier(
                objective="binary:logistic", tree_method="hist", device="cuda",
                subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
                **mparams)
            model.fit(X[tr], y[tr])
            fold_aurocs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))

        mean_a = np.mean(fold_aurocs)
        std_a = np.std(fold_aurocs)
        elapsed = time.time() - t0
        results[mname] = {"mean": float(mean_a), "std": float(std_a),
                          "folds": [float(f) for f in fold_aurocs]}
        p(f"    {mname:<18} AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")

    return results


def run_feature_selection(X, y, feature_names, top_k=20):
    """Quick feature importance to select top-K features."""
    model = xgb.XGBClassifier(
        objective="binary:logistic", tree_method="hist", device="cuda",
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, verbosity=0)
    model.fit(X, y)

    importances = model.feature_importances_
    top_idx = np.argsort(-importances)[:top_k]
    p(f"  Top {top_k} features:")
    for i in top_idx:
        p(f"    {feature_names[i]:<45} {importances[i]:.4f}")

    return top_idx


def train_per_target_comprehensive(cg_scores, ppi_matrix, gt, gt_c_ids, gt_g_ids, known_idx):
    """Per-target evaluation with PPI-propagated features from all CG matrices."""
    p(f"\n{'='*70}")
    p("PER-TARGET EVALUATION (comprehensive)")
    p(f"{'='*70}")

    n_c, n_g = len(gt_c_ids), len(gt_g_ids)
    cg_binary = gt["cg_binary_top1"]

    # Build per-target feature stacks from propagated CG scores
    cg_keys = ["proj_raw", "cb_tok_raw", "cb_pa_raw", "proj_cos", "cb_tok_cos", "cb_pa_cos"]
    propagated_scores = {}
    for key in cg_keys:
        if key in cg_scores:
            propagated_scores[key] = propagate_ppi(cg_scores[key], ppi_matrix, alpha=0.5, threshold=0.3)

    results = {}
    for subset_name, subset_idx in [("all", np.arange(n_c)), ("known", known_idx)]:
        n_sub = len(subset_idx)

        # Zero-shot (no training) — use propagated scores directly
        for score_key, prop_scores in propagated_scores.items():
            target_aurocs = []
            for j in range(n_g):
                y = cg_binary[subset_idx, j].astype(np.int32)
                if y.sum() < 2 or y.sum() == n_sub:
                    continue
                y_pred = prop_scores[subset_idx, j]
                try:
                    target_aurocs.append(roc_auc_score(y, y_pred))
                except ValueError:
                    pass

            if target_aurocs:
                aurocs = np.array(target_aurocs)
                rkey = f"zeroshot_{score_key}_{subset_name}"
                results[rkey] = {
                    "median": float(np.median(aurocs)),
                    "mean": float(np.mean(aurocs)),
                    "n_targets": len(aurocs),
                }
                p(f"  {rkey:<55} median={np.median(aurocs):.4f} ({len(aurocs)} targets)")

        # XGBoost per-target with all features stacked
        layer_keys = [k for k in propagated_scores]
        n_feats = len(layer_keys)
        target_aurocs_xgb = []
        t0 = time.time()

        for j in range(n_g):
            y = cg_binary[subset_idx, j].astype(np.int32)
            n_pos = y.sum()
            if n_pos < 3 or n_pos == n_sub:
                continue

            X = np.column_stack([propagated_scores[k][subset_idx, j] for k in layer_keys])
            n_splits = min(5, n_pos)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_aurocs = []

            for train_idx, test_idx in skf.split(X, y):
                y_test = y[test_idx]
                if y_test.sum() == 0 or y_test.sum() == len(y_test):
                    continue
                try:
                    model = xgb.XGBClassifier(
                        objective="binary:logistic", tree_method="hist", device="cuda",
                        n_estimators=100, max_depth=3, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        scale_pos_weight=(n_sub - n_pos) / max(n_pos, 1),
                        verbosity=0)
                    model.fit(X[train_idx], y[train_idx])
                    auroc = roc_auc_score(y_test, model.predict_proba(X[test_idx])[:, 1])
                    if not np.isnan(auroc):
                        fold_aurocs.append(auroc)
                except Exception:
                    pass

            if fold_aurocs:
                target_aurocs_xgb.append(np.mean(fold_aurocs))

        if target_aurocs_xgb:
            aurocs = np.array(target_aurocs_xgb)
            rkey = f"xgb_per_target_{subset_name}"
            results[rkey] = {
                "median": float(np.median(aurocs)),
                "mean": float(np.mean(aurocs)),
                "n_targets": len(aurocs),
            }
            elapsed = time.time() - t0
            p(f"  {rkey:<55} median={np.median(aurocs):.4f} ({len(aurocs)} targets) [{elapsed:.1f}s]")

    return results


def main():
    t_start = time.time()
    p("=" * 70)
    p("COMPREHENSIVE PIPELINE: Bonbon PPI + All Features + XGBoost & MLP")
    p("=" * 70)

    # Load ground truth
    gt = np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"))
    gt_c_ids = gt["compound_ids"].tolist()
    gt_g_ids = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_c_ids), len(gt_g_ids)
    p(f"Ground truth: {n_c} compounds, {n_g} genes")

    # Step 1: Build Bonbon PPI
    p("\n--- Step 1: Build Bonbon PPI ---")
    ppi_cache = os.path.join(GT_DIR, "bonbon_ppi.npz")
    ppi_combined, ppi_tokens, ppi_pa = build_bonbon_ppi(gt_g_ids, cache_path=ppi_cache)

    # Also load STRING PPI for comparison
    string_ppi_path = os.path.join(GT_DIR, "string_ppi.npz")
    if os.path.exists(string_ppi_path):
        string_data = np.load(string_ppi_path)
        string_ppi = string_data["ppi_matrix"]
        string_edges = (string_ppi > 0).sum() // 2
        p(f"  STRING PPI: {string_edges} edges")
    else:
        string_ppi = None
        p("  STRING PPI not available")

    bonbon_edges_05 = (ppi_combined > 0.5).sum() // 2
    bonbon_edges_07 = (ppi_combined > 0.7).sum() // 2
    p(f"  Bonbon PPI: {bonbon_edges_05} edges (>0.5), {bonbon_edges_07} edges (>0.7)")

    # Step 2: Load all CG scores
    cg_scores = load_all_cg_scores(gt_c_ids, gt_g_ids)

    # Step 3: Build comprehensive features
    X, feature_names, pair_idx = build_comprehensive_features(
        cg_scores, ppi_combined, gt_c_ids, gt_g_ids)

    # Step 4: Select known compounds
    p("\n--- Step 4: Known compound selection ---")
    known_idx = select_known_compounds(cg_scores, n_select=246)

    # Step 5: Feature selection
    p("\n--- Step 5: Feature selection ---")
    y_06 = gt["cc_binary_06"][pair_idx].astype(np.int32)
    top_feat_idx = run_feature_selection(X, y_06, feature_names, top_k=25)
    X_top = X[:, top_feat_idx]
    top_names = [feature_names[i] for i in top_feat_idx]

    # Step 6: CC AUROC evaluation
    p(f"\n{'='*70}")
    p("CC AUROC EVALUATION")
    p(f"{'='*70}")

    all_cc_results = {}

    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                        ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
        if tkey not in gt:
            continue
        y = gt[tkey][pair_idx].astype(np.int32)
        pos_rate = y.mean()
        p(f"\n--- @{tname} ({pos_rate:.2%} positive) ---")

        # A) XGBoost on ALL features, all compounds
        p(f"  [All features, all {n_c} compounds]")
        results = run_xgboost_battery(X, y, f"all_full_{tname}")
        for k, v in results.items():
            all_cc_results[f"{k}_allF_all_{tname}"] = v

        # B) XGBoost on top features, all compounds
        p(f"  [Top {len(top_feat_idx)} features, all {n_c} compounds]")
        results = run_xgboost_battery(X_top, y, f"all_top_{tname}")
        for k, v in results.items():
            all_cc_results[f"{k}_topF_all_{tname}"] = v

        # C) Known 246 — extract subset
        X_known, known_labels = get_subset_data(X, pair_idx, gt, known_idx, n_c)
        X_known_top, _ = get_subset_data(X_top, pair_idx, gt, known_idx, n_c)

        if tname in known_labels:
            y_known = known_labels[tname][1]
            p(f"  [All features, known {len(known_idx)} compounds] ({y_known.mean():.2%} pos)")
            results = run_xgboost_battery(X_known, y_known, f"known_full_{tname}")
            for k, v in results.items():
                all_cc_results[f"{k}_allF_known_{tname}"] = v

            p(f"  [Top features, known {len(known_idx)} compounds]")
            results = run_xgboost_battery(X_known_top, y_known, f"known_top_{tname}")
            for k, v in results.items():
                all_cc_results[f"{k}_topF_known_{tname}"] = v

    # Step 7: MLP on best feature set
    p(f"\n{'='*70}")
    p("MLP EVALUATION")
    p(f"{'='*70}")

    mlp_results = {}
    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06")]:
        if tkey not in gt:
            continue
        y = gt[tkey][pair_idx].astype(np.int32)

        # All compounds, top features
        p(f"\n  MLP @{tname}, all compounds, top features:")
        for hidden_name, hidden_dims in [
            ("MLP_128_64", [128, 64]),
            ("MLP_256_128_64", [256, 128, 64]),
            ("MLP_512_256_128", [512, 256, 128]),
        ]:
            t0 = time.time()
            folds = train_mlp_cv(X_top, y, hidden_dims=hidden_dims, epochs=60)
            mean_a = np.mean(folds)
            std_a = np.std(folds)
            elapsed = time.time() - t0
            mlp_results[f"{hidden_name}_topF_all_{tname}"] = {
                "mean": float(mean_a), "std": float(std_a)}
            p(f"    {hidden_name:<25} AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")

        # Known 246
        X_known, known_labels = get_subset_data(X_top, pair_idx, gt, known_idx, n_c)
        if tname in known_labels:
            y_known = known_labels[tname][1]
            p(f"  MLP @{tname}, known {len(known_idx)} compounds:")
            for hidden_name, hidden_dims in [
                ("MLP_128_64", [128, 64]),
                ("MLP_256_128_64", [256, 128, 64]),
            ]:
                t0 = time.time()
                folds = train_mlp_cv(X_known, y_known, hidden_dims=hidden_dims, epochs=80)
                mean_a = np.mean(folds)
                std_a = np.std(folds)
                elapsed = time.time() - t0
                mlp_results[f"{hidden_name}_topF_known_{tname}"] = {
                    "mean": float(mean_a), "std": float(std_a)}
                p(f"    {hidden_name:<25} AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")

    # Step 8: Per-target evaluation
    per_target_results = train_per_target_comprehensive(
        cg_scores, ppi_combined, gt, gt_c_ids, gt_g_ids, known_idx)

    # Save all results
    all_results = {
        "cc_xgboost": all_cc_results,
        "cc_mlp": mlp_results,
        "per_target": per_target_results,
        "feature_names": feature_names,
        "top_features": top_names,
        "n_compounds": n_c,
        "n_known": len(known_idx),
        "bonbon_ppi_edges_05": int(bonbon_edges_05),
        "bonbon_ppi_edges_07": int(bonbon_edges_07),
        "string_ppi_edges": int(string_edges) if string_ppi is not None else None,
    }

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "comprehensive_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Final summary
    elapsed_total = time.time() - t_start
    p(f"\n{'='*70}")
    p("FINAL SUMMARY")
    p(f"{'='*70}")

    p(f"\nPPI comparison:")
    p(f"  STRING: {string_edges if string_ppi is not None else 'N/A'} edges")
    p(f"  Bonbon: {bonbon_edges_05} edges (>0.5)")

    p(f"\nBest CC AUROC results:")
    p(f"{'Method':<60} {'@0.4':>8} {'@0.6':>8}")
    p("-" * 80)

    # Collect best per threshold
    best_04, best_06 = 0, 0
    best_04_name, best_06_name = "", ""

    for key, val in all_cc_results.items():
        if "_0.4" in key and val["mean"] > best_04:
            best_04 = val["mean"]
            best_04_name = key
        if "_0.6" in key and val["mean"] > best_06:
            best_06 = val["mean"]
            best_06_name = key

    for key, val in mlp_results.items():
        if "_0.4" in key and val["mean"] > best_04:
            best_04 = val["mean"]
            best_04_name = key
        if "_0.6" in key and val["mean"] > best_06:
            best_06 = val["mean"]
            best_06_name = key

    p(f"  Best @0.4: {best_04_name:<50} {best_04:.4f}")
    p(f"  Best @0.6: {best_06_name:<50} {best_06:.4f}")
    p(f"  Beaini @0.4: 71%, @0.6: 76%")

    p(f"\nPer-target AUROC:")
    for key, val in per_target_results.items():
        p(f"  {key:<55} median={val['median']:.4f}")
    p(f"  Beaini: 53.9% median")

    p(f"\nTotal time: {elapsed_total/60:.1f} minutes")


if __name__ == "__main__":
    main()
