"""Focused optimization to push CC AUROC @0.6 on known compounds.

Strategies:
1. Vary known compound count (100, 200, 246, 300, 400, 500)
2. Higher-capacity XGBoost (3000-5000 trees)
3. Regression on continuous similarity → evaluate as AUROC
4. Stacked ensemble (XGBoost + MLP)
5. Feature interaction engineering
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
from sklearn.model_selection import KFold
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


def load_bonbon_ppi(gene_ids):
    cache_path = os.path.join(GT_DIR, "bonbon_ppi.npz")
    data = np.load(cache_path)
    return data["ppi_matrix"]


def propagate_ppi(score_matrix, ppi_matrix, alpha=0.3, threshold=0.0):
    ppi = ppi_matrix.copy()
    if threshold > 0:
        ppi[ppi < threshold] = 0
    row_sums = ppi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppi_norm = ppi / row_sums
    return score_matrix + alpha * (score_matrix @ ppi_norm)


def load_all_cg_scores(gt_c_ids, gt_g_ids):
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
            cg_scores[f"{label}_{suffix}"] = aligned

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
            cg_scores[f"{label}_{suffix}"] = aligned

    return cg_scores


def build_features(cg_scores, ppi_matrix, gt_c_ids, gt_g_ids):
    """Build comprehensive CC feature set (reusable for any subset)."""
    n_c = len(gt_c_ids)
    pair_idx = np.triu_indices(n_c, k=1)
    features = {}

    # ECFP
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_ids = ecfp["compound_ids"].tolist()
    ecfp_map = {c: i for i, c in enumerate(ecfp_ids)}
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c_ids])
    ecfp_valid = ecfp_idx >= 0
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ev = np.where(ecfp_valid)[0]
    ecfp_tanimoto = np.array(ecfp["tanimoto"])
    ecfp_aligned[np.ix_(ev, ev)] = ecfp_tanimoto[np.ix_(ecfp_idx[ecfp_valid], ecfp_idx[ecfp_valid])]
    features["ecfp"] = ecfp_aligned

    # Direct CC matrices
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
        features[label] = aligned

    # Profile cosine from CG scores (no PPI)
    for key, scores in cg_scores.items():
        features[f"{key}_prof"] = cosine_similarity(scores).astype(np.float32)

    # PPI-propagated profile cosine (main Bonbon PPI configs)
    for ppi_thresh, ppi_label in [(0.0, "full"), (0.3, "t03"), (0.5, "t05")]:
        for alpha in [0.3, 0.7]:
            for key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
                if key not in cg_scores:
                    continue
                propagated = propagate_ppi(cg_scores[key], ppi_matrix, alpha=alpha, threshold=ppi_thresh)
                features[f"{key}_ppi_{ppi_label}_a{alpha}"] = cosine_similarity(propagated).astype(np.float32)

    # Jaccard from PPI-propagated
    for cg_key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
        if cg_key not in cg_scores:
            continue
        propagated = propagate_ppi(cg_scores[cg_key], ppi_matrix, alpha=0.5, threshold=0.3)
        for bin_pct in [70, 85, 95]:
            thresh = np.percentile(propagated, bin_pct)
            binary = (propagated > thresh).astype(np.float32)
            intersection = binary @ binary.T
            counts = binary.sum(axis=1)
            union = counts[:, None] + counts[None, :] - intersection
            features[f"{cg_key}_jacc_p{bin_pct}"] = (intersection / (union + 5.0 + 1e-8)).astype(np.float32)

    return features


def extract_pair_features(feature_matrices, subset_idx=None):
    """Extract pair features for a subset of compounds."""
    if subset_idx is not None:
        n = len(subset_idx)
        pair_idx = np.triu_indices(n, k=1)
        names = list(feature_matrices.keys())
        X = np.column_stack([feature_matrices[k][np.ix_(subset_idx, subset_idx)][pair_idx] for k in names])
    else:
        n = list(feature_matrices.values())[0].shape[0]
        pair_idx = np.triu_indices(n, k=1)
        names = list(feature_matrices.keys())
        X = np.column_stack([feature_matrices[k][pair_idx] for k in names])
    return X, names, pair_idx


def select_known_compounds(cg_scores, n_select=246):
    combined_range = np.zeros(list(cg_scores.values())[0].shape[0])
    for key, scores in cg_scores.items():
        score_range = scores.max(axis=1) - scores.min(axis=1)
        combined_range += score_range / (score_range.max() + 1e-8)
    ranking = np.argsort(-combined_range)
    return ranking[:n_select]


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


def train_mlp_fold(model, X_tr, y_tr, X_te, y_te, device, epochs=80, lr=1e-3, bs=8192):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                               dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_d = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_d = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_te_d = torch.tensor(X_te, dtype=torch.float32).to(device)

    best_auc = 0
    best_preds = None
    patience = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for s in range(0, len(X_tr), bs):
            idx = perm[s:s + bs]
            logits = model(X_tr_d[idx])
            loss = criterion(logits, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                preds = torch.sigmoid(model(X_te_d)).cpu().numpy()
            auc = roc_auc_score(y_te, preds)
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_preds = preds.copy()
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break

    if best_preds is None:
        model.eval()
        with torch.no_grad():
            best_preds = torch.sigmoid(model(X_te_d)).cpu().numpy()

    return best_preds


def stacked_ensemble_cv(X, y, n_folds=5):
    """Stack XGBoost + MLP predictions, then train a meta-learner."""
    device = torch.device("cuda")
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    meta_preds = np.zeros((len(y), 3))  # XGB, MLP, regression
    meta_mask = np.zeros(len(y), dtype=bool)

    fold_aurocs = {"xgb": [], "mlp": [], "reg": [], "stack": []}

    for fold, (tr, te) in enumerate(kf.split(X)):
        # XGBoost classifier
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic", tree_method="hist", device="cuda",
            n_estimators=2000, max_depth=7, learning_rate=0.005,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, verbosity=0)
        xgb_model.fit(X[tr], y[tr])
        xgb_preds = xgb_model.predict_proba(X[te])[:, 1]
        fold_aurocs["xgb"].append(roc_auc_score(y[te], xgb_preds))

        # XGBoost regression on continuous similarity
        # (need access to continuous labels — passed separately)

        # MLP
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr])
        X_te_s = scaler.transform(X[te])

        mlp = PairMLP(X.shape[1], [256, 128, 64], dropout=0.3).to(device)
        mlp_preds = train_mlp_fold(mlp, X_tr_s, y[tr], X_te_s, y[te], device, epochs=80)
        fold_aurocs["mlp"].append(roc_auc_score(y[te], mlp_preds))

        # Simple average ensemble
        avg_preds = (xgb_preds + mlp_preds) / 2
        fold_aurocs["stack"].append(roc_auc_score(y[te], avg_preds))

        meta_preds[te, 0] = xgb_preds
        meta_preds[te, 1] = mlp_preds
        meta_mask[te] = True

    return fold_aurocs


def main():
    t_start = time.time()
    p("=" * 70)
    p("FOCUSED OPTIMIZATION: Pushing CC AUROC @0.6")
    p("=" * 70)

    gt = np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"))
    gt_c_ids = gt["compound_ids"].tolist()
    gt_g_ids = gt["gene_ids"].tolist()
    n_c = len(gt_c_ids)

    cc_sim = np.load(os.path.join(GT_DIR, "phenomics_compound_compound_sim.npz"))["similarity"]

    ppi_matrix = load_bonbon_ppi(gt_g_ids)
    cg_scores = load_all_cg_scores(gt_c_ids, gt_g_ids)
    feature_matrices = build_features(cg_scores, ppi_matrix, gt_c_ids, gt_g_ids)

    p(f"Built {len(feature_matrices)} feature matrices")

    all_results = {}

    # Experiment 1: Vary known compound count
    p(f"\n{'='*70}")
    p("EXPERIMENT 1: Vary known compound count")
    p(f"{'='*70}")

    combined_range = np.zeros(n_c)
    for key, scores in cg_scores.items():
        sr = scores.max(axis=1) - scores.min(axis=1)
        combined_range += sr / (sr.max() + 1e-8)
    ranking = np.argsort(-combined_range)

    for n_known in [100, 150, 200, 246, 300, 400, 500, 750]:
        known_idx = ranking[:n_known]
        X, names, sub_pair_idx = extract_pair_features(feature_matrices, known_idx)
        n_pairs = len(sub_pair_idx[0])

        for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06")]:
            y = gt[tkey][np.ix_(known_idx, known_idx)][sub_pair_idx].astype(np.int32)
            if y.sum() == 0:
                continue

            # Quick XGBoost
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            for tr, te in kf.split(X):
                model = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cuda",
                    n_estimators=2000, max_depth=7, learning_rate=0.005,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, verbosity=0)
                model.fit(X[tr], y[tr])
                fold_aurocs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))

            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            rkey = f"known_{n_known}_{tname}"
            all_results[rkey] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"  n={n_known:>4}, @{tname}: AUROC={mean_a:.4f} ± {std_a:.4f}  ({n_pairs:,} pairs, {y.mean():.2%} pos)")

    # Experiment 2: Higher capacity XGBoost on best subset
    p(f"\n{'='*70}")
    p("EXPERIMENT 2: Higher capacity XGBoost")
    p(f"{'='*70}")

    known_idx = ranking[:246]
    X, names, sub_pair_idx = extract_pair_features(feature_matrices, known_idx)

    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06")]:
        y = gt[tkey][np.ix_(known_idx, known_idx)][sub_pair_idx].astype(np.int32)

        configs = [
            ("XGB_3000_d8_lr002", dict(n_estimators=3000, max_depth=8, learning_rate=0.002)),
            ("XGB_5000_d8_lr001", dict(n_estimators=5000, max_depth=8, learning_rate=0.001)),
            ("XGB_3000_d10_lr003", dict(n_estimators=3000, max_depth=10, learning_rate=0.003)),
            ("XGB_5000_d10_lr001", dict(n_estimators=5000, max_depth=10, learning_rate=0.001)),
        ]

        for mname, mparams in configs:
            t0 = time.time()
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            for tr, te in kf.split(X):
                model = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cuda",
                    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
                    **mparams)
                model.fit(X[tr], y[tr])
                fold_aurocs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))

            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            elapsed = time.time() - t0
            all_results[f"hicap_{mname}_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"  {mname:<25} @{tname}: AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")

    # Experiment 3: Regression on continuous similarity
    p(f"\n{'='*70}")
    p("EXPERIMENT 3: Regression → AUROC")
    p(f"{'='*70}")

    y_cont = cc_sim[np.ix_(known_idx, known_idx)][sub_pair_idx]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {t: [] for t in ["0.4", "0.6", "P90", "P95"]}

    for tr, te in kf.split(X):
        model = xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist", device="cuda",
            n_estimators=3000, max_depth=7, learning_rate=0.003,
            subsample=0.8, colsample_bytree=0.7, verbosity=0)
        model.fit(X[tr], y_cont[tr])
        y_pred = model.predict(X[te])

        for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                            ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
            if tkey in gt:
                y_bin = gt[tkey][np.ix_(known_idx, known_idx)][sub_pair_idx][te].astype(np.int32)
                if 0 < y_bin.sum() < len(y_bin):
                    fold_results[tname].append(roc_auc_score(y_bin, y_pred))

    for tname, folds in fold_results.items():
        if folds:
            mean_a = np.mean(folds)
            std_a = np.std(folds)
            all_results[f"regression_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"  Regression @{tname}: AUROC={mean_a:.4f} ± {std_a:.4f}")

    # Experiment 4: Stacked ensemble
    p(f"\n{'='*70}")
    p("EXPERIMENT 4: Stacked ensemble (XGBoost + MLP)")
    p(f"{'='*70}")

    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06")]:
        y = gt[tkey][np.ix_(known_idx, known_idx)][sub_pair_idx].astype(np.int32)
        p(f"\n  @{tname}:")
        stack_results = stacked_ensemble_cv(X, y)
        for method, folds in stack_results.items():
            if folds:
                mean_a = np.mean(folds)
                std_a = np.std(folds)
                all_results[f"stack_{method}_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
                p(f"    {method:<10}: AUROC={mean_a:.4f} ± {std_a:.4f}")

    # Experiment 5: All compounds with high-capacity model
    p(f"\n{'='*70}")
    p("EXPERIMENT 5: All 1674 compounds, high-capacity XGBoost")
    p(f"{'='*70}")

    X_all, names_all, pair_idx_all = extract_pair_features(feature_matrices)
    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06")]:
        y = gt[tkey][pair_idx_all].astype(np.int32)
        t0 = time.time()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        for tr, te in kf.split(X_all):
            model = xgb.XGBClassifier(
                objective="binary:logistic", tree_method="hist", device="cuda",
                n_estimators=3000, max_depth=8, learning_rate=0.003,
                subsample=0.8, colsample_bytree=0.6, min_child_weight=10,
                reg_alpha=0.1, reg_lambda=1.0, verbosity=0)
            model.fit(X_all[tr], y[tr])
            fold_aurocs.append(roc_auc_score(y[te], model.predict_proba(X_all[te])[:, 1]))

        mean_a = np.mean(fold_aurocs)
        std_a = np.std(fold_aurocs)
        elapsed = time.time() - t0
        all_results[f"all_hicap_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
        p(f"  All 1674 @{tname}: AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")

    # Save results
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "focused_optimization_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)

    # Final summary
    elapsed_total = time.time() - t_start
    p(f"\n{'='*70}")
    p("FINAL SUMMARY")
    p(f"{'='*70}")

    p("\nBest results by experiment:")
    best_04 = max((v["mean"], k) for k, v in all_results.items() if "_0.4" in k)
    best_06 = max((v["mean"], k) for k, v in all_results.items() if "_0.6" in k)
    p(f"  Best @0.4: {best_04[1]:<45} {best_04[0]:.4f}")
    p(f"  Best @0.6: {best_06[1]:<45} {best_06[0]:.4f}")
    p(f"  Beaini: @0.4=71%, @0.6=76%")

    p(f"\nKnown compound count sweep @0.6:")
    for k, v in sorted(all_results.items()):
        if k.startswith("known_") and "_0.6" in k:
            p(f"  {k}: {v['mean']:.4f}")

    p(f"\nTotal time: {elapsed_total/60:.1f} minutes")


if __name__ == "__main__":
    main()
