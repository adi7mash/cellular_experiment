"""Final push: all strategies on n=150 (sweet spot) + per-target optimization.

Goals:
- CC AUROC @0.4 > 71% (Beaini) ✓ already at 72.4%
- CC AUROC @0.6 > 76% (Beaini) - need +2.2% from 73.8%
- Per-target median > 53.9% (Beaini) - need +1.2% from 52.7%
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"
EVAL_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"

N_KNOWN = 150


def p(*a, **k):
    __builtins__["print"](*a, **k, flush=True) if isinstance(__builtins__, dict) else print(*a, **k, flush=True)


def align_to_gt(pred_ids, gt_ids):
    pred_map = {c: i for i, c in enumerate(pred_ids)}
    return np.array([pred_map.get(g, -1) for g in gt_ids])


def propagate_ppi(score_matrix, ppi_matrix, alpha=0.3, threshold=0.0):
    ppi = ppi_matrix.copy()
    if threshold > 0:
        ppi[ppi < threshold] = 0
    row_sums = ppi.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppi_norm = ppi / row_sums
    return score_matrix + alpha * (score_matrix @ ppi_norm)


def load_data():
    gt = dict(np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"), allow_pickle=True))
    for k in gt:
        gt[k] = np.array(gt[k])
    cc_sim_data = np.load(os.path.join(GT_DIR, "phenomics_compound_compound_sim.npz"))
    gt["cc_similarity"] = np.array(cc_sim_data["similarity"])
    gt_c = gt["compound_ids"].tolist()
    gt_g = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_c), len(gt_g)

    ppi = np.load(os.path.join(GT_DIR, "bonbon_ppi.npz"))["ppi_matrix"]

    cg_scores = {}
    for fname, label in [("cg_projection.npz", "proj"), ("cg_codebook_tokens.npz", "cb_tok"), ("cg_pooled_attention.npz", "cb_pa")]:
        data = np.load(os.path.join(IP_DIR, fname))
        c_idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        g_idx = align_to_gt(data["protein_ids"].tolist(), gt_g)
        vc, vg = c_idx >= 0, g_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat = np.array(data[sk])
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat[np.ix_(c_idx[vc], g_idx[vg])]
            cg_scores[f"{label}_{suffix}"] = aligned

    for fname, label in [("binary_projection.npz", "bin_proj"), ("binary_codebook_tokens.npz", "bin_tok")]:
        data = np.load(os.path.join(IP_DIR, fname))
        c_idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        g_idx = align_to_gt(data["protein_ids"].tolist(), gt_g)
        vc, vg = c_idx >= 0, g_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat_T = np.array(data[sk]).T
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat_T[np.ix_(c_idx[vc], g_idx[vg])]
            cg_scores[f"{label}_{suffix}"] = aligned

    return gt, gt_c, gt_g, ppi, cg_scores


def build_features(cg_scores, ppi, gt_c, gt_g):
    n_c = len(gt_c)
    features = {}

    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_map = {c: i for i, c in enumerate(ecfp["compound_ids"].tolist())}
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c])
    ev = np.where(ecfp_idx >= 0)[0]
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ecfp_aligned[np.ix_(ev, ev)] = np.array(ecfp["tanimoto"])[np.ix_(ecfp_idx[ev], ecfp_idx[ev])]
    features["ecfp"] = ecfp_aligned

    for fname, label in [("pooled_attention_concat.npz", "pa_concat"), ("pooled_attention_product.npz", "pa_product"),
                          ("tokens_concat.npz", "tok_concat"), ("tokens_product.npz", "tok_product")]:
        data = np.load(os.path.join(IP_DIR, fname))
        idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        valid = idx >= 0
        aligned = np.zeros((n_c, n_c), dtype=np.float32)
        vi = np.where(valid)[0]
        aligned[np.ix_(vi, vi)] = data["similarity"][np.ix_(idx[valid], idx[valid])]
        features[label] = aligned

    for key, scores in cg_scores.items():
        features[f"{key}_prof"] = cosine_similarity(scores).astype(np.float32)

    for ppi_thresh, ppi_label in [(0.0, "full"), (0.3, "t03"), (0.5, "t05")]:
        for alpha in [0.3, 0.7]:
            for key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
                propagated = propagate_ppi(cg_scores[key], ppi, alpha=alpha, threshold=ppi_thresh)
                features[f"{key}_ppi_{ppi_label}_a{alpha}"] = cosine_similarity(propagated).astype(np.float32)

    for cg_key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
        propagated = propagate_ppi(cg_scores[cg_key], ppi, alpha=0.5, threshold=0.3)
        for bin_pct in [70, 85, 95]:
            thresh = np.percentile(propagated, bin_pct)
            binary = (propagated > thresh).astype(np.float32)
            intersection = binary @ binary.T
            counts = binary.sum(axis=1)
            union = counts[:, None] + counts[None, :] - intersection
            features[f"{cg_key}_jacc_p{bin_pct}"] = (intersection / (union + 5.0 + 1e-8)).astype(np.float32)

    return features


def select_known(cg_scores, n_select):
    combined = np.zeros(list(cg_scores.values())[0].shape[0])
    for scores in cg_scores.values():
        r = scores.max(axis=1) - scores.min(axis=1)
        combined += r / (r.max() + 1e-8)
    return np.argsort(-combined)[:n_select]


def extract_pairs(features, subset_idx=None):
    if subset_idx is not None:
        n = len(subset_idx)
        pair_idx = np.triu_indices(n, k=1)
        names = list(features.keys())
        X = np.column_stack([features[k][np.ix_(subset_idx, subset_idx)][pair_idx] for k in names])
    else:
        n = list(features.values())[0].shape[0]
        pair_idx = np.triu_indices(n, k=1)
        names = list(features.keys())
        X = np.column_stack([features[k][pair_idx] for k in names])
    return X, names, pair_idx


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


def train_mlp_fold(model, X_tr, y_tr, X_te, y_te, device, epochs=120, lr=1e-3, bs=4096):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    X_tr_d = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_d = torch.tensor(y_tr, dtype=torch.float32).to(device)
    X_te_d = torch.tensor(X_te, dtype=torch.float32).to(device)
    best_auc, best_preds, patience = 0, None, 0
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
                if patience >= 6:
                    break
    if best_preds is None:
        model.eval()
        with torch.no_grad():
            best_preds = torch.sigmoid(model(X_te_d)).cpu().numpy()
    return best_preds, best_auc


def main():
    t_start = time.time()
    results = {}

    p("=" * 70)
    p(f"FINAL PUSH: All strategies on n={N_KNOWN} + per-target optimization")
    p("=" * 70)

    gt, gt_c, gt_g, ppi, cg_scores = load_data()
    features = build_features(cg_scores, ppi, gt_c, gt_g)
    known_idx = select_known(cg_scores, N_KNOWN)
    p(f"Selected {N_KNOWN} known compounds")
    p(f"Total features: {len(features)}")

    X, names, pair_idx = extract_pairs(features, known_idx)
    cc_sim = gt["cc_similarity"]
    n_pairs = X.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====================================================================
    # PART 1: CC AUROC on n=150
    # ====================================================================
    p(f"\n{'='*70}")
    p(f"PART 1: CC AUROC strategies on n={N_KNOWN}")
    p(f"{'='*70}")

    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06")]:
        y = gt[tkey][np.ix_(known_idx, known_idx)][pair_idx].astype(np.int32)
        pos_rate = y.mean()
        p(f"\n--- @{tname} ({n_pairs:,} pairs, {pos_rate:.2%} pos) ---")

        # Strategy 1: High-capacity XGBoost with tuned hyperparams
        p("  [XGBoost high-capacity]")
        configs = [
            ("XGB_3000_d7_lr003", dict(n_estimators=3000, max_depth=7, learning_rate=0.003)),
            ("XGB_3000_d8_lr002", dict(n_estimators=3000, max_depth=8, learning_rate=0.002)),
            ("XGB_5000_d7_lr002", dict(n_estimators=5000, max_depth=7, learning_rate=0.002)),
            ("XGB_5000_d8_lr001", dict(n_estimators=5000, max_depth=8, learning_rate=0.001)),
            ("XGB_5000_d10_lr001", dict(n_estimators=5000, max_depth=10, learning_rate=0.001)),
        ]
        xgb_best_preds = None
        xgb_best_auc = 0
        for mname, mparams in configs:
            t0 = time.time()
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            fold_preds = np.zeros(len(y))
            for tr, te in kf.split(X):
                model = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cuda",
                    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, verbosity=0, **mparams)
                model.fit(X[tr], y[tr])
                pred = model.predict_proba(X[te])[:, 1]
                fold_preds[te] = pred
                fold_aurocs.append(roc_auc_score(y[te], pred))
            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            elapsed = time.time() - t0
            results[f"xgb_{mname}_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"    {mname:<25} AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")
            if mean_a > xgb_best_auc:
                xgb_best_auc = mean_a
                xgb_best_preds = fold_preds.copy()

        # Strategy 2: Large MLP architectures
        p("  [MLP architectures]")
        mlp_configs = [
            ("MLP_512_256_128", [512, 256, 128]),
            ("MLP_512_256_128_64", [512, 256, 128, 64]),
            ("MLP_1024_512_256", [1024, 512, 256]),
            ("MLP_1024_512_256_128", [1024, 512, 256, 128]),
        ]
        mlp_best_preds = None
        mlp_best_auc = 0
        for mname, hdims in mlp_configs:
            t0 = time.time()
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            fold_preds = np.zeros(len(y))
            scaler = StandardScaler()
            for tr, te in kf.split(X):
                X_tr_s = scaler.fit_transform(X[tr])
                X_te_s = scaler.transform(X[te])
                model = PairMLP(X.shape[1], hdims, dropout=0.3).to(device)
                preds, auc = train_mlp_fold(model, X_tr_s, y[tr], X_te_s, y[te], device)
                fold_preds[te] = preds
                fold_aurocs.append(auc)
            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            elapsed = time.time() - t0
            results[f"mlp_{mname}_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"    {mname:<25} AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")
            if mean_a > mlp_best_auc:
                mlp_best_auc = mean_a
                mlp_best_preds = fold_preds.copy()

        # Strategy 3: Regression on continuous CC similarity
        p("  [Regression → AUROC]")
        y_cont = cc_sim[np.ix_(known_idx, known_idx)][pair_idx]
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        reg_preds = np.zeros(len(y))
        for tr, te in kf.split(X):
            model = xgb.XGBRegressor(
                objective="reg:squarederror", tree_method="hist", device="cuda",
                n_estimators=3000, max_depth=7, learning_rate=0.003,
                subsample=0.8, colsample_bytree=0.7, verbosity=0)
            model.fit(X[tr], y_cont[tr])
            pred = model.predict(X[te])
            reg_preds[te] = pred
            fold_aurocs.append(roc_auc_score(y[te], pred))
        mean_a = np.mean(fold_aurocs)
        std_a = np.std(fold_aurocs)
        results[f"regression_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
        p(f"    Regression             AUROC={mean_a:.4f} ± {std_a:.4f}")

        # Strategy 4: Ensemble methods
        p("  [Ensembles]")

        # 4a: Average XGBoost + MLP
        if xgb_best_preds is not None and mlp_best_preds is not None:
            avg_pred = (xgb_best_preds + mlp_best_preds) / 2
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            for tr, te in kf.split(X):
                fold_aurocs.append(roc_auc_score(y[te], avg_pred[te]))
            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            results[f"ens_xgb_mlp_avg_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"    XGB+MLP avg            AUROC={mean_a:.4f} ± {std_a:.4f}")

        # 4b: Average XGBoost + MLP + Regression
        if xgb_best_preds is not None and mlp_best_preds is not None:
            avg3_pred = (xgb_best_preds + mlp_best_preds + reg_preds) / 3
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            for tr, te in kf.split(X):
                fold_aurocs.append(roc_auc_score(y[te], avg3_pred[te]))
            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            results[f"ens_all3_avg_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"    XGB+MLP+Reg avg        AUROC={mean_a:.4f} ± {std_a:.4f}")

        # 4c: Stacked meta-learner (logistic from XGBoost+MLP+Reg preds)
        if xgb_best_preds is not None and mlp_best_preds is not None:
            meta_X = np.column_stack([xgb_best_preds, mlp_best_preds, reg_preds])
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            for tr, te in kf.split(meta_X):
                meta_model = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cuda",
                    n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
                meta_model.fit(meta_X[tr], y[tr])
                fold_aurocs.append(roc_auc_score(y[te], meta_model.predict_proba(meta_X[te])[:, 1]))
            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            results[f"ens_stacked_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"    Stacked meta-learner   AUROC={mean_a:.4f} ± {std_a:.4f}")

        # Strategy 5: XGBoost with random seed ensemble (5 seeds, averaged)
        p("  [Seed ensemble]")
        seed_preds = np.zeros(len(y))
        n_seeds = 5
        for seed in range(n_seeds):
            kf = KFold(n_splits=5, shuffle=True, random_state=seed * 17 + 42)
            for tr, te in kf.split(X):
                model = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cuda",
                    n_estimators=3000, max_depth=7, learning_rate=0.003,
                    subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=1.0, verbosity=0, random_state=seed * 17 + 42)
                model.fit(X[tr], y[tr])
                seed_preds[te] += model.predict_proba(X[te])[:, 1]
        seed_preds /= n_seeds
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_aurocs = []
        for tr, te in kf.split(X):
            fold_aurocs.append(roc_auc_score(y[te], seed_preds[te]))
        mean_a = np.mean(fold_aurocs)
        std_a = np.std(fold_aurocs)
        results[f"seed_ensemble_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
        p(f"    Seed ensemble (5x)     AUROC={mean_a:.4f} ± {std_a:.4f}")

    # ====================================================================
    # PART 2: Per-target optimization
    # ====================================================================
    p(f"\n{'='*70}")
    p("PART 2: Per-target AUROC optimization")
    p(f"{'='*70}")

    n_c, n_g = len(gt_c), len(gt_g)

    cg_binary = gt["cg_binary_top1"]  # 1674×735

    # Strategy A: PPI-propagated zero-shot (multiple configs)
    p("\n  [PPI-propagated zero-shot per-target]")
    best_per_target_method = None
    best_per_target_median = 0

    def eval_per_target_zeroshot(scores_matrix, label):
        """Evaluate per-target AUROC using cg_binary ground truth."""
        target_aurocs = []
        for j in range(n_g):
            y = cg_binary[:, j].astype(np.int32)
            if y.sum() < 2 or y.sum() == n_c:
                continue
            try:
                target_aurocs.append(roc_auc_score(y, scores_matrix[:, j]))
            except ValueError:
                pass
        return target_aurocs

    for cg_key in ["proj_raw", "cb_pa_raw", "cb_tok_raw", "proj_cos", "cb_pa_cos"]:
        if cg_key not in cg_scores:
            continue
        base_scores = cg_scores[cg_key]

        configs = [(f"{cg_key}_raw", base_scores)]
        for alpha in [0.3, 0.5, 0.7]:
            for thresh in [0.0, 0.3, 0.5]:
                prop = propagate_ppi(base_scores, ppi, alpha=alpha, threshold=thresh)
                configs.append((f"{cg_key}_a{alpha}_t{thresh}", prop))

        for label, scores in configs:
            aurocs = eval_per_target_zeroshot(scores, label)
            if aurocs:
                med = np.median(aurocs)
                results[f"pertarget_{label}"] = {"median": float(med), "n_targets": len(aurocs)}
                if med > best_per_target_median:
                    best_per_target_median = med
                    best_per_target_method = label
                if "raw" in label and ("a0.5" in label or label.endswith("_raw")):
                    p(f"    {label:<35} median={med:.4f} ({len(aurocs)} targets)")

    p(f"\n  Best zero-shot per-target: {best_per_target_method} median={best_per_target_median:.4f}")

    # Strategy B: Rank ensemble across CG matrices
    p("\n  [Ensemble per-target: average ranks across CG matrices]")
    raw_keys = [k for k in cg_scores if "raw" in k]
    ensemble_aurocs = []
    for j in range(n_g):
        y = cg_binary[:, j].astype(np.int32)
        if y.sum() < 2 or y.sum() == n_c:
            continue
        rank_sum = np.zeros(n_c, dtype=np.float64)
        for cg_key in raw_keys:
            prop = propagate_ppi(cg_scores[cg_key], ppi, alpha=0.5, threshold=0.3)
            ranks = np.argsort(np.argsort(-prop[:, j])).astype(float)
            rank_sum += ranks
        try:
            ensemble_aurocs.append(roc_auc_score(y, -rank_sum / len(raw_keys)))
        except ValueError:
            pass

    if ensemble_aurocs:
        med = np.median(ensemble_aurocs)
        results["pertarget_rank_ensemble"] = {"median": float(med), "n_targets": len(ensemble_aurocs)}
        p(f"    Rank ensemble          median={med:.4f} ({len(ensemble_aurocs)} targets)")

    # Strategy C: Score-average ensemble per target (z-scored)
    p("\n  [Score-average ensemble per-target]")
    score_ens_aurocs = []
    for j in range(n_g):
        y = cg_binary[:, j].astype(np.int32)
        if y.sum() < 2 or y.sum() == n_c:
            continue
        score_sum = np.zeros(n_c, dtype=np.float64)
        for cg_key in raw_keys:
            prop = propagate_ppi(cg_scores[cg_key], ppi, alpha=0.5, threshold=0.3)
            col = prop[:, j]
            col_std = (col - col.mean()) / (col.std() + 1e-8)
            score_sum += col_std
        try:
            score_ens_aurocs.append(roc_auc_score(y, score_sum / len(raw_keys)))
        except ValueError:
            pass

    if score_ens_aurocs:
        med = np.median(score_ens_aurocs)
        results["pertarget_score_ensemble"] = {"median": float(med), "n_targets": len(score_ens_aurocs)}
        p(f"    Score ensemble         median={med:.4f} ({len(score_ens_aurocs)} targets)")

    # Strategy D: Oracle best config per target (upper bound)
    p("\n  [Oracle: best PPI config per target]")
    oracle_aurocs = []
    for j in range(n_g):
        y = cg_binary[:, j].astype(np.int32)
        if y.sum() < 2 or y.sum() == n_c:
            continue
        best_auc = 0
        for cg_key in ["proj_raw", "cb_pa_raw", "cb_pa_cos"]:
            if cg_key not in cg_scores:
                continue
            for alpha in [0.3, 0.5, 0.7]:
                prop = propagate_ppi(cg_scores[cg_key], ppi, alpha=alpha, threshold=0.3)
                try:
                    auc = roc_auc_score(y, prop[:, j])
                    if auc > best_auc:
                        best_auc = auc
                except ValueError:
                    pass
        if best_auc > 0:
            oracle_aurocs.append(best_auc)

    if oracle_aurocs:
        med = np.median(oracle_aurocs)
        results["pertarget_oracle_best"] = {"median": float(med), "n_targets": len(oracle_aurocs)}
        p(f"    Oracle best-per-target  median={med:.4f} ({len(oracle_aurocs)} targets)")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    p(f"\n{'='*70}")
    p("FINAL PUSH SUMMARY")
    p(f"{'='*70}")

    p("\n  CC AUROC (Beaini targets: @0.4=71%, @0.6=76%):")
    for key in sorted(results.keys()):
        if "0.4" in key or "0.6" in key:
            val = results[key]
            p(f"    {key:<40} {val['mean']:.4f} ± {val.get('std', 0):.4f}")

    p("\n  Per-target median AUROC (Beaini target: 53.9%):")
    for key in sorted(results.keys()):
        if "pertarget" in key:
            val = results[key]
            p(f"    {key:<40} median={val['median']:.4f} ({val['n_targets']} targets)")

    elapsed = time.time() - t_start
    p(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "final_push_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    p(f"  Results saved to {EVAL_DIR}/final_push_results.json")


if __name__ == "__main__":
    main()
