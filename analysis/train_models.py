"""Train models to predict phenomics from Bonbon embeddings (GPU-accelerated).

Two tasks:
1. CC (compound-compound): predict phenomics similarity from compound pair features
2. Per-target: predict which compounds are phenomically active for each target
"""
import json
import os
import sys
import time
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"


def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def align_to_gt(pred_ids, gt_ids):
    pred_map = {c: i for i, c in enumerate(pred_ids)}
    return np.array([pred_map.get(g, -1) for g in gt_ids])


def align_cc_matrix(mat, pred_ids, gt_ids):
    idx = align_to_gt(pred_ids, gt_ids)
    valid = idx >= 0
    n = len(gt_ids)
    aligned = np.zeros((n, n), dtype=np.float32)
    vi = np.where(valid)[0]
    aligned[np.ix_(vi, vi)] = mat[np.ix_(idx[valid], idx[valid])]
    return aligned


def align_cg_matrix(mat, pred_c_ids, pred_g_ids, gt_c_ids, gt_g_ids):
    c_idx = align_to_gt(pred_c_ids, gt_c_ids)
    g_idx = align_to_gt(pred_g_ids, gt_g_ids)
    n_c, n_g = len(gt_c_ids), len(gt_g_ids)
    aligned = np.zeros((n_c, n_g), dtype=np.float32)
    vc, vg = c_idx >= 0, g_idx >= 0
    aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat[np.ix_(c_idx[vc], g_idx[vg])]
    return aligned


def build_cc_features(ip_dir, gt_c_ids, gt_dir):
    """Build CC pair features. Returns X (n_pairs, n_features), names, pair_indices."""
    flush_print("\n--- Building CC features ---")
    n = len(gt_c_ids)
    pair_idx = np.triu_indices(n, k=1)
    features = {}

    # ECFP Tanimoto
    ecfp = np.load(os.path.join(gt_dir, "ecfp_tanimoto.npz"))
    aligned = align_cc_matrix(ecfp["tanimoto"], ecfp["compound_ids"].tolist(), gt_c_ids)
    features["ecfp"] = aligned[pair_idx]
    flush_print(f"  ecfp: done")

    # Direct CC sim matrices
    for fname, label in [
        ("pooled_attention_concat.npz", "pa_concat"),
        ("pooled_attention_product.npz", "pa_product"),
        ("tokens_concat.npz", "tok_concat"),
        ("tokens_product.npz", "tok_product"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        aligned = align_cc_matrix(data["similarity"], data["compound_ids"].tolist(), gt_c_ids)
        features[label] = aligned[pair_idx]
        flush_print(f"  {label}: done")

    # CC profile similarity from CG matrices (compound × gene)
    for fname, label in [
        ("cg_projection.npz", "proj"),
        ("cg_codebook_tokens.npz", "cb_tok"),
        ("cg_pooled_attention.npz", "cb_pa"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        pred_c = data["compound_ids"].tolist()
        c_idx = align_to_gt(pred_c, gt_c_ids)
        valid = c_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            if sk not in data:
                continue
            mat = data[sk]
            profiles = np.zeros((n, mat.shape[1]), dtype=np.float32)
            profiles[np.where(valid)[0]] = mat[c_idx[valid]]
            cc = cosine_similarity(profiles).astype(np.float32)
            features[f"{label}_{suffix}_prof"] = cc[pair_idx]
            flush_print(f"  {label}_{suffix}_prof: done")

    # CC profile from binary matrices (n_proteins × n_compounds → transpose)
    for fname, label in [
        ("binary_projection.npz", "bin_proj"),
        ("binary_codebook_tokens.npz", "bin_tok"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        pred_c = data["compound_ids"].tolist()
        c_idx = align_to_gt(pred_c, gt_c_ids)
        valid = c_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            if sk not in data:
                continue
            mat_T = data[sk].T  # (n_mol, n_prot)
            profiles = np.zeros((n, mat_T.shape[1]), dtype=np.float32)
            profiles[np.where(valid)[0]] = mat_T[c_idx[valid]]
            cc = cosine_similarity(profiles).astype(np.float32)
            features[f"{label}_{suffix}_prof"] = cc[pair_idx]
            flush_print(f"  {label}_{suffix}_prof: done")

    names = list(features.keys())
    X = np.column_stack([features[k] for k in names])
    flush_print(f"\nCC features: {X.shape[1]} features, {X.shape[0]:,} pairs")
    for i, nm in enumerate(names):
        flush_print(f"  {nm:<30} mean={X[:, i].mean():.4f}  std={X[:, i].std():.4f}")
    return X, names, pair_idx


def build_cg_features(ip_dir, gt_c_ids, gt_g_ids):
    """Build per-(compound, gene) feature layers."""
    flush_print("\n--- Building CG features ---")
    n_c, n_g = len(gt_c_ids), len(gt_g_ids)
    features = {}

    for fname, label in [
        ("cg_projection.npz", "proj"),
        ("cg_codebook_tokens.npz", "cb_tok"),
        ("cg_pooled_attention.npz", "cb_pa"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        pred_c, pred_g = data["compound_ids"].tolist(), data["protein_ids"].tolist()
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            if sk not in data:
                continue
            aligned = align_cg_matrix(data[sk], pred_c, pred_g, gt_c_ids, gt_g_ids)
            features[f"{label}_{suffix}"] = aligned
            flush_print(f"  {label}_{suffix}: {aligned.shape}")

    for fname, label in [
        ("binary_projection.npz", "bin_proj"),
        ("binary_codebook_tokens.npz", "bin_tok"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue
        data = np.load(fpath)
        pred_c, pred_g = data["compound_ids"].tolist(), data["protein_ids"].tolist()
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            if sk not in data:
                continue
            aligned = align_cg_matrix(data[sk].T, pred_c, pred_g, gt_c_ids, gt_g_ids)
            features[f"{label}_{suffix}"] = aligned
            flush_print(f"  {label}_{suffix}: {aligned.shape}")

    names = list(features.keys())
    layers = [features[k] for k in names]
    flush_print(f"CG features: {len(names)} per (compound, gene) pair")
    return layers, names


def xgb_cv(X, y, n_folds=5, gpu=True, **params):
    """Run XGBoost with CV, return per-fold AUROCs."""
    default = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        device="cuda" if gpu else "cpu",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        verbosity=0,
    )
    default.update(params)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_aurocs = []
    for train_idx, test_idx in kf.split(X):
        model = xgb.XGBClassifier(**default)
        model.fit(X[train_idx], y[train_idx], verbose=False)
        y_pred = model.predict_proba(X[test_idx])[:, 1]
        fold_aurocs.append(roc_auc_score(y[test_idx], y_pred))
    return fold_aurocs


def train_cc(X, gt, pair_idx, feature_names, eval_dir):
    """Train CC models at all thresholds."""
    flush_print(f"\n{'='*70}")
    flush_print(f"COMPOUND-COMPOUND MODEL TRAINING (XGBoost GPU)")
    flush_print(f"{'='*70}")

    configs = [
        ("XGB_100_d3", dict(n_estimators=100, max_depth=3, learning_rate=0.1)),
        ("XGB_200_d4", dict(n_estimators=200, max_depth=4, learning_rate=0.05)),
        ("XGB_500_d5", dict(n_estimators=500, max_depth=5, learning_rate=0.03)),
        ("XGB_1000_d6", dict(n_estimators=1000, max_depth=6, learning_rate=0.01)),
    ]

    all_results = {}
    for thresh_name, thresh_key in [
        ("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
        ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95"),
    ]:
        if thresh_key not in gt:
            continue
        y = gt[thresh_key][pair_idx].astype(np.int32)
        pos_rate = y.mean()
        flush_print(f"\n--- CC threshold {thresh_name} ({pos_rate:.2%} positive) ---")

        results = {}
        for mname, mparams in configs:
            t0 = time.time()
            folds = xgb_cv(X, y, **mparams)
            mean_a = np.mean(folds)
            std_a = np.std(folds)
            elapsed = time.time() - t0
            results[mname] = {"mean": float(mean_a), "std": float(std_a),
                              "folds": [float(f) for f in folds]}
            flush_print(f"  {mname:<18} AUROC={mean_a:.4f} ± {std_a:.4f}  [{elapsed:.1f}s]")

        # Feature importance from best model
        best_name = max(results, key=lambda k: results[k]["mean"])
        flush_print(f"  Best: {best_name} = {results[best_name]['mean']:.4f}")

        best_params = dict(configs[[c[0] for c in configs].index(best_name)][1])
        model = xgb.XGBClassifier(
            objective="binary:logistic", tree_method="hist", device="cuda",
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, verbosity=0, **best_params)
        model.fit(X, y)
        flush_print("  Feature importances:")
        for fn, imp in sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1]):
            flush_print(f"    {fn:<30} {imp:.4f}")

        all_results[thresh_name] = results

    with open(os.path.join(eval_dir, "cc_model_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    return all_results


def train_per_target(cg_layers, cg_names, cg_binary, gene_ids, eval_dir):
    """Train per-target models with XGBoost GPU."""
    flush_print(f"\n{'='*70}")
    flush_print(f"PER-TARGET MODEL TRAINING (XGBoost GPU)")
    flush_print(f"{'='*70}")

    n_c, n_g = cg_binary.shape
    flush_print(f"Compounds: {n_c}, Targets: {n_g}, Features: {len(cg_layers)}")
    flush_print(f"Positives per target: ~{int(cg_binary[:, 0].sum())}")

    configs = [
        ("XGB_50_d3", dict(n_estimators=50, max_depth=3, learning_rate=0.1)),
        ("XGB_100_d4", dict(n_estimators=100, max_depth=4, learning_rate=0.05)),
    ]

    all_results = {}

    for mname, mparams in configs:
        t0 = time.time()
        target_aurocs = []

        for j in range(n_g):
            y = cg_binary[:, j].astype(np.int32)
            n_pos = y.sum()
            if n_pos < 3 or n_pos == n_c:
                continue

            X = np.column_stack([layer[:, j] for layer in cg_layers])
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
                        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                        scale_pos_weight=(n_c - n_pos) / max(n_pos, 1),
                        verbosity=0, **mparams)
                    model.fit(X[train_idx], y[train_idx], verbose=False)
                    y_pred = model.predict_proba(X[test_idx])[:, 1]
                    auroc = roc_auc_score(y_test, y_pred)
                    if not np.isnan(auroc):
                        fold_aurocs.append(auroc)
                except Exception:
                    pass
            if fold_aurocs:
                target_aurocs.append(np.mean(fold_aurocs))

        aurocs = np.array(target_aurocs)
        elapsed = time.time() - t0
        all_results[mname] = {
            "median": float(np.median(aurocs)),
            "mean": float(np.mean(aurocs)),
            "p25": float(np.percentile(aurocs, 25)),
            "p75": float(np.percentile(aurocs, 75)),
            "n_targets": len(aurocs),
            "per_target": [float(a) for a in aurocs],
        }
        flush_print(f"  {mname:<18} median={np.median(aurocs):.4f}  mean={np.mean(aurocs):.4f}  "
                     f"p25={np.percentile(aurocs, 25):.4f}  p75={np.percentile(aurocs, 75):.4f}  "
                     f"({len(aurocs)} targets) [{elapsed:.1f}s]")

    with open(os.path.join(eval_dir, "per_target_model_results.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_target"}
                   for k, v in all_results.items()}, f, indent=2)
    return all_results


def train_cc_regression(X, gt_dir, gt, pair_idx, feature_names, eval_dir):
    """Train regression model on continuous phenomics similarity, evaluate as AUROC."""
    flush_print(f"\n{'='*70}")
    flush_print(f"CC REGRESSION (continuous sim → AUROC)")
    flush_print(f"{'='*70}")

    cc_sim = np.load(os.path.join(gt_dir, "phenomics_compound_compound_sim.npz"))["similarity"]
    y_cont = cc_sim[pair_idx]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    params = dict(
        objective="reg:squarederror", tree_method="hist", device="cuda",
        n_estimators=300, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, verbosity=0)

    fold_results = {t: [] for t in ["0.4", "0.6", "P90", "P95"]}
    t0 = time.time()

    for train_idx, test_idx in kf.split(X):
        model = xgb.XGBRegressor(**params)
        model.fit(X[train_idx], y_cont[train_idx])
        y_pred = model.predict(X[test_idx])

        for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                            ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
            if tkey in gt:
                y_bin = gt[tkey][pair_idx][test_idx].astype(np.int32)
                if 0 < y_bin.sum() < len(y_bin):
                    fold_results[tname].append(roc_auc_score(y_bin, y_pred))

    elapsed = time.time() - t0
    results = {}
    for tname, folds in fold_results.items():
        if folds:
            mean_a, std_a = np.mean(folds), np.std(folds)
            results[tname] = {"mean": float(mean_a), "std": float(std_a)}
            flush_print(f"  XGB_Reg @{tname:<4}: AUROC={mean_a:.4f} ± {std_a:.4f}")
    flush_print(f"  [{elapsed:.1f}s]")

    return results


def main(results_dir, gt_dir=GT_DIR):
    ip_dir = os.path.join(results_dir, "interaction_prints")
    eval_dir = os.path.join(results_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    gt = np.load(os.path.join(gt_dir, "ground_truth_binary.npz"))
    gt_c_ids = gt["compound_ids"].tolist()
    gt_g_ids = gt["gene_ids"].tolist()
    flush_print(f"Ground truth: {len(gt_c_ids)} compounds, {len(gt_g_ids)} genes")

    # Build features
    X_cc, cc_names, pair_idx = build_cc_features(ip_dir, gt_c_ids, gt_dir)
    cg_layers, cg_names = build_cg_features(ip_dir, gt_c_ids, gt_g_ids)

    # CC classification
    cc_results = train_cc(X_cc, gt, pair_idx, cc_names, eval_dir)

    # CC regression
    reg_results = train_cc_regression(X_cc, gt_dir, gt, pair_idx, cc_names, eval_dir)

    # Per-target
    cg_binary = gt["cg_binary_top1"]
    pt_results = train_per_target(cg_layers, cg_names, cg_binary, gt_g_ids, eval_dir)

    # Summary
    flush_print(f"\n{'='*70}")
    flush_print("FINAL SUMMARY")
    flush_print(f"{'='*70}")

    flush_print("\nCC AUROC (best per threshold):")
    for thresh, results in cc_results.items():
        best_name = max(results, key=lambda k: results[k]["mean"])
        best = results[best_name]
        flush_print(f"  {thresh:<6} {best_name:<18} {best['mean']:.4f} ± {best['std']:.4f}")

    flush_print("\nCC Regression AUROC:")
    for tname, r in reg_results.items():
        flush_print(f"  @{tname:<6} {r['mean']:.4f} ± {r['std']:.4f}")

    flush_print(f"\n  Beaini: 71-76% (CC, 6-param + PPI + transcriptomics, 246 known compounds)")
    flush_print(f"  Beaini ECFP: 63-64% (through same pipeline)")

    flush_print("\nPer-target AUROC:")
    for mname, r in pt_results.items():
        flush_print(f"  {mname:<18} median={r['median']:.4f}  mean={r['mean']:.4f}")
    flush_print(f"\n  Beaini: 53.9% median (across 7K proteins, zero-shot binary)")


if __name__ == "__main__":
    results_dir = sys.argv[1]
    gt_dir = sys.argv[2] if len(sys.argv) > 2 else GT_DIR
    main(results_dir, gt_dir)
