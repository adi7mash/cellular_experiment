"""Evaluate fusion CLS + signatures combined.

Adds fusion features on top of the full 52-feature set (42 baseline + 10 signatures)
to test whether fusion cross-attention and phenomics signatures provide complementary signal.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.home() / "cellular_experiment" / "analysis"))
from phenomics_signature import (
    load_data, build_original_42_features, select_known,
    learn_phenomics_signature, signature_similarity,
    build_signature_features, build_cg_signature_features,
    extract_pairs, propagate_ppi, PairMLP, train_mlp_fold,
)
from fusion_cls_integration import load_ground_truth, load_fusion_cls, build_fusion_features

RESULTS_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"
DEVICE = "cuda"


def p(*a, **k):
    print(*a, **k, flush=True)


def run_pair_cv(X, y, y_cont, label, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = {"xgb": np.zeros(len(y)), "mlp": np.zeros(len(y)), "xgb_reg": np.zeros(len(y))}
    fold_aucs = {"xgb": [], "mlp": [], "xgb_reg": []}

    for fold, (tr, te) in enumerate(kf.split(X)):
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_te)) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        X_tr_s = scaler.transform(X[tr])
        X_te_s = scaler.transform(X[te])

        xgb_m = xgb.XGBClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        )
        xgb_m.fit(X_tr_s, y_tr)
        pred = xgb_m.predict_proba(X_te_s)[:, 1]
        oof["xgb"][te] = pred
        fold_aucs["xgb"].append(roc_auc_score(y_te, pred))

        xgb_reg = xgb.XGBRegressor(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
        )
        xgb_reg.fit(X_tr_s, y_cont[tr])
        pred = xgb_reg.predict(X_te_s)
        oof["xgb_reg"][te] = pred
        fold_aucs["xgb_reg"].append(roc_auc_score(y_te, pred))

        pred, auc = train_mlp_fold(X_tr_s, y_tr, X_te_s, y_te, [1024, 512, 256, 128])
        oof["mlp"][te] = pred
        fold_aucs["mlp"].append(auc)

        p(f"    Fold {fold}: xgb={fold_aucs['xgb'][-1]:.4f} mlp={fold_aucs['mlp'][-1]:.4f} "
          f"reg={fold_aucs['xgb_reg'][-1]:.4f}")

    results = {}
    for m in oof:
        if fold_aucs[m]:
            mean_a = np.mean(fold_aucs[m])
            results[m] = {"mean": mean_a, "std": np.std(fold_aucs[m])}
            p(f"    {m:15s} {mean_a:.4f} +/- {np.std(fold_aucs[m]):.4f}")

    avg3 = (oof["xgb"] + oof["mlp"] + oof["xgb_reg"]) / 3
    ens_aucs = []
    for _, te in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X):
        ens_aucs.append(roc_auc_score(y[te], avg3[te]))
    ens_mean = np.mean(ens_aucs)
    results["ensemble"] = {"mean": ens_mean, "std": np.std(ens_aucs)}
    p(f"    {'ensemble':15s} {ens_mean:.4f} +/- {np.std(ens_aucs):.4f}")

    return results


def main():
    p("=" * 70)
    p("FUSION CLS + SIGNATURES EVALUATION")
    p("=" * 70)

    # Load all data
    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores = data[:7]
    mol_tok, mol_pa, mol_proj = data[7:10]
    cc_sim = gt["cc_similarity"]

    known_idx = select_known(cg_scores, 150)
    ki = known_idx
    p(f"  {n_c} compounds, {n_g} genes, {len(ki)} known")

    # Build baseline 42 features
    baseline_features = build_original_42_features(cg_scores, ppi, gt_c, n_c)
    p(f"  Baseline features: {len(baseline_features)}")

    # Build signature features (global — same as pair-level CV)
    p("\n  Building signatures...")
    sig_features = build_signature_features(mol_tok, mol_pa, mol_proj, cc_sim, ki)
    cg_sig_features = build_cg_signature_features(cg_scores, ppi, cc_sim, ki)
    sig_all = {**sig_features, **cg_sig_features}
    p(f"  Signature features: {len(sig_all)}")

    # Build fusion features
    p("\n  Loading fusion CLS...")
    gt2, gt_c2, gt_g2 = load_ground_truth()
    fusion_data, valid_compounds = load_fusion_cls(gt_c, gt_g)
    fusion_features = build_fusion_features(fusion_data, gt_g, n_c, valid_compounds)
    p(f"  Fusion features: {len(fusion_features)}")

    # Feature sets to test
    feature_sets = {
        "baseline_42": baseline_features,
        "baseline_sig_52": {**baseline_features, **sig_all},
        "baseline_fusion_47": {**baseline_features, **fusion_features},
        "baseline_sig_fusion_57": {**baseline_features, **sig_all, **fusion_features},
    }

    results = {}

    for thresh, label in [(0.6, "0.6"), (0.4, "0.4")]:
        p(f"\n{'='*70}")
        p(f"CC AUROC — PAIR-LEVEL CV @{label}")
        p(f"{'='*70}")

        for feat_name, feat_set in feature_sets.items():
            p(f"\n  --- {feat_name} ({len(feat_set)} features) ---")
            X, names, pair_idx = extract_pairs(feat_set, ki)
            sub_sim = cc_sim[np.ix_(ki, ki)]
            y = (sub_sim[pair_idx] > thresh).astype(np.float32)
            y_cont = sub_sim[pair_idx].astype(np.float32)

            res = run_pair_cv(X, y, y_cont, f"{label}_{feat_name}")
            for model_name, model_res in res.items():
                results[f"pair_{feat_name}_{model_name}_{label}"] = model_res["mean"]

    # Summary
    p(f"\n{'='*70}")
    p("SUMMARY")
    p(f"{'='*70}")

    p("\n  Pair-level CV @0.6:")
    p(f"  {'Feature set':35s} {'XGB':>8s} {'MLP':>8s} {'Reg':>8s} {'Ens':>8s}")
    for feat_name in feature_sets:
        xgb_v = results.get(f"pair_{feat_name}_xgb_0.6", 0)
        mlp_v = results.get(f"pair_{feat_name}_mlp_0.6", 0)
        reg_v = results.get(f"pair_{feat_name}_xgb_reg_0.6", 0)
        ens_v = results.get(f"pair_{feat_name}_ensemble_0.6", 0)
        p(f"  {feat_name:35s} {xgb_v:8.4f} {mlp_v:8.4f} {reg_v:8.4f} {ens_v:8.4f}")

    p("\n  Pair-level CV @0.4:")
    p(f"  {'Feature set':35s} {'XGB':>8s} {'MLP':>8s} {'Reg':>8s} {'Ens':>8s}")
    for feat_name in feature_sets:
        xgb_v = results.get(f"pair_{feat_name}_xgb_0.4", 0)
        mlp_v = results.get(f"pair_{feat_name}_mlp_0.4", 0)
        reg_v = results.get(f"pair_{feat_name}_xgb_reg_0.4", 0)
        ens_v = results.get(f"pair_{feat_name}_ensemble_0.4", 0)
        p(f"  {feat_name:35s} {xgb_v:8.4f} {mlp_v:8.4f} {reg_v:8.4f} {ens_v:8.4f}")

    out_path = os.path.join(RESULTS_DIR, "fusion_signatures_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    p(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
