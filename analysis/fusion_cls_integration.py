"""Integrate fusion CLS tokens into the phenomics prediction pipeline.

Uses the extracted per-(protein, molecule) fusion CLS vectors to build new
features for CC AUROC and per-target prediction.

Feature strategies:
  1. Compound-level mean CLS: average fusion CLS across all 735 proteins → 1024-dim compound vector
  2. CG score from CLS norm: scalar interaction strength per (compound, protein)
  3. CC similarity from CLS profile cosine: for each protein, cosine(CLS(i,p), CLS(j,p)), averaged
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path.home() / "cellular_experiment" / "analysis"))

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"
FUSION_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/fusion_cls")
RESULTS_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"


def p(*a, **k):
    print(*a, **k, flush=True)


def load_ground_truth():
    gt = dict(np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"), allow_pickle=True))
    for k in gt:
        gt[k] = np.array(gt[k])
    cc_sim_data = np.load(os.path.join(GT_DIR, "phenomics_compound_compound_sim.npz"))
    gt["cc_similarity"] = np.array(cc_sim_data["similarity"])
    gt_c = gt["compound_ids"].tolist()
    gt_g = gt["gene_ids"].tolist()
    return gt, gt_c, gt_g


def load_fusion_cls(gt_c, gt_g):
    """Load fusion CLS tokens. Returns dict of {protein: tensor[n_compounds, hidden_dim]}."""
    mol_index = json.load(open(FUSION_DIR / "molecule_index.json"))
    mol_to_idx = {m: i for i, m in enumerate(mol_index)}

    # Map gt compounds to molecule index positions
    compound_idx = []
    valid_compounds = []
    for i, c in enumerate(gt_c):
        if c in mol_to_idx:
            compound_idx.append(mol_to_idx[c])
            valid_compounds.append(i)
    compound_idx = np.array(compound_idx)
    valid_compounds = np.array(valid_compounds)

    p(f"  {len(valid_compounds)}/{len(gt_c)} compounds matched in fusion CLS")

    # Load per-protein CLS matrices, select only gt compounds
    fusion_data = {}
    loaded = 0
    for gene in gt_g:
        fpath = FUSION_DIR / f"{gene}.pt"
        if fpath.exists():
            cls_mat = torch.load(fpath, map_location="cpu", weights_only=True)
            fusion_data[gene] = cls_mat[compound_idx].numpy()  # [n_valid_compounds, hidden_dim]
            loaded += 1

    p(f"  {loaded}/{len(gt_g)} proteins loaded from fusion CLS")
    return fusion_data, valid_compounds


def build_fusion_features(fusion_data, gt_g, n_c, valid_compounds):
    """Build CC similarity features from fusion CLS tokens."""
    features = {}
    hidden_dim = next(iter(fusion_data.values())).shape[1]
    n_valid = len(valid_compounds)

    p(f"  Building fusion features: {n_valid} compounds, {len(fusion_data)} proteins, {hidden_dim}-dim CLS")

    # Strategy 1: Compound-level mean CLS vector
    # Average fusion CLS across all proteins → 1024-dim per compound
    p("  Strategy 1: Compound mean CLS → CC cosine similarity...")
    mean_cls = np.zeros((n_valid, hidden_dim), dtype=np.float32)
    count = 0
    for gene in gt_g:
        if gene in fusion_data:
            mean_cls += fusion_data[gene]
            count += 1
    mean_cls /= max(count, 1)

    # Normalize for cosine
    norms = np.linalg.norm(mean_cls, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    mean_cls_normed = mean_cls / norms

    sim_mean_cls = np.zeros((n_c, n_c), dtype=np.float32)
    sim_mean_cls[np.ix_(valid_compounds, valid_compounds)] = mean_cls_normed @ mean_cls_normed.T
    features["fusion_mean_cls_cos"] = sim_mean_cls

    # Strategy 2: CG score from CLS norm → compound profile cosine
    p("  Strategy 2: CLS norm CG scores → profile cosine...")
    cg_norm = np.zeros((n_c, len(gt_g)), dtype=np.float32)
    for j, gene in enumerate(gt_g):
        if gene in fusion_data:
            cg_norm[valid_compounds, j] = np.linalg.norm(fusion_data[gene], axis=1)
    # Normalize per-gene to z-scores
    for j in range(cg_norm.shape[1]):
        col = cg_norm[:, j]
        if col.std() > 0:
            cg_norm[:, j] = (col - col.mean()) / col.std()

    from sklearn.metrics.pairwise import cosine_similarity
    features["fusion_norm_cg_cos"] = cosine_similarity(cg_norm).astype(np.float32)

    # Strategy 3: Per-protein CLS cosine, averaged across proteins
    p("  Strategy 3: Per-protein CLS cosine (averaged)...")
    avg_cls_sim = np.zeros((n_valid, n_valid), dtype=np.float32)
    count = 0
    for gene in gt_g:
        if gene in fusion_data:
            cls = fusion_data[gene]  # [n_valid, hidden_dim]
            n = np.linalg.norm(cls, axis=1, keepdims=True)
            n = np.maximum(n, 1e-8)
            normed = cls / n
            avg_cls_sim += normed @ normed.T
            count += 1
    avg_cls_sim /= max(count, 1)

    full_sim = np.zeros((n_c, n_c), dtype=np.float32)
    full_sim[np.ix_(valid_compounds, valid_compounds)] = avg_cls_sim
    features["fusion_per_prot_cls_cos"] = full_sim

    # Strategy 4: Top-K proteins by CLS variance (most discriminative)
    p("  Strategy 4: Top-50 discriminative proteins CLS cosine...")
    prot_var = {}
    for gene in gt_g:
        if gene in fusion_data:
            prot_var[gene] = np.var(fusion_data[gene])
    top_k_prots = sorted(prot_var, key=prot_var.get, reverse=True)[:50]

    top_cls_sim = np.zeros((n_valid, n_valid), dtype=np.float32)
    for gene in top_k_prots:
        cls = fusion_data[gene]
        n = np.linalg.norm(cls, axis=1, keepdims=True)
        n = np.maximum(n, 1e-8)
        normed = cls / n
        top_cls_sim += normed @ normed.T
    top_cls_sim /= len(top_k_prots)

    full_top = np.zeros((n_c, n_c), dtype=np.float32)
    full_top[np.ix_(valid_compounds, valid_compounds)] = top_cls_sim
    features["fusion_top50_cls_cos"] = full_top

    # Strategy 5: CLS dot product CG score → profile
    p("  Strategy 5: Mean CLS per compound as embedding → pairwise dot product...")
    dot_sim = np.zeros((n_c, n_c), dtype=np.float32)
    dot_sim[np.ix_(valid_compounds, valid_compounds)] = mean_cls @ mean_cls.T
    features["fusion_mean_cls_dot"] = dot_sim

    p(f"  Built {len(features)} fusion features")
    return features


def select_known(cg_scores, n_select):
    combined = np.zeros(list(cg_scores.values())[0].shape[0])
    for scores in cg_scores.values():
        row_std = scores.std(axis=1)
        combined += row_std / (row_std.max() + 1e-8)
    return np.argsort(-combined)[:n_select]


def load_baseline_features():
    """Load the same baseline features as phenomics_signature.py."""
    from phenomics_signature import load_data, build_original_42_features, select_known as sk
    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores = data[:7]
    mol_tok, mol_pa, mol_proj = data[7:10]

    baseline_features = build_original_42_features(cg_scores, ppi, gt_c, n_c)
    known_idx = sk(cg_scores, 150)

    return (gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores,
            mol_tok, mol_pa, mol_proj, baseline_features, known_idx)


def extract_pairs(features, subset_idx):
    ki = subset_idx
    n = len(ki)
    names = list(features.keys())
    rows = []
    pair_idx = []
    for i in range(n):
        for j in range(i + 1, n):
            row = [features[k][ki[i], ki[j]] for k in names]
            rows.append(row)
            pair_idx.append((i, j))
    X = np.array(rows, dtype=np.float32)
    return X, names, pair_idx


def run_cv(X, y, y_cont, thresh_label, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []
    oof = np.zeros(len(y))

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
        oof[te] = pred
        fold_aucs.append(roc_auc_score(y_te, pred))
        p(f"    Fold {fold}: xgb={fold_aucs[-1]:.4f}")

    mean_auc = np.mean(fold_aucs)
    p(f"    Mean: {mean_auc:.4f} +/- {np.std(fold_aucs):.4f}")
    return mean_auc


def run_compound_cv(features, cc_sim, known_idx, thresh_val, n_splits=5):
    """Compound-level CV with per-fold evaluation."""
    ki = known_idx
    n_k = len(ki)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    names = list(features.keys())

    for fold, (tr_comp, te_comp) in enumerate(kf.split(ki)):
        tr_global = ki[tr_comp]
        te_global = ki[te_comp]

        tr_rows, tr_labels = [], []
        for i in range(len(tr_comp)):
            for j in range(i + 1, len(tr_comp)):
                ci, cj = tr_global[i], tr_global[j]
                row = [features[k][ci, cj] for k in names]
                tr_rows.append(row)
                tr_labels.append(1.0 if cc_sim[ci, cj] > thresh_val else 0.0)

        te_rows, te_labels = [], []
        for i in range(len(te_comp)):
            for j in range(i + 1, len(te_comp)):
                ci, cj = te_global[i], te_global[j]
                row = [features[k][ci, cj] for k in names]
                te_rows.append(row)
                te_labels.append(1.0 if cc_sim[ci, cj] > thresh_val else 0.0)

        X_tr = np.array(tr_rows, dtype=np.float32)
        y_tr = np.array(tr_labels)
        X_te = np.array(te_rows, dtype=np.float32)
        y_te = np.array(te_labels)

        if len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler().fit(X_tr)
        xgb_m = xgb.XGBClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        )
        xgb_m.fit(scaler.transform(X_tr), y_tr)
        pred = xgb_m.predict_proba(scaler.transform(X_te))[:, 1]
        auc = roc_auc_score(y_te, pred)
        fold_aucs.append(auc)
        p(f"    Fold {fold}: xgb={auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    p(f"    Mean: {mean_auc:.4f} +/- {np.std(fold_aucs):.4f}")
    return mean_auc


def per_target_with_fusion(gt, gt_c, gt_g, n_c, n_g, fusion_data, valid_compounds, known_idx):
    """Per-target AUROC using fusion CLS norm as CG score."""
    p("\n" + "=" * 70)
    p("PER-TARGET AUROC — fusion CLS features")
    p("=" * 70)

    cg_binary = gt["cg_binary_top1"]

    # Build CG score from fusion CLS norm
    cg_fusion_norm = np.zeros((n_c, len(gt_g)), dtype=np.float32)
    for j, gene in enumerate(gt_g):
        if gene in fusion_data:
            cg_fusion_norm[valid_compounds, j] = np.linalg.norm(fusion_data[gene], axis=1)

    ppi = np.load(os.path.join(GT_DIR, "bonbon_ppi.npz"))["ppi_matrix"]
    from phenomics_signature import propagate_ppi

    ki = known_idx
    results = {}
    for config_name, cg_score, alpha, thresh in [
        ("fusion_norm", cg_fusion_norm, 0.0, 0.0),
        ("fusion_norm_05_03", cg_fusion_norm, 0.5, 0.3),
    ]:
        if alpha > 0:
            score = propagate_ppi(cg_score, ppi, alpha=alpha, threshold=thresh)
        else:
            score = cg_score

        target_aucs = []
        for j in range(n_g):
            y = cg_binary[ki, j]
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue
            pred = score[ki, j]
            if pred.std() < 1e-8:
                continue
            try:
                auc = roc_auc_score(y, pred)
                target_aucs.append(auc)
            except:
                pass

        if target_aucs:
            med = np.median(target_aucs)
            results[config_name] = {"median": med, "n_targets": len(target_aucs)}
            p(f"  {config_name}: median={med:.4f} ({len(target_aucs)} targets)")

    return results


def main():
    p("=" * 70)
    p("FUSION CLS INTEGRATION")
    p("=" * 70)

    # Load ground truth
    gt, gt_c, gt_g = load_ground_truth()
    n_c, n_g = len(gt_c), len(gt_g)
    cc_sim = gt["cc_similarity"]
    p(f"  {n_c} compounds, {n_g} genes")

    # Load fusion CLS
    p("\nLoading fusion CLS tokens...")
    fusion_data, valid_compounds = load_fusion_cls(gt_c, gt_g)

    # Build fusion features
    p("\nBuilding fusion features...")
    t0 = time.time()
    fusion_features = build_fusion_features(fusion_data, gt_g, n_c, valid_compounds)
    p(f"  Features built in {time.time() - t0:.1f}s")

    # Load baseline features
    p("\nLoading baseline features...")
    data = load_baseline_features()
    (_, _, _, _, _, ppi, cg_scores,
     mol_tok, mol_pa, mol_proj, baseline_features, known_idx) = data

    # Combine features
    combined_features = {**baseline_features, **fusion_features}
    p(f"\n  Baseline features: {len(baseline_features)}")
    p(f"  Fusion features: {len(fusion_features)}")
    p(f"  Combined: {len(combined_features)}")

    # Run evaluations
    ki = known_idx
    results = {}

    p("\n" + "=" * 70)
    p("CC AUROC — PAIR-LEVEL CV")
    p("=" * 70)

    for thresh, label in [(0.6, "0.6"), (0.4, "0.4")]:
        for feat_set_name, feat_set in [
            ("baseline", baseline_features),
            ("fusion_only", fusion_features),
            ("combined", combined_features),
        ]:
            p(f"\n  --- @{label} {feat_set_name} ({len(feat_set)} features) ---")
            X, names, pair_idx = extract_pairs(feat_set, ki)
            rows, cols = zip(*pair_idx)
            sub_sim = cc_sim[np.ix_(ki, ki)]
            y = (sub_sim[rows, cols] > thresh).astype(np.float32)
            y_cont = sub_sim[rows, cols]
            auc = run_cv(X, y, y_cont, f"{label}_{feat_set_name}")
            results[f"pair_{feat_set_name}_{label}"] = auc

    p("\n" + "=" * 70)
    p("CC AUROC — COMPOUND-LEVEL CV")
    p("=" * 70)

    for feat_set_name, feat_set in [
        ("baseline", baseline_features),
        ("fusion_only", fusion_features),
        ("combined", combined_features),
    ]:
        p(f"\n  --- @0.6 {feat_set_name} (compound-level) ---")
        auc = run_compound_cv(feat_set, cc_sim, ki, 0.6)
        results[f"compound_{feat_set_name}_0.6"] = auc

    # Per-target
    pt_results = per_target_with_fusion(gt, gt_c, gt_g, n_c, n_g,
                                         fusion_data, valid_compounds, known_idx)
    results["per_target"] = pt_results

    # Summary
    p("\n" + "=" * 70)
    p("SUMMARY")
    p("=" * 70)
    for k, v in sorted(results.items()):
        if k != "per_target":
            p(f"  {k}: {v:.4f}")
    for k, v in results.get("per_target", {}).items():
        p(f"  per_target_{k}: median={v['median']:.4f} ({v['n_targets']} targets)")

    # Save
    import json
    out_path = os.path.join(RESULTS_DIR, "fusion_cls_results.json")
    json.dump(results, open(out_path, "w"), indent=2, default=str)
    p(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
