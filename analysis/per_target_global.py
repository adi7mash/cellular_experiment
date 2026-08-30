"""Global compound-target model for per-target AUROC > 60%.

Instead of picking one CG config per target, train a single model on ALL
compound-target pairs that learns which CG features matter for which targets.

Target-level CV: split targets into folds, train on (all compounds × train targets),
predict on (all compounds × val targets). Clean — no target info leakage.
"""
import numpy as np
import xgboost as xgb
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"
EMB_DIR = "/opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245"
PROT_CB_DIR = f"{EMB_DIR}/rxrx3_pairs_protein_codebook_dark-snowball-245"
MOL_CB_DIR = f"{EMB_DIR}/rxrx3_pairs_molecule_codebook_dark-snowball-245"
PROT_PROJ_DIR = f"{EMB_DIR}/rxrx3_pairs_protein_protein_mean_dark-snowball-245"
MOL_PROJ_DIR = f"{EMB_DIR}/rxrx3_pairs_molecule_molecule_mean_dark-snowball-245"
import os


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


def load_embeddings(ids, cb_dir, proj_dir, entity_type="protein"):
    tok_dir = os.path.join(cb_dir, "tokens")
    pa_dir = os.path.join(cb_dir, "pooled_attention")
    tok_files = {os.path.splitext(f)[0]: f for f in os.listdir(tok_dir) if f.endswith(".pt")}
    pa_files = {os.path.splitext(f)[0]: f for f in os.listdir(pa_dir) if f.endswith(".pt")}
    proj_files = {os.path.splitext(f)[0]: f for f in os.listdir(proj_dir) if f.endswith(".pt")}

    n = len(ids)
    sample = torch.load(os.path.join(tok_dir, list(tok_files.values())[0]), map_location="cpu")
    tok_dim = sample.shape[0]
    sample = torch.load(os.path.join(pa_dir, list(pa_files.values())[0]), map_location="cpu")
    pa_dim = sample.shape[0]
    sample = torch.load(os.path.join(proj_dir, list(proj_files.values())[0]), map_location="cpu")
    proj_dim = sample.shape[0]

    tok_embs = np.zeros((n, tok_dim), dtype=np.float32)
    pa_embs = np.zeros((n, pa_dim), dtype=np.float32)
    proj_embs = np.zeros((n, proj_dim), dtype=np.float32)
    matched = 0
    for i, eid in enumerate(ids):
        if eid in tok_files:
            tok_embs[i] = torch.load(os.path.join(tok_dir, tok_files[eid]), map_location="cpu").float().numpy()
            pa_embs[i] = torch.load(os.path.join(pa_dir, pa_files[eid]), map_location="cpu").float().numpy()
            proj_embs[i] = torch.load(os.path.join(proj_dir, proj_files[eid]), map_location="cpu").float().numpy()
            matched += 1
    p(f"  {entity_type}: {matched}/{n} matched")
    return tok_embs, pa_embs, proj_embs


def build_ct_features(cg_scores, ppi, compound_idx, target_idx,
                      prot_pca, mol_pca, prot_stats, mol_stats):
    """Build features for compound-target pairs.

    For each (compound i, target g):
      - 5 raw CG scores
      - 5 PPI-propagated CG scores (alpha=0.5, thresh=0.3)
      - 5 PPI-propagated CG scores (alpha=0.7, thresh=0.0)
      - protein PCA (16 dims)
      - compound PCA (16 dims)
      - CG score statistics per target (mean, std, max, p90) per key
      - CG score statistics per compound
    """
    n_c = len(compound_idx)
    n_g = len(target_idx)
    n_pairs = n_c * n_g

    cg_keys = ["proj_raw", "proj_cos", "cb_pa_raw", "cb_pa_cos", "cb_tok_raw"]

    # CG score features
    feats = []
    feat_names = []

    for key in cg_keys:
        mat = cg_scores[key][np.ix_(compound_idx, target_idx)]
        feats.append(mat.reshape(-1, 1))
        feat_names.append(f"{key}")

    for key in cg_keys:
        mat = propagate_ppi(cg_scores[key], ppi, alpha=0.5, threshold=0.3)
        feats.append(mat[np.ix_(compound_idx, target_idx)].reshape(-1, 1))
        feat_names.append(f"{key}_ppi05")

    for key in cg_keys:
        mat = propagate_ppi(cg_scores[key], ppi, alpha=0.7, threshold=0.0)
        feats.append(mat[np.ix_(compound_idx, target_idx)].reshape(-1, 1))
        feat_names.append(f"{key}_ppi07")

    # Protein PCA features (repeated per compound)
    prot_feats = prot_pca[target_idx]  # (n_g, d_prot)
    prot_tiled = np.tile(prot_feats, (n_c, 1))  # (n_c*n_g, d_prot)
    feats.append(prot_tiled)
    for d in range(prot_feats.shape[1]):
        feat_names.append(f"prot_pca{d}")

    # Compound PCA features (repeated per target)
    mol_feats = mol_pca[compound_idx]  # (n_c, d_mol)
    mol_tiled = np.repeat(mol_feats, n_g, axis=0)  # (n_c*n_g, d_mol)
    feats.append(mol_tiled)
    for d in range(mol_feats.shape[1]):
        feat_names.append(f"mol_pca{d}")

    # Per-target CG statistics (how active is this target across all compounds?)
    prot_stat_feats = prot_stats[target_idx]
    prot_stat_tiled = np.tile(prot_stat_feats, (n_c, 1))
    feats.append(prot_stat_tiled)
    for d in range(prot_stat_feats.shape[1]):
        feat_names.append(f"prot_stat{d}")

    # Per-compound CG statistics
    mol_stat_feats = mol_stats[compound_idx]
    mol_stat_tiled = np.repeat(mol_stat_feats, n_g, axis=0)
    feats.append(mol_stat_tiled)
    for d in range(mol_stat_feats.shape[1]):
        feat_names.append(f"mol_stat{d}")

    X = np.hstack(feats)
    return X, feat_names


def main():
    p("Loading data...")
    gt = dict(np.load(f"{GT_DIR}/ground_truth_binary.npz", allow_pickle=True))
    for k in gt:
        gt[k] = np.array(gt[k])
    gt_c = gt["compound_ids"].tolist()
    gt_g = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_c), len(gt_g)
    p(f"  {n_c} compounds, {n_g} genes")

    ppi = np.load(f"{GT_DIR}/bonbon_ppi.npz")["ppi_matrix"]
    cg_binary = gt["cg_binary_top1"]

    cg_scores = {}
    for fname, label in [("cg_projection.npz", "proj"), ("cg_codebook_tokens.npz", "cb_tok"),
                          ("cg_pooled_attention.npz", "cb_pa")]:
        data = np.load(f"{IP_DIR}/{fname}")
        c_idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        g_idx = align_to_gt(data["protein_ids"].tolist(), gt_g)
        vc, vg = c_idx >= 0, g_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat = np.array(data[sk])
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat[np.ix_(c_idx[vc], g_idx[vg])]
            cg_scores[f"{label}_{suffix}"] = aligned

    for fname, label in [("binary_projection.npz", "bin_proj"), ("binary_codebook_tokens.npz", "bin_tok")]:
        data = np.load(f"{IP_DIR}/{fname}")
        c_idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        g_idx = align_to_gt(data["protein_ids"].tolist(), gt_g)
        vc, vg = c_idx >= 0, g_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat_T = np.array(data[sk]).T
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat_T[np.ix_(c_idx[vc], g_idx[vg])]
            cg_scores[f"{label}_{suffix}"] = aligned
    p(f"  {len(cg_scores)} CG score matrices")

    p("  Loading embeddings...")
    mol_tok, mol_pa, mol_proj = load_embeddings(gt_c, MOL_CB_DIR, MOL_PROJ_DIR, "molecule")
    prot_tok, prot_pa, prot_proj = load_embeddings(gt_g, PROT_CB_DIR, PROT_PROJ_DIR, "protein")

    # PCA of embeddings
    prot_all = np.hstack([prot_tok, prot_pa, prot_proj])
    mol_all = np.hstack([mol_tok, mol_pa, mol_proj])
    prot_pca = PCA(n_components=16, random_state=42).fit_transform(prot_all)
    mol_pca = PCA(n_components=16, random_state=42).fit_transform(mol_all)
    p(f"  PCA: prot={prot_pca.shape}, mol={mol_pca.shape}")

    # Per-entity CG statistics
    cg_keys = ["proj_raw", "proj_cos", "cb_pa_raw", "cb_pa_cos", "cb_tok_raw"]
    prot_stats_list = []
    mol_stats_list = []
    for key in cg_keys:
        mat = cg_scores[key]
        prot_stats_list.append(mat.mean(axis=0).reshape(-1, 1))
        prot_stats_list.append(mat.std(axis=0).reshape(-1, 1))
        mol_stats_list.append(mat.mean(axis=1).reshape(-1, 1))
        mol_stats_list.append(mat.std(axis=1).reshape(-1, 1))
    prot_stats = np.hstack(prot_stats_list)
    mol_stats = np.hstack(mol_stats_list)
    p(f"  Stats: prot={prot_stats.shape}, mol={mol_stats.shape}")

    # Select known compounds
    combined = np.zeros(n_c)
    for scores in cg_scores.values():
        r = scores.max(axis=1) - scores.min(axis=1)
        combined += r / (r.max() + 1e-8)
    known_idx = np.argsort(-combined)[:150]
    p(f"  Selected 150 known compounds")

    # =====================================================================
    # EXPERIMENT 1: Target-level CV with global model
    # =====================================================================
    p("\n" + "=" * 70)
    p("Global compound-target model (target-level CV)")
    p("=" * 70)

    # Find valid targets (enough positives)
    valid_targets = []
    for g in range(n_g):
        y = cg_binary[known_idx, g]
        if y.sum() >= 2 and y.sum() <= len(known_idx) - 2:
            valid_targets.append(g)
    valid_targets = np.array(valid_targets)
    p(f"  {len(valid_targets)} valid targets")

    target_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_per_target_aucs = {}

    for fold, (tr_tgt, va_tgt) in enumerate(target_kf.split(valid_targets)):
        tr_targets = valid_targets[tr_tgt]
        va_targets = valid_targets[va_tgt]
        p(f"\n  Fold {fold}: {len(tr_targets)} train targets, {len(va_targets)} val targets")

        # Build training data
        X_tr, feat_names = build_ct_features(
            cg_scores, ppi, known_idx, tr_targets,
            prot_pca, mol_pca, prot_stats, mol_stats)
        y_tr = cg_binary[np.ix_(known_idx, tr_targets)].reshape(-1)
        p(f"    Train: {X_tr.shape}, pos_rate={y_tr.mean():.4f}")

        # Build val data
        X_va, _ = build_ct_features(
            cg_scores, ppi, known_idx, va_targets,
            prot_pca, mol_pca, prot_stats, mol_stats)
        y_va = cg_binary[np.ix_(known_idx, va_targets)].reshape(-1)
        p(f"    Val: {X_va.shape}, pos_rate={y_va.mean():.4f}")

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # XGBoost
        model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
        )
        model.fit(X_tr_s, y_tr)
        preds = model.predict_proba(X_va_s)[:, 1]

        # Reshape predictions to (n_compounds, n_val_targets)
        pred_matrix = preds.reshape(len(known_idx), len(va_targets))

        # Per-target AUROC
        fold_aucs = []
        for gi, g in enumerate(va_targets):
            y_g = cg_binary[known_idx, g]
            if y_g.sum() < 2 or y_g.sum() > len(known_idx) - 2:
                continue
            try:
                auc = roc_auc_score(y_g, pred_matrix[:, gi])
                fold_aucs.append(auc)
                all_per_target_aucs[g] = auc
            except:
                pass

        fold_median = np.median(fold_aucs) if fold_aucs else 0
        p(f"    Per-target AUROC: median={fold_median:.4f} ({len(fold_aucs)} targets)")

    # Overall results
    all_aucs = list(all_per_target_aucs.values())
    global_median = np.median(all_aucs)
    p(f"\n  GLOBAL MODEL: median per-target AUROC = {global_median:.4f} ({len(all_aucs)} targets)")
    p(f"    >0.6: {sum(a > 0.6 for a in all_aucs)}, >0.7: {sum(a > 0.7 for a in all_aucs)}")

    # =====================================================================
    # EXPERIMENT 2: Baseline comparison (best single config, same valid targets)
    # =====================================================================
    p("\n" + "=" * 70)
    p("Baseline: best single config on same targets")
    p("=" * 70)

    baseline_configs = []
    for key in cg_keys:
        for alpha in [0.0, 0.3, 0.5, 0.7]:
            for thresh in [0.0, 0.3, 0.5]:
                if alpha == 0.0 and thresh > 0:
                    continue
                baseline_configs.append((key, alpha, thresh))
                if alpha == 0.0:
                    break

    best_baseline_median = 0
    best_baseline_config = None
    for key, alpha, thresh in baseline_configs:
        mat = cg_scores[key]
        if alpha > 0:
            mat = propagate_ppi(mat, ppi, alpha, thresh)
        scores_sub = mat[known_idx]
        aucs = []
        for g in valid_targets:
            y = cg_binary[known_idx, g]
            if y.sum() < 2 or y.sum() > len(known_idx) - 2:
                continue
            try:
                aucs.append(roc_auc_score(y, scores_sub[:, g]))
            except:
                pass
        if aucs:
            med = np.median(aucs)
            if med > best_baseline_median:
                best_baseline_median = med
                best_baseline_config = (key, alpha, thresh)

    p(f"  Best single config: {best_baseline_config} median={best_baseline_median:.4f}")

    # Oracle
    per_target_all = np.full((n_g, len(baseline_configs)), np.nan)
    for ci, (key, alpha, thresh) in enumerate(baseline_configs):
        mat = cg_scores[key]
        if alpha > 0:
            mat = propagate_ppi(mat, ppi, alpha, thresh)
        scores_sub = mat[known_idx]
        for g in valid_targets:
            y = cg_binary[known_idx, g]
            if y.sum() < 2 or y.sum() > len(known_idx) - 2:
                continue
            try:
                per_target_all[g, ci] = roc_auc_score(y, scores_sub[:, g])
            except:
                pass
    oracle_aucs = np.nanmax(per_target_all[valid_targets], axis=1)
    oracle_valid = ~np.isnan(oracle_aucs)
    oracle_median = np.median(oracle_aucs[oracle_valid])
    p(f"  Oracle: median={oracle_median:.4f} ({oracle_valid.sum()} targets)")

    # =====================================================================
    # EXPERIMENT 3: Compound-level CV within global model (honest)
    # =====================================================================
    p("\n" + "=" * 70)
    p("Global model with compound-level CV (honest)")
    p("=" * 70)

    compound_kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cmpd_per_target_preds = np.zeros((len(known_idx), len(valid_targets)))

    for fold, (tr_c, va_c) in enumerate(compound_kf.split(np.arange(len(known_idx)))):
        tr_compounds = known_idx[tr_c]
        va_compounds = known_idx[va_c]
        p(f"\n  Fold {fold}: {len(tr_compounds)} train compounds, {len(va_compounds)} val compounds")

        X_tr, _ = build_ct_features(
            cg_scores, ppi, tr_compounds, valid_targets,
            prot_pca, mol_pca, prot_stats, mol_stats)
        y_tr = cg_binary[np.ix_(tr_compounds, valid_targets)].reshape(-1)

        X_va, _ = build_ct_features(
            cg_scores, ppi, va_compounds, valid_targets,
            prot_pca, mol_pca, prot_stats, mol_stats)

        scaler = StandardScaler().fit(X_tr)

        model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
        )
        model.fit(scaler.transform(X_tr), y_tr)
        preds = model.predict_proba(scaler.transform(X_va))[:, 1]
        pred_matrix = preds.reshape(len(va_compounds), len(valid_targets))
        cmpd_per_target_preds[va_c] = pred_matrix

    # Evaluate per-target with all OOF compound predictions
    cmpd_aucs = []
    for gi, g in enumerate(valid_targets):
        y = cg_binary[known_idx, g]
        if y.sum() < 2 or y.sum() > len(known_idx) - 2:
            continue
        try:
            cmpd_aucs.append(roc_auc_score(y, cmpd_per_target_preds[:, gi]))
        except:
            pass

    cmpd_median = np.median(cmpd_aucs) if cmpd_aucs else 0
    p(f"\n  COMPOUND-CV GLOBAL MODEL: median={cmpd_median:.4f} ({len(cmpd_aucs)} targets)")
    p(f"    >0.6: {sum(a > 0.6 for a in cmpd_aucs)}, >0.7: {sum(a > 0.7 for a in cmpd_aucs)}")

    # =====================================================================
    # EXPERIMENT 4: Hybrid — global model score as extra CG feature
    # =====================================================================
    p("\n" + "=" * 70)
    p("Hybrid: original configs + global model score")
    p("=" * 70)

    # Add global model OOF predictions as an extra "config"
    hybrid_aucs = []
    for gi, g in enumerate(valid_targets):
        y = cg_binary[known_idx, g]
        if y.sum() < 2 or y.sum() > len(known_idx) - 2:
            continue

        # All baseline config scores for this target
        target_scores = []
        for ci, (key, alpha, thresh) in enumerate(baseline_configs):
            s = per_target_all[g, ci]
            if not np.isnan(s):
                target_scores.append(s)

        # Global model score
        try:
            global_auc = roc_auc_score(y, cmpd_per_target_preds[:, gi])
        except:
            global_auc = 0.5

        # Use max of global model and best baseline
        best_baseline = max(target_scores) if target_scores else 0.5
        hybrid_aucs.append(max(global_auc, best_baseline))

    hybrid_median = np.median(hybrid_aucs) if hybrid_aucs else 0
    p(f"  Hybrid (max of global + oracle): median={hybrid_median:.4f} ({len(hybrid_aucs)} targets)")

    # Also: average of global model + best single config
    p("\n  Weighted blends:")
    for w_global in [0.3, 0.5, 0.7]:
        blend_aucs = []
        best_key, best_alpha, best_thresh = best_baseline_config
        best_mat = cg_scores[best_key]
        if best_alpha > 0:
            best_mat = propagate_ppi(best_mat, ppi, best_alpha, best_thresh)

        for gi, g in enumerate(valid_targets):
            y = cg_binary[known_idx, g]
            if y.sum() < 2 or y.sum() > len(known_idx) - 2:
                continue
            baseline_scores = best_mat[known_idx, g]
            global_scores = cmpd_per_target_preds[:, gi]
            # Normalize both to [0,1] range
            b_min, b_max = baseline_scores.min(), baseline_scores.max()
            g_min, g_max = global_scores.min(), global_scores.max()
            if b_max > b_min:
                baseline_norm = (baseline_scores - b_min) / (b_max - b_min)
            else:
                baseline_norm = np.zeros_like(baseline_scores)
            if g_max > g_min:
                global_norm = (global_scores - g_min) / (g_max - g_min)
            else:
                global_norm = np.zeros_like(global_scores)
            blended = w_global * global_norm + (1 - w_global) * baseline_norm
            try:
                blend_aucs.append(roc_auc_score(y, blended))
            except:
                pass
        if blend_aucs:
            p(f"    w_global={w_global}: median={np.median(blend_aucs):.4f} ({len(blend_aucs)} targets)")

    # Summary
    p(f"\n{'=' * 70}")
    p("SUMMARY")
    p(f"{'=' * 70}")
    p(f"  Best single config: {best_baseline_config} median={best_baseline_median:.4f}")
    p(f"  Oracle: median={oracle_median:.4f}")
    p(f"  Global model (target-CV): median={global_median:.4f}")
    p(f"  Global model (compound-CV): median={cmpd_median:.4f}")
    p(f"  Hybrid (max): median={hybrid_median:.4f}")


if __name__ == "__main__":
    main()
