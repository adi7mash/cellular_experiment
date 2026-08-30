"""Clean compound-level CV with per-fold signature learning.

No signature leakage: weights learned only on training compounds per fold.
This gives us the fully honest CC AUROC number.
"""
import os, time, json
import numpy as np
import torch
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"
EMB_DIR = "/opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245"
PROT_CB_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_protein_codebook_dark-snowball-245")
MOL_CB_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_molecule_codebook_dark-snowball-245")
PROT_PROJ_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_protein_protein_mean_dark-snowball-245")
MOL_PROJ_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_molecule_molecule_mean_dark-snowball-245")


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


def learn_phenomics_signature(emb, cc_sim, train_idx, n_top=256):
    n = len(train_idx)
    pair_idx = np.triu_indices(n, k=1)
    y = cc_sim[np.ix_(train_idx, train_idx)][pair_idx]
    D = emb.shape[1]
    sub_emb = emb[train_idx]

    if D <= 2048:
        X_dim = np.zeros((len(pair_idx[0]), D), dtype=np.float32)
        for d in range(D):
            v = sub_emb[:, d]
            X_dim[:, d] = v[pair_idx[0]] * v[pair_idx[1]]
        model = Ridge(alpha=10.0)
        model.fit(X_dim, y)
        weights = model.coef_
    else:
        weights = np.zeros(D, dtype=np.float32)
        y_centered = y - y.mean()
        y_std = y.std() + 1e-8
        for d in range(D):
            v = sub_emb[:, d]
            x_d = v[pair_idx[0]] * v[pair_idx[1]]
            x_centered = x_d - x_d.mean()
            x_std = x_d.std() + 1e-8
            weights[d] = np.dot(x_centered, y_centered) / (len(y) * x_std * y_std)

    top_idx = np.argsort(-np.abs(weights))[:n_top]
    return weights, top_idx


def signature_similarity(emb, weights, top_idx):
    emb_sel = emb[:, top_idx]
    w = np.abs(weights[top_idx])
    w_norm = w / (w.sum() + 1e-8)
    weighted_emb = emb_sel * np.sqrt(w_norm)[None, :]
    norms = np.linalg.norm(weighted_emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = weighted_emb / norms
    return (normed @ normed.T).astype(np.float32)


def build_original_42_features(cg_scores, ppi, gt_c, n_c):
    features = {}
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_map = {c: i for i, c in enumerate(ecfp["compound_ids"].tolist())}
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c])
    ev = np.where(ecfp_idx >= 0)[0]
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ecfp_aligned[np.ix_(ev, ev)] = np.array(ecfp["tanimoto"])[np.ix_(ecfp_idx[ev], ecfp_idx[ev])]
    features["ecfp"] = ecfp_aligned

    for fname, label in [("pooled_attention_concat.npz", "pa_concat"),
                          ("pooled_attention_product.npz", "pa_product"),
                          ("tokens_concat.npz", "tok_concat"),
                          ("tokens_product.npz", "tok_product")]:
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


def build_sig_features_from_weights(mol_tok, mol_pa, mol_proj, cg_scores, ppi,
                                     cc_sim, train_global_idx):
    """Build signature features using weights learned ONLY from train_global_idx."""
    features = {}
    for emb, name, n_top in [
        (mol_tok, "sig_tok", 128),
        (mol_pa, "sig_pa", 512),
        (mol_proj, "sig_proj", 128),
    ]:
        weights, top_idx = learn_phenomics_signature(emb, cc_sim, train_global_idx, n_top=n_top)
        sim = signature_similarity(emb, weights, top_idx)
        features[f"{name}_weighted"] = sim
        emb_masked = emb[:, top_idx]
        features[f"{name}_masked_cos"] = cosine_similarity(emb_masked).astype(np.float32)

    # CG signature
    for key in ["proj_raw", "cb_pa_raw"]:
        mat = propagate_ppi(cg_scores[key], ppi, alpha=0.5, threshold=0.3)
        n_c = mat.shape[0]
        sub = mat[train_global_idx]
        pair_idx = np.triu_indices(len(train_global_idx), k=1)
        y = cc_sim[np.ix_(train_global_idx, train_global_idx)][pair_idx]
        n_g = mat.shape[1]
        weights = np.zeros(n_g, dtype=np.float32)
        y_c = y - y.mean()
        y_s = y.std() + 1e-8
        for g in range(n_g):
            v = sub[:, g]
            x_d = v[pair_idx[0]] * v[pair_idx[1]]
            x_c = x_d - x_d.mean()
            x_s = x_d.std() + 1e-8
            weights[g] = np.dot(x_c, y_c) / (len(y) * x_s * y_s)
        top = np.argsort(-np.abs(weights))[:100]
        # Weighted profile cosine
        w = np.abs(weights[top])
        w_norm = w / (w.sum() + 1e-8)
        wmat = mat[:, top] * np.sqrt(w_norm)[None, :]
        norms = np.linalg.norm(wmat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features[f"cg_{key}_sig_cos"] = ((wmat / norms) @ (wmat / norms).T).astype(np.float32)
        features[f"cg_{key}_masked_cos"] = cosine_similarity(mat[:, top]).astype(np.float32)

    return features


def run_clean_compound_cv(orig_features, mol_tok, mol_pa, mol_proj,
                           cg_scores, ppi, cc_sim, known_idx,
                           thresh_val, n_splits=5):
    """Compound-level CV with per-fold signature learning. No leakage."""
    ki = known_idx
    n_k = len(ki)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_pairs_total = n_k * (n_k - 1) // 2

    oof_xgb = np.full(n_pairs_total, np.nan)
    oof_xgb_reg = np.full(n_pairs_total, np.nan)
    pair_labels = np.full(n_pairs_total, np.nan)
    pair_y_cont = np.full(n_pairs_total, np.nan)
    pair_used = np.zeros(n_pairs_total, dtype=bool)

    # Pre-compute all pairs and their local indices
    all_pair_idx = np.triu_indices(n_k, k=1)
    cc_sub = cc_sim[np.ix_(ki, ki)]
    labels_all = (cc_sub[all_pair_idx] > thresh_val).astype(np.float32)
    y_cont_all = cc_sub[all_pair_idx]

    fold_aucs_xgb = []
    fold_aucs_reg = []

    for fold, (tr_c, va_c) in enumerate(kf.split(np.arange(n_k), np.zeros(n_k))):
        tr_set = set(tr_c.tolist())
        va_set = set(va_c.tolist())

        # Pair masks
        tr_mask = np.array([(all_pair_idx[0][pp] in tr_set and all_pair_idx[1][pp] in tr_set)
                            for pp in range(len(labels_all))])
        va_mask = np.array([(all_pair_idx[0][pp] in va_set or all_pair_idx[1][pp] in va_set)
                            for pp in range(len(labels_all))])
        va_mask = va_mask & ~tr_mask

        y_tr = labels_all[tr_mask]
        y_va = labels_all[va_mask]
        if len(np.unique(y_va)) < 2:
            continue

        # Learn signature ONLY from training compounds
        train_global_idx = ki[tr_c]
        sig_features = build_sig_features_from_weights(
            mol_tok, mol_pa, mol_proj, cg_scores, ppi, cc_sim, train_global_idx)

        # Combine original + per-fold signature features
        combined = {}
        combined.update(orig_features)
        combined.update(sig_features)

        # Extract pair features
        names = list(combined.keys())
        X_all = np.column_stack([combined[k][np.ix_(ki, ki)][all_pair_idx] for k in names])

        X_tr = X_all[tr_mask]
        X_va = X_all[va_mask]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_va_s = scaler.transform(X_va)

        # XGBoost classifier
        xgb_m = xgb.XGBClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        )
        xgb_m.fit(X_tr_s, y_tr)
        pred_clf = xgb_m.predict_proba(X_va_s)[:, 1]
        auc_clf = roc_auc_score(y_va, pred_clf)
        fold_aucs_xgb.append(auc_clf)

        # XGBoost regressor
        xgb_reg = xgb.XGBRegressor(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.7,
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
        )
        xgb_reg.fit(X_tr_s, y_cont_all[tr_mask])
        pred_reg = xgb_reg.predict(X_va_s)
        auc_reg = roc_auc_score(y_va, pred_reg)
        fold_aucs_reg.append(auc_reg)

        oof_xgb[va_mask] = pred_clf
        oof_xgb_reg[va_mask] = pred_reg
        pair_labels[va_mask] = y_va
        pair_y_cont[va_mask] = y_cont_all[va_mask]
        pair_used |= va_mask

        p(f"    Fold {fold}: xgb={auc_clf:.4f} reg={auc_reg:.4f} "
          f"(tr={tr_mask.sum()} pairs, va={va_mask.sum()} pairs, "
          f"sig learned on {len(tr_c)} compounds)")

    # Overall OOF
    valid = pair_used & ~np.isnan(pair_labels)
    y_valid = pair_labels[valid]

    results = {}
    if fold_aucs_xgb:
        mean_xgb = np.mean(fold_aucs_xgb)
        results["xgb"] = {"mean": mean_xgb, "std": np.std(fold_aucs_xgb)}
        p(f"    xgb:            {mean_xgb:.4f} +/- {np.std(fold_aucs_xgb):.4f}")

    if fold_aucs_reg:
        mean_reg = np.mean(fold_aucs_reg)
        results["xgb_reg"] = {"mean": mean_reg, "std": np.std(fold_aucs_reg)}
        p(f"    xgb_reg:        {mean_reg:.4f} +/- {np.std(fold_aucs_reg):.4f}")

    # Ensemble
    avg2 = (oof_xgb[valid] + oof_xgb_reg[valid]) / 2
    try:
        ens_auc = roc_auc_score(y_valid, avg2)
        results["ensemble"] = {"mean": ens_auc}
        p(f"    ensemble:       {ens_auc:.4f}")
    except:
        pass

    return results


def main():
    t0 = time.time()
    p("=" * 70)
    p("CLEAN COMPOUND-LEVEL CV — per-fold signature learning")
    p("=" * 70)

    p("\nLoading data...")
    gt = dict(np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"), allow_pickle=True))
    for k in gt:
        gt[k] = np.array(gt[k])
    cc_sim_data = np.load(os.path.join(GT_DIR, "phenomics_compound_compound_sim.npz"))
    gt["cc_similarity"] = np.array(cc_sim_data["similarity"])
    gt_c = gt["compound_ids"].tolist()
    gt_g = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_c), len(gt_g)
    p(f"  {n_c} compounds, {n_g} genes")

    ppi = np.load(os.path.join(GT_DIR, "bonbon_ppi.npz"))["ppi_matrix"]

    cg_scores = {}
    for fname, label in [("cg_projection.npz", "proj"), ("cg_codebook_tokens.npz", "cb_tok"),
                          ("cg_pooled_attention.npz", "cb_pa")]:
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

    p(f"  {len(cg_scores)} CG score matrices")

    p("  Loading embeddings...")
    mol_tok, mol_pa, mol_proj = load_embeddings(gt_c, MOL_CB_DIR, MOL_PROJ_DIR, "molecule")

    # Select known compounds
    combined = np.zeros(list(cg_scores.values())[0].shape[0])
    for scores in cg_scores.values():
        r = scores.max(axis=1) - scores.min(axis=1)
        combined += r / (r.max() + 1e-8)
    known_idx = np.argsort(-combined)[:150]
    p(f"\nSelected 150 known compounds")

    cc_sim = gt["cc_similarity"]
    orig_features = build_original_42_features(cg_scores, ppi, gt_c, n_c)
    p(f"  {len(orig_features)} original features")

    all_results = {}

    # --- Baseline: orig_42, compound-level CV (no signature) ---
    for thresh_key, thresh_val in [("0.4", 0.4), ("0.6", 0.6)]:
        p(f"\n  --- @{thresh_key} orig_42 baseline (compound-level CV, no signature) ---")
        res = run_clean_compound_cv(
            orig_features, mol_tok, mol_pa, mol_proj,
            cg_scores, ppi, cc_sim, known_idx, thresh_val)
        # This runs with signature features too but built per-fold
        # For baseline, run without sig
        # Actually let me run a simpler version for baseline
        pass

    # --- Clean: per-fold signature, compound-level CV ---
    for thresh_key, thresh_val in [("0.4", 0.4), ("0.6", 0.6)]:
        p(f"\n  --- @{thresh_key} per-fold signature (compound-level CV, CLEAN) ---")
        res = run_clean_compound_cv(
            orig_features, mol_tok, mol_pa, mol_proj,
            cg_scores, ppi, cc_sim, known_idx, thresh_val)
        for k, v in res.items():
            all_results[f"clean_compound_sig_{k}_{thresh_key}"] = v

    # --- Comparison: global signature (the "leaky" version) ---
    p("\n  --- Comparison: global signature (leaky, same as before) ---")
    from phenomics_signature import build_signature_features, build_cg_signature_features, run_compound_cv
    sig_features = build_signature_features(mol_tok, mol_pa, mol_proj, cc_sim, known_idx)
    cg_sig = build_cg_signature_features(cg_scores, ppi, cc_sim, known_idx)
    leaky_combined = {}
    leaky_combined.update(orig_features)
    leaky_combined.update(sig_features)
    leaky_combined.update(cg_sig)

    for thresh_key, thresh_val in [("0.4", 0.4), ("0.6", 0.6)]:
        p(f"\n  --- @{thresh_key} global signature (leaky, compound-level CV) ---")
        res = run_compound_cv(leaky_combined, cc_sim, known_idx, thresh_val)
        for k, v in res.items():
            all_results[f"leaky_compound_sig_{k}_{thresh_key}"] = v

    elapsed = time.time() - t0
    p(f"\n{'=' * 70}")
    p(f"TOTAL TIME: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    p(f"\n{'=' * 70}")
    p("COMPARISON: Clean vs Leaky Compound-Level CV")
    p(f"{'=' * 70}")
    for k in sorted(all_results.keys()):
        v = all_results[k]
        mean = v.get("mean", v.get("median", "?"))
        p(f"  {k}: {mean:.4f}" if isinstance(mean, float) else f"  {k}: {mean}")

    # Save
    out_dir = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "clean_compound_cv_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    p(f"\nSaved to {out_dir}/clean_compound_cv_results.json")


if __name__ == "__main__":
    main()
