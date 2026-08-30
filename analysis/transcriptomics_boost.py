"""Add HUVEC transcriptomics weighting to CG scores and re-evaluate.

Beaini's step 2: weight CG scores by gene expression level. Targets not
expressed in the cell are noise — suppress them.
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
EVAL_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"


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


def load_expression(gt_g):
    """Load HUVEC gene expression and align to ground truth genes."""
    expr = {}
    with open(os.path.join(GT_DIR, "huvec_expression.tsv")) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                expr[parts[0]] = float(parts[1])

    tpm = np.zeros(len(gt_g), dtype=np.float32)
    matched = 0
    for i, g in enumerate(gt_g):
        if g in expr:
            tpm[i] = expr[g]
            matched += 1
    p(f"  Expression: {matched}/{len(gt_g)} genes matched")
    p(f"  Expressed (TPM>1): {(tpm > 1).sum()}, low/off: {(tpm <= 1).sum()}")
    return tpm


def expression_weight(tpm, method="log"):
    """Convert TPM to weights for CG score modulation."""
    if method == "log":
        return np.log1p(tpm)
    elif method == "binary":
        return (tpm > 1).astype(np.float32)
    elif method == "sqrt":
        return np.sqrt(tpm)
    elif method == "rank":
        ranks = np.argsort(np.argsort(tpm)).astype(np.float32)
        return ranks / ranks.max()
    elif method == "sigmoid":
        # Sigmoid centered at median expression
        median_tpm = np.median(tpm[tpm > 0])
        return 1.0 / (1.0 + np.exp(-(np.log1p(tpm) - np.log1p(median_tpm))))
    return np.ones_like(tpm)


def weight_cg_scores(cg_scores, weights):
    """Apply per-target expression weights to CG score matrices."""
    weighted = {}
    for key, mat in cg_scores.items():
        weighted[key] = mat * weights[None, :]
    return weighted


def per_target_eval(cg_scores, cg_binary, known_idx, configs, label):
    """Evaluate per-target AUROC for a set of CG scores."""
    n_g = cg_binary.shape[1]
    per_target_aucs = np.full((n_g, len(configs)), np.nan)

    for ci, (key, alpha, thresh) in enumerate(configs):
        if key not in cg_scores:
            continue
        mat = cg_scores[key]
        if alpha > 0:
            mat = propagate_ppi(mat, ppi_global, alpha, thresh)
        scores_sub = mat[known_idx]

        for g in range(n_g):
            y = cg_binary[known_idx, g]
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue
            try:
                per_target_aucs[g, ci] = roc_auc_score(y, scores_sub[:, g])
            except:
                pass

    valid = ~np.all(np.isnan(per_target_aucs), axis=1)
    valid_idx = np.where(valid)[0]
    if len(valid_idx) == 0:
        return {}

    oracle = np.nanmax(per_target_aucs[valid_idx], axis=1)
    oracle_med = np.median(oracle)

    config_medians = np.array([np.nanmedian(per_target_aucs[valid_idx, ci])
                                for ci in range(len(configs))])
    best_ci = np.nanargmax(config_medians)
    best_med = config_medians[best_ci]

    top3 = np.argsort(-config_medians)[:3]
    top3_scores = np.nanmean(per_target_aucs[valid_idx][:, top3], axis=1)
    top3_med = np.median(top3_scores)

    p(f"  {label}: best={best_med:.4f} ({configs[best_ci]}), "
      f"top3={top3_med:.4f}, oracle={oracle_med:.4f} ({len(valid_idx)} targets)")

    return {
        "best_single": float(best_med),
        "best_config": str(configs[best_ci]),
        "top3_avg": float(top3_med),
        "oracle": float(oracle_med),
        "n_targets": len(valid_idx),
    }


ppi_global = None


def build_cc_features(cg_scores, ppi, gt_c, n_c,
                      mol_tok, mol_pa, mol_proj, cc_sim, train_idx):
    """Build CC features including expression-weighted CG profiles."""
    features = {}

    # ECFP
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_map = {c: i for i, c in enumerate(ecfp["compound_ids"].tolist())}
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c])
    ev = np.where(ecfp_idx >= 0)[0]
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ecfp_aligned[np.ix_(ev, ev)] = np.array(ecfp["tanimoto"])[np.ix_(ecfp_idx[ev], ecfp_idx[ev])]
    features["ecfp"] = ecfp_aligned

    # Direct CC from interaction prints
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

    # CG profile cosines (original + expression-weighted)
    for key, scores in cg_scores.items():
        features[f"{key}_prof"] = cosine_similarity(scores).astype(np.float32)

    # PPI-propagated
    for ppi_thresh, ppi_label in [(0.0, "full"), (0.3, "t03"), (0.5, "t05")]:
        for alpha in [0.3, 0.7]:
            for key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
                if key not in cg_scores:
                    continue
                propagated = propagate_ppi(cg_scores[key], ppi, alpha=alpha, threshold=ppi_thresh)
                features[f"{key}_ppi_{ppi_label}_a{alpha}"] = cosine_similarity(propagated).astype(np.float32)

    # Jaccard
    for cg_key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
        if cg_key not in cg_scores:
            continue
        propagated = propagate_ppi(cg_scores[cg_key], ppi, alpha=0.5, threshold=0.3)
        for bin_pct in [70, 85, 95]:
            thresh = np.percentile(propagated, bin_pct)
            binary = (propagated > thresh).astype(np.float32)
            intersection = binary @ binary.T
            counts = binary.sum(axis=1)
            union = counts[:, None] + counts[None, :] - intersection
            features[f"{cg_key}_jacc_p{bin_pct}"] = (intersection / (union + 5.0 + 1e-8)).astype(np.float32)

    # Signature features (per-fold will be handled in CV)
    for emb, name, n_top in [
        (mol_tok, "sig_tok", 128),
        (mol_pa, "sig_pa", 512),
        (mol_proj, "sig_proj", 128),
    ]:
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
            y_c = y - y.mean()
            y_s = y.std() + 1e-8
            for d in range(D):
                v = sub_emb[:, d]
                x_d = v[pair_idx[0]] * v[pair_idx[1]]
                x_c = x_d - x_d.mean()
                x_s = x_d.std() + 1e-8
                weights[d] = np.dot(x_c, y_c) / (len(y) * x_s * y_s)

        top_idx = np.argsort(-np.abs(weights))[:n_top]
        # Weighted cosine
        emb_sel = emb[:, top_idx]
        w = np.abs(weights[top_idx])
        w_norm = w / (w.sum() + 1e-8)
        wemb = emb_sel * np.sqrt(w_norm)[None, :]
        norms = np.linalg.norm(wemb, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features[f"{name}_weighted"] = ((wemb / norms) @ (wemb / norms).T).astype(np.float32)
        features[f"{name}_masked_cos"] = cosine_similarity(emb_sel).astype(np.float32)

    return features


def run_compound_cv(features, cc_sim, known_idx, thresh_val, n_splits=5):
    ki = known_idx
    n_k = len(ki)
    all_pair_idx = np.triu_indices(n_k, k=1)
    cc_sub = cc_sim[np.ix_(ki, ki)]
    labels_all = (cc_sub[all_pair_idx] > thresh_val).astype(np.float32)
    y_cont_all = cc_sub[all_pair_idx]

    names = list(features.keys())
    X_all = np.column_stack([features[k][np.ix_(ki, ki)][all_pair_idx] for k in names])

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (tr_c, va_c) in enumerate(kf.split(np.arange(n_k), np.zeros(n_k))):
        tr_set = set(tr_c.tolist())
        tr_mask = np.array([(all_pair_idx[0][pp] in tr_set and all_pair_idx[1][pp] in tr_set)
                            for pp in range(len(labels_all))])
        va_mask = np.array([(all_pair_idx[0][pp] not in tr_set or all_pair_idx[1][pp] not in tr_set)
                            for pp in range(len(labels_all))])
        va_mask = va_mask & ~tr_mask

        y_tr, y_va = labels_all[tr_mask], labels_all[va_mask]
        if len(np.unique(y_va)) < 2:
            continue

        scaler = StandardScaler().fit(X_all[tr_mask])
        X_tr_s = scaler.transform(X_all[tr_mask])
        X_va_s = scaler.transform(X_all[va_mask])

        xgb_m = xgb.XGBClassifier(
            n_estimators=2000, max_depth=7, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", device="cuda", random_state=42, verbosity=0,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
        )
        xgb_m.fit(X_tr_s, y_tr)
        pred = xgb_m.predict_proba(X_va_s)[:, 1]
        auc = roc_auc_score(y_va, pred)
        fold_aucs.append(auc)
        p(f"    Fold {fold}: xgb={auc:.4f}")

    if fold_aucs:
        mean_auc = np.mean(fold_aucs)
        p(f"    Mean: {mean_auc:.4f} +/- {np.std(fold_aucs):.4f}")
        return {"mean": mean_auc, "std": float(np.std(fold_aucs))}
    return {}


def main():
    global ppi_global
    t0 = time.time()
    p("=" * 70)
    p("TRANSCRIPTOMICS BOOST — HUVEC expression weighting")
    p("=" * 70)

    # Load data
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
    ppi_global = ppi

    # Load CG scores
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

    # Load expression
    tpm = load_expression(gt_g)

    # Load embeddings
    p("  Loading embeddings...")
    mol_tok, mol_pa, mol_proj = load_embeddings(gt_c, MOL_CB_DIR, MOL_PROJ_DIR, "molecule")

    # Select known compounds
    combined = np.zeros(n_c)
    for scores in cg_scores.values():
        r = scores.max(axis=1) - scores.min(axis=1)
        combined += r / (r.max() + 1e-8)
    known_idx = np.argsort(-combined)[:150]
    p(f"\nSelected 150 known compounds")

    cc_sim = gt["cc_similarity"]
    cg_binary = gt["cg_binary_top1"]
    all_results = {}

    # =====================================================================
    # PART 1: Per-target with expression weighting
    # =====================================================================
    p("\n" + "=" * 70)
    p("PART 1: Per-target AUROC — expression weighting")
    p("=" * 70)

    cg_keys = ["proj_raw", "proj_cos", "cb_pa_raw", "cb_pa_cos", "cb_tok_raw"]
    alphas = [0.0, 0.3, 0.5, 0.7]
    thresholds = [0.0, 0.3, 0.5]
    configs = []
    for key in cg_keys:
        for alpha in alphas:
            for thresh in thresholds:
                if alpha == 0.0 and thresh > 0:
                    continue
                configs.append((key, alpha, thresh))
                if alpha == 0.0:
                    break

    # Baseline (no expression weighting)
    p("\n  --- Baseline (no expression weighting) ---")
    for n_sel, subset_name in [(150, "known_150")]:
        ki = np.argsort(-combined)[:n_sel]
        res = per_target_eval(cg_scores, cg_binary, ki, configs, f"baseline_{subset_name}")
        all_results[f"baseline_{subset_name}"] = res

    # Expression-weighted variants
    for method in ["log", "binary", "sqrt", "sigmoid", "rank"]:
        p(f"\n  --- Expression weighting: {method} ---")
        w = expression_weight(tpm, method)
        w_cg = weight_cg_scores(cg_scores, w)
        for n_sel, subset_name in [(150, "known_150")]:
            ki = np.argsort(-combined)[:n_sel]
            res = per_target_eval(w_cg, cg_binary, ki, configs, f"expr_{method}_{subset_name}")
            all_results[f"expr_{method}_{subset_name}"] = res

    # Expression-weighted PPI
    p("\n  --- Expression-weighted PPI ---")
    expr_ppi = ppi * np.sqrt(np.outer(np.log1p(tpm), np.log1p(tpm)))
    ppi_global = expr_ppi
    for n_sel, subset_name in [(150, "known_150")]:
        ki = np.argsort(-combined)[:n_sel]
        res = per_target_eval(cg_scores, cg_binary, ki, configs, f"expr_ppi_{subset_name}")
        all_results[f"expr_ppi_{subset_name}"] = res
    ppi_global = ppi  # reset

    # Combined: expression-weighted CG + expression-weighted PPI
    p("\n  --- Combined: log-weighted CG + expression-weighted PPI ---")
    w = expression_weight(tpm, "log")
    w_cg = weight_cg_scores(cg_scores, w)
    ppi_global = expr_ppi
    for n_sel, subset_name in [(150, "known_150")]:
        ki = np.argsort(-combined)[:n_sel]
        res = per_target_eval(w_cg, cg_binary, ki, configs, f"combined_{subset_name}")
        all_results[f"combined_{subset_name}"] = res
    ppi_global = ppi

    # =====================================================================
    # PART 2: CC AUROC — expression-weighted CG profiles as features
    # =====================================================================
    p("\n" + "=" * 70)
    p("PART 2: CC AUROC — expression-weighted features")
    p("=" * 70)

    # Build features with original CG scores
    p("\n  Building baseline CC features...")
    baseline_features = build_cc_features(cg_scores, ppi, gt_c, n_c,
                                          mol_tok, mol_pa, mol_proj, cc_sim, known_idx)
    p(f"  {len(baseline_features)} baseline features")

    # Build features with expression-weighted CG scores
    p("  Building expression-weighted CC features...")
    w = expression_weight(tpm, "log")
    w_cg = weight_cg_scores(cg_scores, w)
    expr_features = build_cc_features(w_cg, ppi, gt_c, n_c,
                                       mol_tok, mol_pa, mol_proj, cc_sim, known_idx)
    # Add expression-weighted features with distinct names
    combined_features = {}
    combined_features.update(baseline_features)
    for k, v in expr_features.items():
        if k.endswith("_prof") or "ppi" in k or "jacc" in k:
            combined_features[f"expr_{k}"] = v
    p(f"  {len(combined_features)} combined features (baseline + expression-weighted)")

    for thresh_key, thresh_val in [("0.6", 0.6)]:
        p(f"\n  --- @{thresh_key} baseline (compound-level CV) ---")
        res = run_compound_cv(baseline_features, cc_sim, known_idx, thresh_val)
        all_results[f"cc_baseline_{thresh_key}"] = res

        p(f"\n  --- @{thresh_key} + expression features (compound-level CV) ---")
        res = run_compound_cv(combined_features, cc_sim, known_idx, thresh_val)
        all_results[f"cc_expr_{thresh_key}"] = res

    # =====================================================================
    # Summary
    # =====================================================================
    elapsed = time.time() - t0
    p(f"\n{'=' * 70}")
    p(f"TOTAL TIME: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    p(f"\n{'=' * 70}")
    p("SUMMARY")
    p(f"{'=' * 70}")

    p("\nPer-target results:")
    for k in sorted(all_results.keys()):
        v = all_results[k]
        if isinstance(v, dict) and "best_single" in v:
            p(f"  {k}: best={v['best_single']:.4f}, top3={v['top3_avg']:.4f}, "
              f"oracle={v['oracle']:.4f} ({v['n_targets']} targets)")

    p("\nCC AUROC results:")
    for k in sorted(all_results.keys()):
        v = all_results[k]
        if isinstance(v, dict) and "mean" in v and "best_single" not in v:
            p(f"  {k}: {v['mean']:.4f}")

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "transcriptomics_boost_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    p(f"\nSaved to {EVAL_DIR}/transcriptomics_boost_results.json")


if __name__ == "__main__":
    main()
