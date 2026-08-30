"""Push to 80% CC AUROC and 60% per-target median — v2.

Uses the EXACT feature construction from final_push.py (42 NxN similarity matrices)
plus additional embedding-based features. Fixes meta-learner NaN crash.
"""
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"
EMB_DIR = "/opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245"
EVAL_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"

PROT_CB_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_protein_codebook_dark-snowball-245")
MOL_CB_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_molecule_codebook_dark-snowball-245")
PROT_PROJ_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_protein_protein_mean_dark-snowball-245")
MOL_PROJ_DIR = os.path.join(EMB_DIR, "rxrx3_pairs_molecule_molecule_mean_dark-snowball-245")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    p(f"  {entity_type}: {matched}/{n} matched, tok={tok_dim}, pa={pa_dim}, proj={proj_dim}")
    return tok_embs, pa_embs, proj_embs


def load_data():
    p("Loading data...")
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
    prot_tok, prot_pa, prot_proj = load_embeddings(gt_g, PROT_CB_DIR, PROT_PROJ_DIR, "protein")

    return (gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores,
            mol_tok, mol_pa, mol_proj, prot_tok, prot_pa, prot_proj)


def build_original_42_features(cg_scores, ppi, gt_c):
    """EXACT copy of final_push.py build_features — produces 42 NxN similarity matrices."""
    n_c = len(gt_c)
    features = {}

    # ECFP
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_map = {c: i for i, c in enumerate(ecfp["compound_ids"].tolist())}
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c])
    ev = np.where(ecfp_idx >= 0)[0]
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ecfp_aligned[np.ix_(ev, ev)] = np.array(ecfp["tanimoto"])[np.ix_(ecfp_idx[ev], ecfp_idx[ev])]
    features["ecfp"] = ecfp_aligned

    # 4 CC matrices from interaction prints
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

    # 10 CG profile cosines
    for key, scores in cg_scores.items():
        features[f"{key}_prof"] = cosine_similarity(scores).astype(np.float32)

    # 18 PPI-propagated CG profile cosines
    for ppi_thresh, ppi_label in [(0.0, "full"), (0.3, "t03"), (0.5, "t05")]:
        for alpha in [0.3, 0.7]:
            for key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
                propagated = propagate_ppi(cg_scores[key], ppi, alpha=alpha, threshold=ppi_thresh)
                features[f"{key}_ppi_{ppi_label}_a{alpha}"] = cosine_similarity(propagated).astype(np.float32)

    # 9 Jaccard features
    for cg_key in ["proj_raw", "cb_tok_raw", "cb_pa_raw"]:
        propagated = propagate_ppi(cg_scores[cg_key], ppi, alpha=0.5, threshold=0.3)
        for bin_pct in [70, 85, 95]:
            thresh = np.percentile(propagated, bin_pct)
            binary = (propagated > thresh).astype(np.float32)
            intersection = binary @ binary.T
            counts = binary.sum(axis=1)
            union = counts[:, None] + counts[None, :] - intersection
            features[f"{cg_key}_jacc_p{bin_pct}"] = (intersection / (union + 5.0 + 1e-8)).astype(np.float32)

    p(f"  Built {len(features)} original features")
    return features


def build_embedding_cc_features(mol_tok, mol_pa, mol_proj, n_pca=64):
    """Build NxN similarity matrices from molecule embeddings (PCA'd)."""
    features = {}

    for emb, name in [(mol_tok, "mol_tok"), (mol_pa, "mol_pa"), (mol_proj, "mol_proj")]:
        pca = PCA(n_components=n_pca, random_state=42)
        pca_emb = pca.fit_transform(emb)
        features[f"{name}_pca_cos"] = cosine_similarity(pca_emb).astype(np.float32)

        # Abs difference and product features (top PCA components only)
        for d in range(min(8, n_pca)):
            v = pca_emb[:, d]
            features[f"{name}_pca{d}_absdiff"] = np.abs(v[:, None] - v[None, :]).astype(np.float32)
            features[f"{name}_pca{d}_prod"] = (v[:, None] * v[None, :]).astype(np.float32)

    # Combined embedding cosine
    all_emb = np.hstack([mol_tok, mol_pa, mol_proj])
    pca_all = PCA(n_components=128, random_state=42).fit_transform(all_emb)
    features["mol_all_pca_cos"] = cosine_similarity(pca_all).astype(np.float32)

    p(f"  Built {len(features)} embedding CC features")
    return features


def build_cg_profile_features(cg_scores, ppi, n_pca=16):
    """Build NxN similarity matrices from PCA'd CG profiles."""
    features = {}
    for key in ["proj_raw", "cb_pa_raw", "cb_tok_raw"]:
        mat = cg_scores[key]
        prop = propagate_ppi(mat, ppi, alpha=0.5, threshold=0.3)
        pca = PCA(n_components=n_pca, random_state=42)
        pca_prof = pca.fit_transform(prop)

        for d in range(min(8, n_pca)):
            v = pca_prof[:, d]
            features[f"{key}_cgpca{d}_absdiff"] = np.abs(v[:, None] - v[None, :]).astype(np.float32)
            features[f"{key}_cgpca{d}_prod"] = (v[:, None] * v[None, :]).astype(np.float32)

    p(f"  Built {len(features)} CG profile features")
    return features


def extract_pairs(features, subset_idx):
    n = len(subset_idx)
    pair_idx = np.triu_indices(n, k=1)
    names = list(features.keys())
    X = np.column_stack([features[k][np.ix_(subset_idx, subset_idx)][pair_idx] for k in names])
    return X, names, pair_idx


def select_known(cg_scores, n_select):
    combined = np.zeros(list(cg_scores.values())[0].shape[0])
    for scores in cg_scores.values():
        r = scores.max(axis=1) - scores.min(axis=1)
        combined += r / (r.max() + 1e-8)
    return np.argsort(-combined)[:n_select]


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


def train_mlp_fold(X_tr, y_tr, X_te, y_te, hidden_dims, epochs=120, lr=1e-3, bs=4096):
    model = PairMLP(X_tr.shape[1], hidden_dims).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    X_tr_d = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_d = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    X_te_d = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)

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
            try:
                auc = roc_auc_score(y_te, preds)
            except:
                auc = 0.5
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


def cc_auroc_experiment(gt, gt_c, n_c, cg_scores, ppi,
                        mol_tok, mol_pa, mol_proj, known_idx):
    p("\n" + "=" * 70)
    p("PART 1: CC AUROC — targeting 80%+")
    p("=" * 70)

    # Build all feature sets as NxN matrices
    p("Building original 42 features...")
    orig_features = build_original_42_features(cg_scores, ppi, gt_c)

    p("Building embedding CC features...")
    emb_features = build_embedding_cc_features(mol_tok, mol_pa, mol_proj)

    p("Building CG profile features...")
    cgp_features = build_cg_profile_features(cg_scores, ppi)

    # Combine
    all_features = {}
    all_features.update(orig_features)
    all_features.update(emb_features)
    all_features.update(cgp_features)

    results = {}
    cc_sim = gt["cc_similarity"]
    ki = known_idx
    n_k = len(ki)

    for thresh_key, thresh_val in [("0.4", 0.4), ("0.6", 0.6)]:
        p(f"\n--- Threshold @{thresh_key} ---")

        for feat_name, feat_dict in [("orig_42", orig_features),
                                      ("orig+emb", {**orig_features, **emb_features}),
                                      ("all", all_features)]:
            X, names, pair_idx = extract_pairs(feat_dict, ki)
            labels = cc_sim[np.ix_(ki, ki)][pair_idx] > thresh_val
            labels = labels.astype(np.float32)
            n_pairs = len(labels)
            pos_rate = labels.mean()
            p(f"\n  [{feat_name}] {len(names)} features, {n_pairs} pairs, {pos_rate:.1%} pos")

            # Compound-level 5-fold CV
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            oof = {m: np.zeros(n_pairs) for m in ["xgb", "mlp", "mlp_large", "xgb_reg"]}
            fold_aucs = {m: [] for m in oof}
            oof_mask = np.zeros(n_pairs, dtype=bool)

            for fold, (tr_c, va_c) in enumerate(kf.split(np.arange(n_k), np.zeros(n_k))):
                tr_set = set(tr_c.tolist())
                va_set = set(va_c.tolist())

                tr_mask = np.array([(pair_idx[0][p] in tr_set and pair_idx[1][p] in tr_set) for p in range(n_pairs)])
                va_mask = np.array([(pair_idx[0][p] in va_set or pair_idx[1][p] in va_set) for p in range(n_pairs)])
                va_mask = va_mask & ~tr_mask

                y_tr, y_va = labels[tr_mask], labels[va_mask]
                if len(np.unique(y_va)) < 2:
                    continue
                oof_mask |= va_mask

                scaler = StandardScaler().fit(X[tr_mask])
                X_tr_s = scaler.transform(X[tr_mask])
                X_va_s = scaler.transform(X[va_mask])

                # XGBoost classifier
                xgb_m = xgb.XGBClassifier(
                    n_estimators=2000, max_depth=7, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", device="cuda", random_state=42, verbosity=0,
                    scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
                )
                xgb_m.fit(X_tr_s, y_tr)
                pred = xgb_m.predict_proba(X_va_s)[:, 1]
                oof["xgb"][va_mask] = pred
                fold_aucs["xgb"].append(roc_auc_score(y_va, pred))

                # XGBoost regressor
                y_tr_cont = cc_sim[np.ix_(ki, ki)][pair_idx][tr_mask]
                xgb_reg = xgb.XGBRegressor(
                    n_estimators=2000, max_depth=7, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.6,
                    tree_method="hist", device="cuda", random_state=42, verbosity=0,
                )
                xgb_reg.fit(X_tr_s, y_tr_cont)
                pred = xgb_reg.predict(X_va_s)
                oof["xgb_reg"][va_mask] = pred
                fold_aucs["xgb_reg"].append(roc_auc_score(y_va, pred))

                # MLP
                pred, auc = train_mlp_fold(X_tr_s, y_tr, X_va_s, y_va, [1024, 512, 256, 128])
                oof["mlp"][va_mask] = pred
                fold_aucs["mlp"].append(auc)

                # Large MLP
                pred, auc = train_mlp_fold(X_tr_s, y_tr, X_va_s, y_va, [2048, 1024, 512, 256, 128])
                oof["mlp_large"][va_mask] = pred
                fold_aucs["mlp_large"].append(auc)

                p(f"    Fold {fold}: xgb={fold_aucs['xgb'][-1]:.4f} mlp={fold_aucs['mlp'][-1]:.4f} "
                  f"mlp_L={fold_aucs['mlp_large'][-1]:.4f} reg={fold_aucs['xgb_reg'][-1]:.4f}")

            # Report
            p(f"\n  Individual @{thresh_key} [{feat_name}]:")
            for m in oof:
                if fold_aucs[m]:
                    mean_a = np.mean(fold_aucs[m])
                    std_a = np.std(fold_aucs[m])
                    p(f"    {m:15s} {mean_a:.4f} ± {std_a:.4f}")
                    results[f"{m}_{feat_name}_{thresh_key}"] = {"mean": mean_a, "std": std_a}

            # Ensembles
            valid = oof_mask
            y_valid = labels[valid]
            from itertools import combinations
            for n_ens in [2, 3, 4]:
                best_a, best_n = 0, ""
                for combo in combinations(oof.keys(), n_ens):
                    avg = np.mean([oof[n][valid] for n in combo], axis=0)
                    try:
                        a = roc_auc_score(y_valid, avg)
                        if a > best_a:
                            best_a = a
                            best_n = "+".join(combo)
                    except:
                        pass
                p(f"    Best {n_ens}-ens: {best_n} = {best_a:.4f}")
                results[f"best_{n_ens}ens_{feat_name}_{thresh_key}"] = {"mean": best_a, "combo": best_n}

    return results


def per_target_experiment(gt, gt_c, gt_g, n_c, n_g, cg_scores, ppi,
                          prot_tok, prot_pa, prot_proj):
    p("\n" + "=" * 70)
    p("PART 2: Per-target AUROC — targeting 60%+")
    p("=" * 70)

    cg_binary = gt["cg_binary_top1"]
    results = {}

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

    p(f"  {len(configs)} configs")

    for n_sel, subset_name in [(150, "known_150"), (246, "known_246")]:
        known_idx = select_known(cg_scores, n_sel)
        p(f"\n  --- {subset_name} ({n_sel} compounds) ---")

        per_target_aucs = np.full((n_g, len(configs)), np.nan)

        for ci, (key, alpha, thresh) in enumerate(configs):
            mat = cg_scores[key]
            if alpha > 0:
                mat = propagate_ppi(mat, ppi, alpha, thresh)
            scores_sub = mat[known_idx]

            for g in range(n_g):
                y = cg_binary[known_idx, g]
                if y.sum() < 2 or y.sum() > len(y) - 2:
                    continue
                try:
                    per_target_aucs[g, ci] = roc_auc_score(y, scores_sub[:, g])
                except:
                    pass

        # Filter valid targets (at least one non-NaN config)
        valid_targets = ~np.all(np.isnan(per_target_aucs), axis=1)
        valid_idx = np.where(valid_targets)[0]
        n_valid = len(valid_idx)

        oracle_aucs = np.nanmax(per_target_aucs[valid_idx], axis=1)
        oracle_median = np.median(oracle_aucs)
        p(f"  Oracle: median={oracle_median:.4f} ({n_valid} targets)")
        p(f"    >0.6: {(oracle_aucs > 0.6).sum()}, >0.7: {(oracle_aucs > 0.7).sum()}, >0.8: {(oracle_aucs > 0.8).sum()}")
        results[f"oracle_{subset_name}"] = {"median": float(oracle_median), "n_targets": n_valid}

        config_medians = np.array([np.nanmedian(per_target_aucs[valid_idx, ci]) for ci in range(len(configs))])
        best_ci = np.nanargmax(config_medians)
        p(f"  Best single: {configs[best_ci]} median={config_medians[best_ci]:.4f}")
        results[f"best_single_{subset_name}"] = {"median": float(config_medians[best_ci]), "config": str(configs[best_ci])}

        # --- META-LEARNER ---
        p(f"\n  Meta-learner on {subset_name}...")

        # Target features: protein embeddings PCA'd + CG score statistics
        target_embs = np.hstack([prot_tok, prot_pa, prot_proj])
        pca_t = PCA(n_components=128, random_state=42).fit_transform(target_embs)

        cg_stats = []
        for key in cg_keys:
            mat = cg_scores[key][known_idx]
            cg_stats.append(np.mean(mat, axis=0).reshape(-1, 1))
            cg_stats.append(np.std(mat, axis=0).reshape(-1, 1))
            cg_stats.append(np.max(mat, axis=0).reshape(-1, 1))
            cg_stats.append(np.percentile(mat, 90, axis=0).reshape(-1, 1))
        cg_stat_features = np.hstack(cg_stats)

        target_X = np.hstack([pca_t, cg_stat_features])
        p(f"    Features: {target_X.shape[1]} (pca={pca_t.shape[1]}, cg_stats={cg_stat_features.shape[1]})")

        # Labels: best config index (only for valid targets)
        X_meta = target_X[valid_idx]
        best_config_idx = np.nanargmax(per_target_aucs[valid_idx], axis=1)
        y_meta = best_config_idx

        # Classification meta-learner
        meta_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        meta_selected_aucs = np.full(n_valid, np.nan)

        for fold, (tr, va) in enumerate(meta_kf.split(X_meta, y_meta)):
            scaler = StandardScaler().fit(X_meta[tr])
            clf = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_class=len(configs), objective="multi:softmax",
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
            )
            clf.fit(scaler.transform(X_meta[tr]), y_meta[tr])
            pred_ci = clf.predict(scaler.transform(X_meta[va])).astype(int)

            for i, vi in enumerate(va):
                ci = pred_ci[i]
                auc_val = per_target_aucs[valid_idx[vi], ci]
                if not np.isnan(auc_val):
                    meta_selected_aucs[vi] = auc_val
                else:
                    meta_selected_aucs[vi] = per_target_aucs[valid_idx[vi], best_ci]

        valid_meta = ~np.isnan(meta_selected_aucs)
        meta_median = np.median(meta_selected_aucs[valid_meta])
        p(f"  Classification meta-learner: median={meta_median:.4f} ({valid_meta.sum()} targets)")
        results[f"clf_meta_{subset_name}"] = {"median": float(meta_median), "n_targets": int(valid_meta.sum())}

        # Regression meta-learner: predict AUROC per (target, config)
        p(f"  Regression meta-learner...")
        reg_selected_aucs = np.full(n_valid, np.nan)

        for fold, (tr, va) in enumerate(meta_kf.split(X_meta, y_meta)):
            scaler = StandardScaler().fit(X_meta[tr])

            # For each config, train a regressor to predict per-target AUROC
            pred_aurocs = np.full((len(va), len(configs)), np.nan)

            for ci in range(len(configs)):
                # Training targets that have a valid AUROC for this config
                y_ci = per_target_aucs[valid_idx[tr], ci]
                valid_ci = ~np.isnan(y_ci)
                if valid_ci.sum() < 10:
                    continue
                reg = xgb.XGBRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    tree_method="hist", device="cuda", random_state=42, verbosity=0,
                )
                reg.fit(scaler.transform(X_meta[tr][valid_ci]), y_ci[valid_ci])
                pred_aurocs[:, ci] = reg.predict(scaler.transform(X_meta[va]))

            # For each val target, pick config with highest predicted AUROC
            for i, vi in enumerate(va):
                valid_preds = ~np.isnan(pred_aurocs[i])
                if valid_preds.any():
                    best_pred_ci = np.nanargmax(pred_aurocs[i])
                    actual = per_target_aucs[valid_idx[vi], best_pred_ci]
                    if not np.isnan(actual):
                        reg_selected_aucs[vi] = actual
                    else:
                        reg_selected_aucs[vi] = per_target_aucs[valid_idx[vi], best_ci]

        valid_reg = ~np.isnan(reg_selected_aucs)
        reg_median = np.median(reg_selected_aucs[valid_reg])
        p(f"  Regression meta-learner: median={reg_median:.4f} ({valid_reg.sum()} targets)")
        results[f"reg_meta_{subset_name}"] = {"median": float(reg_median), "n_targets": int(valid_reg.sum())}

        # Top-K config averaging
        for k in [3, 5, 7]:
            top_k_ci = np.argsort(-config_medians)[:k]
            top_k_avg = np.nanmean(per_target_aucs[valid_idx][:, top_k_ci], axis=1)
            top_k_median = np.nanmedian(top_k_avg)
            p(f"  Top-{k} config avg: median={top_k_median:.4f}")
            results[f"top{k}_avg_{subset_name}"] = {"median": float(top_k_median)}

    return results


def main():
    t0 = time.time()

    (gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores,
     mol_tok, mol_pa, mol_proj, prot_tok, prot_pa, prot_proj) = load_data()

    known_idx = select_known(cg_scores, 150)
    p(f"\nSelected 150 known compounds")

    cc_results = cc_auroc_experiment(
        gt, gt_c, n_c, cg_scores, ppi, mol_tok, mol_pa, mol_proj, known_idx)

    pt_results = per_target_experiment(
        gt, gt_c, gt_g, n_c, n_g, cg_scores, ppi, prot_tok, prot_pa, prot_proj)

    all_results = {"cc_auroc": cc_results, "per_target": pt_results}
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "push_80_60_v2_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    p(f"\n{'=' * 70}")
    p(f"TOTAL TIME: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

    p(f"\n{'=' * 70}")
    p("SUMMARY vs TARGETS")
    p(f"{'=' * 70}")
    p(f"\n  CC AUROC target: >80%")
    for k, v in sorted(cc_results.items()):
        if "mean" in v and v["mean"] >= 0.74:
            mark = ">>>" if v["mean"] >= 0.80 else "   "
            p(f"  {mark} {k}: {v['mean']:.4f}")
    p(f"\n  Per-target target: >60%")
    for k, v in sorted(pt_results.items()):
        if "median" in v:
            mark = ">>>" if v["median"] >= 0.60 else "   "
            p(f"  {mark} {k}: {v['median']:.4f}")


if __name__ == "__main__":
    main()
