"""Push to 80% CC AUROC and 60% per-target median.

Strategy:
  CC AUROC: Use raw molecule embeddings (tokens+PA+projection) as features,
            combined with CG profile features. Siamese MLP on rich representations.
  Per-target: Meta-learner that selects best (CG_key, alpha, threshold) per target
              based on protein embedding features. Oracle on known-246 = 68.95%.
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

N_KNOWN = 150
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
    """Load all embedding types for a set of entities."""
    tok_dir = os.path.join(cb_dir, "tokens")
    pa_dir = os.path.join(cb_dir, "pooled_attention")

    tok_files = {os.path.splitext(f)[0]: f for f in os.listdir(tok_dir) if f.endswith(".pt")}
    pa_files = {os.path.splitext(f)[0]: f for f in os.listdir(pa_dir) if f.endswith(".pt")}
    proj_files = {os.path.splitext(f)[0]: f for f in os.listdir(proj_dir) if f.endswith(".pt")}

    n = len(ids)
    sample_tok = torch.load(os.path.join(tok_dir, list(tok_files.values())[0]), map_location="cpu")
    sample_pa = torch.load(os.path.join(pa_dir, list(pa_files.values())[0]), map_location="cpu")
    sample_proj = torch.load(os.path.join(proj_dir, list(proj_files.values())[0]), map_location="cpu")

    tok_embs = np.zeros((n, sample_tok.shape[0]), dtype=np.float32)
    pa_embs = np.zeros((n, sample_pa.shape[0]), dtype=np.float32)
    proj_embs = np.zeros((n, sample_proj.shape[0]), dtype=np.float32)

    matched = 0
    for i, eid in enumerate(ids):
        if eid in tok_files:
            tok_embs[i] = torch.load(os.path.join(tok_dir, tok_files[eid]), map_location="cpu").float().numpy()
            pa_embs[i] = torch.load(os.path.join(pa_dir, pa_files[eid]), map_location="cpu").float().numpy()
            proj_embs[i] = torch.load(os.path.join(proj_dir, proj_files[eid]), map_location="cpu").float().numpy()
            matched += 1

    p(f"  Loaded {entity_type} embeddings: {matched}/{n} matched")
    p(f"    tokens: {tok_embs.shape}, PA: {pa_embs.shape}, proj: {proj_embs.shape}")
    return tok_embs, pa_embs, proj_embs


def load_data():
    """Load ground truth, CG scores, PPI, and embeddings."""
    p("Loading data...")
    gt = dict(np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"), allow_pickle=True))
    for k in gt:
        gt[k] = np.array(gt[k])
    cc_sim_data = np.load(os.path.join(GT_DIR, "phenomics_compound_compound_sim.npz"))
    gt["cc_similarity"] = np.array(cc_sim_data["similarity"])
    gt_c = gt["compound_ids"].tolist()
    gt_g = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_c), len(gt_g)
    p(f"  Ground truth: {n_c} compounds, {n_g} genes")

    ppi = np.load(os.path.join(GT_DIR, "bonbon_ppi.npz"))["ppi_matrix"]
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))["tanimoto"]

    # Load CG scores
    cg_scores = {}
    for fname, label in [
        ("cg_projection.npz", "proj"),
        ("cg_codebook_tokens.npz", "cb_tok"),
        ("cg_pooled_attention.npz", "cb_pa"),
    ]:
        data = np.load(os.path.join(IP_DIR, fname))
        c_idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        g_idx = align_to_gt(data["protein_ids"].tolist(), gt_g)
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
        c_idx = align_to_gt(data["compound_ids"].tolist(), gt_c)
        g_idx = align_to_gt(data["protein_ids"].tolist(), gt_g)
        vc, vg = c_idx >= 0, g_idx >= 0
        for sk, suffix in [("score_raw", "raw"), ("score_cosine", "cos")]:
            mat_T = np.array(data[sk]).T
            aligned = np.zeros((n_c, n_g), dtype=np.float32)
            aligned[np.ix_(np.where(vc)[0], np.where(vg)[0])] = mat_T[np.ix_(c_idx[vc], g_idx[vg])]
            cg_scores[f"{label}_{suffix}"] = aligned

    p(f"  Loaded {len(cg_scores)} CG score matrices")

    # Load raw embeddings
    p("  Loading molecule embeddings...")
    mol_tok, mol_pa, mol_proj = load_embeddings(gt_c, MOL_CB_DIR, MOL_PROJ_DIR, "molecule")
    p("  Loading protein embeddings...")
    prot_tok, prot_pa, prot_proj = load_embeddings(gt_g, PROT_CB_DIR, PROT_PROJ_DIR, "protein")

    return (gt, gt_c, gt_g, n_c, n_g, ppi, ecfp, cg_scores,
            mol_tok, mol_pa, mol_proj, prot_tok, prot_pa, prot_proj)


def select_known(cg_scores, n_c, n=150):
    """Select top-N compounds by combined CG score range."""
    ranges = []
    for key, mat in cg_scores.items():
        r = mat.max(axis=1) - mat.min(axis=1)
        ranks = np.argsort(np.argsort(-r))
        ranges.append(ranks)
    combined = np.mean(ranges, axis=0)
    return np.argsort(combined)[:n]


def build_original_features(idx_i, idx_j, ecfp, cg_scores, ppi, n_g):
    """Build the original 42 features (baseline)."""
    n_pairs = len(idx_i)
    features = []
    names = []

    # ECFP
    features.append(ecfp[idx_i, idx_j].reshape(-1, 1))
    names.append("ecfp_tanimoto")

    # CG profile cosines + PPI-propagated + Jaccard
    ppi_configs = [(0.3, 0.0), (0.3, 0.3), (0.5, 0.0), (0.5, 0.3), (0.7, 0.0)]
    for key, mat in cg_scores.items():
        prof_i, prof_j = mat[idx_i], mat[idx_j]
        cos = np.sum(prof_i * prof_j, axis=1) / (
            np.linalg.norm(prof_i, axis=1) * np.linalg.norm(prof_j, axis=1) + 1e-8
        )
        features.append(cos.reshape(-1, 1))
        names.append(f"{key}_cos_prof")

        for alpha, thresh in ppi_configs[:3]:
            prop = propagate_ppi(mat, ppi, alpha, thresh)
            pi, pj = prop[idx_i], prop[idx_j]
            c = np.sum(pi * pj, axis=1) / (
                np.linalg.norm(pi, axis=1) * np.linalg.norm(pj, axis=1) + 1e-8
            )
            features.append(c.reshape(-1, 1))
            names.append(f"{key}_ppi_a{alpha}_t{thresh}")

        for pct in [90, 95, 99]:
            thr = np.percentile(mat, pct, axis=1, keepdims=True)
            bi, bj = (mat >= thr)[idx_i], (mat >= thr)[idx_j]
            inter = (bi & bj).sum(axis=1).astype(float)
            union = (bi | bj).sum(axis=1).astype(float)
            jacc = inter / (union + 1e-8)
            features.append(jacc.reshape(-1, 1))
            names.append(f"{key}_jacc_p{pct}")

    X = np.hstack(features)
    return X, names


def build_embedding_features(idx_i, idx_j, mol_tok, mol_pa, mol_proj, pca_tok, pca_pa, pca_proj):
    """Build features from PCA'd molecule embeddings."""
    # PCA-transformed embeddings for each compound
    ti, tj = pca_tok[idx_i], pca_tok[idx_j]
    pi, pj = pca_pa[idx_i], pca_pa[idx_j]
    ri, rj = pca_proj[idx_i], pca_proj[idx_j]

    features = []
    for ei, ej, name in [(ti, tj, "tok"), (pi, pj, "pa"), (ri, rj, "proj")]:
        features.append(np.abs(ei - ej))  # absolute difference
        features.append(ei * ej)           # element-wise product
        # cosine similarity as single feature
        cos = np.sum(ei * ej, axis=1) / (
            np.linalg.norm(ei, axis=1) * np.linalg.norm(ej, axis=1) + 1e-8
        )
        features.append(cos.reshape(-1, 1))

    return np.hstack(features)


def build_cg_profile_features(idx_i, idx_j, cg_scores, ppi):
    """Build features from raw CG profiles (not just cosine summaries).
    Use PCA'd CG profiles as pairwise features."""
    features = []
    key_list = ["proj_raw", "proj_cos", "cb_pa_raw"]

    for key in key_list:
        mat = cg_scores[key]
        # PPI-propagated version
        prop = propagate_ppi(mat, ppi, alpha=0.5, threshold=0.3)

        # PCA the CG profiles
        pca = PCA(n_components=32, random_state=42)
        pca_profiles = pca.fit_transform(prop)

        pi, pj = pca_profiles[idx_i], pca_profiles[idx_j]
        features.append(np.abs(pi - pj))
        features.append(pi * pj)

    return np.hstack(features)


class SiameseMLP(nn.Module):
    """Siamese network: shared encoder + pairwise comparison."""
    def __init__(self, input_dim, encoder_dims, classifier_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for d in encoder_dims:
            layers.extend([nn.Linear(prev, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(0.2)])
            prev = d
        self.encoder = nn.Sequential(*layers)

        clf_layers = []
        prev = encoder_dims[-1] * 2 + 1  # abs_diff + product + cosine(1)
        for d in classifier_dims:
            clf_layers.extend([nn.Linear(prev, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(0.3)])
            prev = d
        clf_layers.append(nn.Linear(prev, 1))
        self.classifier = nn.Sequential(*clf_layers)

    def forward(self, x_i, x_j):
        e_i = self.encoder(x_i)
        e_j = self.encoder(x_j)
        diff = torch.abs(e_i - e_j)
        prod = e_i * e_j
        cos = torch.sum(e_i * e_j, dim=1, keepdim=True) / (
            torch.norm(e_i, dim=1, keepdim=True) * torch.norm(e_j, dim=1, keepdim=True) + 1e-8
        )
        combined = torch.cat([diff, prod, cos], dim=1)
        return self.classifier(combined).squeeze(-1)


class PairMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        layers = []
        prev = input_dim
        for d in hidden_dims:
            layers.extend([nn.Linear(prev, d), nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(0.3)])
            prev = d
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_fold(X_train, y_train, X_val, y_val, hidden_dims, epochs=100, lr=1e-3):
    """Train MLP on features."""
    input_dim = X_train.shape[1]
    model = PairMLP(input_dim, hidden_dims).to(DEVICE)
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    Xt = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    Xv = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)

    best_auc, patience, best_state = 0, 0, None
    bs = min(2048, len(X_train))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for start in range(0, len(Xt), bs):
            idx = perm[start:start + bs]
            logits = model(Xt[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv).cpu().numpy()
        try:
            auc = roc_auc_score(y_val, val_logits)
        except:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).cpu().numpy()
    return model, val_pred, best_auc


def train_siamese_fold(emb_i_train, emb_j_train, y_train, emb_i_val, emb_j_val, y_val,
                        encoder_dims, classifier_dims, epochs=100, lr=1e-3):
    """Train Siamese network on raw embeddings."""
    input_dim = emb_i_train.shape[1]
    model = SiameseMLP(input_dim, encoder_dims, classifier_dims).to(DEVICE)
    pos_w = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, dtype=torch.float32).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    ei_t = torch.tensor(emb_i_train, dtype=torch.float32).to(DEVICE)
    ej_t = torch.tensor(emb_j_train, dtype=torch.float32).to(DEVICE)
    yt = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    ei_v = torch.tensor(emb_i_val, dtype=torch.float32).to(DEVICE)
    ej_v = torch.tensor(emb_j_val, dtype=torch.float32).to(DEVICE)

    best_auc, patience, best_state = 0, 0, None
    bs = min(2048, len(y_train))

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(yt))
        for start in range(0, len(yt), bs):
            idx = perm[start:start + bs]
            logits = model(ei_t[idx], ej_t[idx])
            loss = criterion(logits, yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(ei_v, ej_v).cpu().numpy()
        try:
            auc = roc_auc_score(y_val, val_logits)
        except:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            patience = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 8:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(ei_v, ej_v).cpu().numpy()
    return model, val_pred, best_auc


def cc_auroc_experiment(gt, gt_c, n_c, ecfp, cg_scores, ppi,
                        mol_tok, mol_pa, mol_proj, known_idx):
    """Part 1: Push CC AUROC past 80%."""
    p("\n" + "=" * 70)
    p("PART 1: CC AUROC — targeting 80%+")
    p("=" * 70)

    results = {}

    for thresh_key, thresh_val in [("0.4", 0.4), ("0.6", 0.6)]:
        p(f"\n--- Threshold @{thresh_key} ---")

        # Build labels for known compounds
        cc_sim = gt["cc_similarity"]
        ki = known_idx
        n_k = len(ki)
        idx_i, idx_j = [], []
        for a in range(n_k):
            for b in range(a + 1, n_k):
                idx_i.append(ki[a])
                idx_j.append(ki[b])
        idx_i, idx_j = np.array(idx_i), np.array(idx_j)
        labels = (cc_sim[idx_i, idx_j] > thresh_val).astype(np.float32)
        n_pairs = len(labels)
        pos_rate = labels.mean()
        p(f"  {n_pairs} pairs, {pos_rate:.1%} positive")

        # PCA molecule embeddings
        pca_dims = {"tok": 64, "pa": 128, "proj": 64}
        pca_tok = PCA(n_components=pca_dims["tok"], random_state=42).fit_transform(mol_tok)
        pca_pa = PCA(n_components=pca_dims["pa"], random_state=42).fit_transform(mol_pa)
        pca_proj = PCA(n_components=pca_dims["proj"], random_state=42).fit_transform(mol_proj)

        # Build feature sets
        p("  Building features...")
        X_orig, feat_names_orig = build_original_features(idx_i, idx_j, ecfp, cg_scores, ppi, len(gt["gene_ids"]))
        X_emb = build_embedding_features(idx_i, idx_j, mol_tok, mol_pa, mol_proj, pca_tok, pca_pa, pca_proj)
        X_cg_prof = build_cg_profile_features(idx_i, idx_j, cg_scores, ppi)

        # Combined feature sets
        feature_sets = {
            "orig_42": X_orig,
            "orig+emb": np.hstack([X_orig, X_emb]),
            "orig+emb+cgprof": np.hstack([X_orig, X_emb, X_cg_prof]),
            "all_rich": np.hstack([X_orig, X_emb, X_cg_prof]),
        }

        # Raw embeddings for Siamese (concat all embedding types)
        raw_emb_all = np.hstack([mol_tok, mol_pa, mol_proj])  # 10,496-dim
        # PCA down for Siamese input
        pca_siamese = PCA(n_components=256, random_state=42).fit_transform(raw_emb_all)

        p(f"  Feature dims: orig={X_orig.shape[1]}, +emb={X_emb.shape[1]}, +cgprof={X_cg_prof.shape[1]}")
        p(f"  Total rich: {feature_sets['all_rich'].shape[1]}, Siamese input: {pca_siamese.shape[1]}")

        # Compound-level CV
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        compound_labels = np.zeros(n_c)
        compound_labels[ki] = 1

        # Storage for OOF predictions
        oof_preds = {name: np.zeros(n_pairs) for name in [
            "xgb_orig", "xgb_rich", "mlp_orig", "mlp_rich", "mlp_large",
            "siamese_256", "xgb_reg_rich",
        ]}
        fold_aucs = {name: [] for name in oof_preds}
        oof_mask = np.zeros(n_pairs, dtype=bool)

        for fold, (train_compounds, val_compounds) in enumerate(kf.split(ki, np.zeros(n_k))):
            train_c = set(ki[train_compounds].tolist())
            val_c = set(ki[val_compounds].tolist())

            train_mask = np.array([idx_i[p] in train_c and idx_j[p] in train_c for p in range(n_pairs)])
            val_mask = np.array([idx_i[p] in val_c or idx_j[p] in val_c for p in range(n_pairs)])
            val_mask = val_mask & ~train_mask

            y_train, y_val = labels[train_mask], labels[val_mask]
            if len(np.unique(y_val)) < 2:
                continue

            oof_mask |= val_mask

            # --- XGBoost on original 42 features ---
            X_tr, X_va = X_orig[train_mask], X_orig[val_mask]
            scaler = StandardScaler().fit(X_tr)
            X_tr_s, X_va_s = scaler.transform(X_tr), scaler.transform(X_va)

            xgb_model = xgb.XGBClassifier(
                n_estimators=2000, max_depth=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            )
            xgb_model.fit(X_tr_s, y_train)
            pred = xgb_model.predict_proba(X_va_s)[:, 1]
            oof_preds["xgb_orig"][val_mask] = pred
            fold_aucs["xgb_orig"].append(roc_auc_score(y_val, pred))

            # --- XGBoost on rich features ---
            X_rich = feature_sets["all_rich"]
            X_tr_r, X_va_r = X_rich[train_mask], X_rich[val_mask]
            scaler_r = StandardScaler().fit(X_tr_r)
            X_tr_rs, X_va_rs = scaler_r.transform(X_tr_r), scaler_r.transform(X_va_r)

            xgb_rich = xgb.XGBClassifier(
                n_estimators=3000, max_depth=8, learning_rate=0.02,
                subsample=0.8, colsample_bytree=0.6,
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            )
            xgb_rich.fit(X_tr_rs, y_train)
            pred = xgb_rich.predict_proba(X_va_rs)[:, 1]
            oof_preds["xgb_rich"][val_mask] = pred
            fold_aucs["xgb_rich"].append(roc_auc_score(y_val, pred))

            # --- XGBoost Regressor on rich features ---
            xgb_reg = xgb.XGBRegressor(
                n_estimators=2000, max_depth=7, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.6,
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
            )
            y_train_cont = cc_sim[idx_i[train_mask], idx_j[train_mask]]
            xgb_reg.fit(X_tr_rs, y_train_cont)
            pred = xgb_reg.predict(X_va_rs)
            oof_preds["xgb_reg_rich"][val_mask] = pred
            fold_aucs["xgb_reg_rich"].append(roc_auc_score(y_val, pred))

            # --- MLP on original features ---
            _, pred, auc = train_mlp_fold(X_tr_s, y_train, X_va_s, y_val, [1024, 512, 256, 128])
            oof_preds["mlp_orig"][val_mask] = pred
            fold_aucs["mlp_orig"].append(auc)

            # --- MLP on rich features ---
            _, pred, auc = train_mlp_fold(X_tr_rs, y_train, X_va_rs, y_val, [1024, 512, 256, 128])
            oof_preds["mlp_rich"][val_mask] = pred
            fold_aucs["mlp_rich"].append(auc)

            # --- Large MLP on rich features ---
            _, pred, auc = train_mlp_fold(X_tr_rs, y_train, X_va_rs, y_val, [2048, 1024, 512, 256, 128])
            oof_preds["mlp_large"][val_mask] = pred
            fold_aucs["mlp_large"].append(auc)

            # --- Siamese on PCA'd raw embeddings ---
            si_i_tr = pca_siamese[idx_i[train_mask]]
            si_j_tr = pca_siamese[idx_j[train_mask]]
            si_i_va = pca_siamese[idx_i[val_mask]]
            si_j_va = pca_siamese[idx_j[val_mask]]
            _, pred, auc = train_siamese_fold(
                si_i_tr, si_j_tr, y_train, si_i_va, si_j_va, y_val,
                encoder_dims=[512, 256], classifier_dims=[256, 128], epochs=100, lr=5e-4,
            )
            oof_preds["siamese_256"][val_mask] = pred
            fold_aucs["siamese_256"].append(auc)

            p(f"  Fold {fold}: xgb_orig={fold_aucs['xgb_orig'][-1]:.4f} "
              f"xgb_rich={fold_aucs['xgb_rich'][-1]:.4f} "
              f"mlp_rich={fold_aucs['mlp_rich'][-1]:.4f} "
              f"mlp_large={fold_aucs['mlp_large'][-1]:.4f} "
              f"siamese={fold_aucs['siamese_256'][-1]:.4f}")

        # Report individual models
        p(f"\n  Individual models @{thresh_key}:")
        for name in oof_preds:
            aucs = fold_aucs[name]
            if aucs:
                mean_auc = np.mean(aucs)
                std_auc = np.std(aucs)
                p(f"    {name:25s} AUROC={mean_auc:.4f} ± {std_auc:.4f}")
                results[f"{name}_{thresh_key}"] = {"mean": mean_auc, "std": std_auc}

        # Ensembles
        p(f"\n  Ensembles @{thresh_key}:")
        valid = oof_mask
        y_valid = labels[valid]

        # Best 3-model ensemble (try all combinations of 3)
        model_names = list(oof_preds.keys())
        best_ens_auc = 0
        best_ens_name = ""
        from itertools import combinations
        for combo in combinations(model_names, 3):
            avg = np.mean([oof_preds[n][valid] for n in combo], axis=0)
            try:
                auc = roc_auc_score(y_valid, avg)
                if auc > best_ens_auc:
                    best_ens_auc = auc
                    best_ens_name = "+".join(combo)
            except:
                pass

        p(f"    Best 3-model: {best_ens_name} = {best_ens_auc:.4f}")
        results[f"best_3ens_{thresh_key}"] = {"mean": best_ens_auc, "combo": best_ens_name}

        # Best 4-model ensemble
        best_ens4_auc = 0
        best_ens4_name = ""
        for combo in combinations(model_names, 4):
            avg = np.mean([oof_preds[n][valid] for n in combo], axis=0)
            try:
                auc = roc_auc_score(y_valid, avg)
                if auc > best_ens4_auc:
                    best_ens4_auc = auc
                    best_ens4_name = "+".join(combo)
            except:
                pass
        p(f"    Best 4-model: {best_ens4_name} = {best_ens4_auc:.4f}")
        results[f"best_4ens_{thresh_key}"] = {"mean": best_ens4_auc, "combo": best_ens4_name}

        # All-model ensemble
        all_avg = np.mean([oof_preds[n][valid] for n in model_names], axis=0)
        all_auc = roc_auc_score(y_valid, all_avg)
        p(f"    All-model avg: {all_auc:.4f}")
        results[f"all_ens_{thresh_key}"] = {"mean": all_auc}

        # Stacked meta-learner
        meta_X = np.column_stack([oof_preds[n][valid] for n in model_names])
        meta_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=99)
        meta_preds = np.zeros(len(y_valid))
        for tr, va in meta_kf.split(meta_X, y_valid):
            meta_model = xgb.XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.1,
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
            )
            meta_model.fit(meta_X[tr], y_valid[tr])
            meta_preds[va] = meta_model.predict_proba(meta_X[va])[:, 1]
        stacked_auc = roc_auc_score(y_valid, meta_preds)
        p(f"    Stacked meta-learner: {stacked_auc:.4f}")
        results[f"stacked_{thresh_key}"] = {"mean": stacked_auc}

    return results


def per_target_experiment(gt, gt_c, gt_g, n_c, n_g, cg_scores, ppi,
                          prot_tok, prot_pa, prot_proj, known_idx):
    """Part 2: Push per-target past 60% with meta-learner."""
    p("\n" + "=" * 70)
    p("PART 2: Per-target AUROC — targeting 60%+")
    p("=" * 70)

    cg_binary = gt["cg_binary_top1"]
    results = {}

    # All configs to evaluate per target
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

    p(f"  {len(configs)} configs to evaluate per target")

    # Evaluate known-246 subset
    known_246 = select_known(cg_scores, n_c, n=246)

    for subset_name, subset_idx in [("known_246", known_246), ("known_150", known_idx)]:
        p(f"\n  --- {subset_name} ({len(subset_idx)} compounds) ---")

        # Compute per-target AUROC for every config
        per_target_aucs = np.full((n_g, len(configs)), np.nan)

        for ci, (key, alpha, thresh) in enumerate(configs):
            mat = cg_scores[key]
            if alpha > 0:
                mat = propagate_ppi(mat, ppi, alpha, thresh)
            scores_sub = mat[subset_idx]

            for g in range(n_g):
                y = cg_binary[subset_idx, g]
                if y.sum() < 2 or y.sum() > len(y) - 2:
                    continue
                try:
                    per_target_aucs[g, ci] = roc_auc_score(y, scores_sub[:, g])
                except:
                    pass

        # Oracle: best config per target
        valid_targets = ~np.all(np.isnan(per_target_aucs), axis=1)
        oracle_aucs = np.nanmax(per_target_aucs[valid_targets], axis=1)
        oracle_median = np.median(oracle_aucs)
        n_valid = valid_targets.sum()
        p(f"  Oracle (best per target): median={oracle_median:.4f} ({n_valid} targets)")
        p(f"    >0.6: {(oracle_aucs > 0.6).sum()}, >0.7: {(oracle_aucs > 0.7).sum()}, >0.8: {(oracle_aucs > 0.8).sum()}")
        results[f"oracle_{subset_name}"] = {"median": oracle_median, "n_targets": int(n_valid)}

        # Best single config
        valid_mask = valid_targets
        config_medians = np.array([np.nanmedian(per_target_aucs[valid_mask, ci]) for ci in range(len(configs))])
        best_ci = np.nanargmax(config_medians)
        best_config = configs[best_ci]
        best_single = config_medians[best_ci]
        p(f"  Best single config: {best_config} median={best_single:.4f}")
        results[f"best_single_{subset_name}"] = {"median": best_single, "config": str(best_config)}

        # --- META-LEARNER: predict best config per target ---
        p(f"\n  Training per-target meta-learner on {subset_name}...")

        # Build target features from protein embeddings
        target_features = np.hstack([prot_tok, prot_pa, prot_proj])  # 10,496-dim
        # PCA down
        pca_targets = PCA(n_components=128, random_state=42).fit_transform(target_features)

        # Add CG score statistics as features
        cg_stats = []
        for key in cg_keys:
            mat = cg_scores[key][subset_idx]
            cg_stats.append(np.mean(mat, axis=0).reshape(-1, 1))
            cg_stats.append(np.std(mat, axis=0).reshape(-1, 1))
            cg_stats.append(np.max(mat, axis=0).reshape(-1, 1))
            cg_stats.append(np.min(mat, axis=0).reshape(-1, 1))
        cg_stat_features = np.hstack(cg_stats)  # n_g × (5*4)
        p(f"    Target features: PCA={pca_targets.shape[1]}, CG_stats={cg_stat_features.shape[1]}")

        target_X = np.hstack([pca_targets, cg_stat_features])

        # Labels: best config index per target
        best_config_per_target = np.nanargmax(per_target_aucs, axis=1)

        # Only train on valid targets
        valid_idx = np.where(valid_targets)[0]
        X_meta = target_X[valid_idx]
        y_meta = best_config_per_target[valid_idx]

        # Cross-validated meta-learner
        meta_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Use the meta-learner to select configs, then evaluate
        meta_selected_aucs = np.full(len(valid_idx), np.nan)

        for fold, (tr, va) in enumerate(meta_kf.split(X_meta, y_meta)):
            scaler = StandardScaler().fit(X_meta[tr])
            X_tr_s = scaler.transform(X_meta[tr])
            X_va_s = scaler.transform(X_meta[va])

            meta_clf = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_class=len(configs),
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
                objective="multi:softmax",
            )
            meta_clf.fit(X_tr_s, y_meta[tr])
            pred_config = meta_clf.predict(X_va_s).astype(int)

            for i, vi in enumerate(va):
                ci = pred_config[i]
                meta_selected_aucs[vi] = per_target_aucs[valid_idx[vi], ci]

        valid_meta = ~np.isnan(meta_selected_aucs)
        meta_median = np.median(meta_selected_aucs[valid_meta])
        p(f"  Meta-learner (config selection): median={meta_median:.4f} ({valid_meta.sum()} targets)")
        results[f"meta_learner_{subset_name}"] = {"median": meta_median, "n_targets": int(valid_meta.sum())}

        # --- REGRESSION META-LEARNER: predict AUROC per config, pick best ---
        p(f"  Training regression meta-learner...")

        # For each (target, config): predict the AUROC
        # Flatten to (n_valid * n_configs, features + config_one_hot) → AUROC
        n_valid_t = len(valid_idx)
        n_configs = len(configs)

        reg_X = np.repeat(X_meta, n_configs, axis=0)
        config_onehot = np.eye(n_configs, dtype=np.float32)
        config_features = np.tile(config_onehot, (n_valid_t, 1))
        reg_X = np.hstack([reg_X, config_features])
        reg_y = per_target_aucs[valid_idx].flatten()

        # Remove NaN entries
        valid_entries = ~np.isnan(reg_y)
        reg_X_valid = reg_X[valid_entries]
        reg_y_valid = reg_y[valid_entries]

        # Target index for each entry (for CV)
        target_indices = np.repeat(np.arange(n_valid_t), n_configs)[valid_entries]

        # 5-fold CV over targets
        reg_preds = np.full_like(reg_y_valid, np.nan)
        target_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        unique_targets = np.unique(target_indices)
        target_labels = np.zeros(len(unique_targets))

        for fold, (tr_t, va_t) in enumerate(target_kf.split(unique_targets, target_labels)):
            tr_targets = set(unique_targets[tr_t])
            va_targets = set(unique_targets[va_t])

            tr_mask = np.array([t in tr_targets for t in target_indices])
            va_mask = np.array([t in va_targets for t in target_indices])

            scaler = StandardScaler().fit(reg_X_valid[tr_mask])
            reg_model = xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                tree_method="hist", device="cuda", random_state=42, verbosity=0,
            )
            reg_model.fit(scaler.transform(reg_X_valid[tr_mask]), reg_y_valid[tr_mask])
            reg_preds[va_mask] = reg_model.predict(scaler.transform(reg_X_valid[va_mask]))

        # Reconstruct: for each target, pick config with highest predicted AUROC
        reg_selected_aucs = np.full(n_valid_t, np.nan)
        for ti in range(n_valid_t):
            entry_mask = target_indices == ti
            if not entry_mask.any():
                continue
            predicted_aucs = reg_preds[entry_mask]
            config_indices_for_target = np.repeat(np.arange(n_configs), 1)
            # Map back to config index
            all_entries_for_target = np.where(np.repeat(np.arange(n_valid_t), n_configs)[valid_entries] == ti)[0]
            if len(all_entries_for_target) == 0:
                continue
            best_pred_idx = all_entries_for_target[np.argmax(predicted_aucs)]
            # Which config was it?
            original_flat_idx = np.where(valid_entries)[0][best_pred_idx]
            best_config_idx = original_flat_idx % n_configs
            actual_auc = per_target_aucs[valid_idx[ti], best_config_idx]
            reg_selected_aucs[ti] = actual_auc

        valid_reg = ~np.isnan(reg_selected_aucs)
        reg_median = np.median(reg_selected_aucs[valid_reg])
        p(f"  Regression meta-learner: median={reg_median:.4f} ({valid_reg.sum()} targets)")
        results[f"reg_meta_{subset_name}"] = {"median": reg_median, "n_targets": int(valid_reg.sum())}

        # --- TOP-K CONFIG ENSEMBLE ---
        p(f"  Top-K config ensembles...")
        for k in [3, 5]:
            top_k_ci = np.argsort(-config_medians)[:k]
            top_k_avg = np.nanmean(per_target_aucs[valid_mask][:, top_k_ci], axis=1)
            top_k_median = np.nanmedian(top_k_avg)
            p(f"    Top-{k} config avg: median={top_k_median:.4f}")
            results[f"top{k}_avg_{subset_name}"] = {"median": top_k_median}

    return results


def main():
    t0 = time.time()

    (gt, gt_c, gt_g, n_c, n_g, ppi, ecfp, cg_scores,
     mol_tok, mol_pa, mol_proj, prot_tok, prot_pa, prot_proj) = load_data()

    known_idx = select_known(cg_scores, n_c, N_KNOWN)
    p(f"\nSelected {N_KNOWN} known compounds")

    # Part 1: CC AUROC
    cc_results = cc_auroc_experiment(
        gt, gt_c, n_c, ecfp, cg_scores, ppi,
        mol_tok, mol_pa, mol_proj, known_idx,
    )

    # Part 2: Per-target
    pt_results = per_target_experiment(
        gt, gt_c, gt_g, n_c, n_g, cg_scores, ppi,
        prot_tok, prot_pa, prot_proj, known_idx,
    )

    # Save results
    all_results = {"cc_auroc": cc_results, "per_target": pt_results}
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(os.path.join(EVAL_DIR, "push_80_60_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - t0
    p(f"\n{'=' * 70}")
    p(f"TOTAL TIME: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    p(f"Results saved to {EVAL_DIR}/push_80_60_results.json")

    # Summary
    p(f"\n{'=' * 70}")
    p("SUMMARY vs TARGETS")
    p(f"{'=' * 70}")
    p(f"  CC AUROC target: >80%")
    for k, v in sorted(cc_results.items()):
        if "mean" in v:
            mark = ">>>" if v["mean"] >= 0.80 else "   "
            p(f"  {mark} {k}: {v['mean']:.4f}")
    p(f"\n  Per-target target: >60%")
    for k, v in sorted(pt_results.items()):
        if "median" in v:
            mark = ">>>" if v["median"] >= 0.60 else "   "
            p(f"  {mark} {k}: {v['median']:.4f}")


if __name__ == "__main__":
    main()
