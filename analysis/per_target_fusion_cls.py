"""Per-target AUROC using full 1024-dim fusion CLS vectors.

Instead of collapsing the CLS to a scalar norm (which destroys directional
information), use the full 1024-dim vector per (compound, target) pair.

Approaches:
1. Zero-shot: cosine similarity between compound's CLS vector and a learned
   "active" direction per target (mean CLS of known actives)
2. Trained: global MLP taking 1024-dim CLS as input, trained across all targets
3. Combined: CLS features + existing CG score features
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.home() / "cellular_experiment" / "analysis"))
from phenomics_signature import load_data, select_known, propagate_ppi

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
FUSION_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/fusion_cls")
RESULTS_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"
DEVICE = "cuda"


def p(*a, **k):
    print(*a, **k, flush=True)


def load_ground_truth():
    gt = dict(np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"), allow_pickle=True))
    for k in gt:
        gt[k] = np.array(gt[k])
    gt_c = gt["compound_ids"].tolist()
    gt_g = gt["gene_ids"].tolist()
    return gt, gt_c, gt_g


def load_fusion_cls(gt_c, gt_g):
    mol_index = json.load(open(FUSION_DIR / "molecule_index.json"))
    mol_to_idx = {m: i for i, m in enumerate(mol_index)}

    compound_idx = []
    valid_compounds = []
    for i, c in enumerate(gt_c):
        if c in mol_to_idx:
            compound_idx.append(mol_to_idx[c])
            valid_compounds.append(i)
    compound_idx = np.array(compound_idx)
    valid_compounds = np.array(valid_compounds)
    p(f"  {len(valid_compounds)}/{len(gt_c)} compounds matched")

    fusion_data = {}
    for gene in gt_g:
        fpath = FUSION_DIR / f"{gene}.pt"
        if fpath.exists():
            cls_mat = torch.load(fpath, map_location="cpu", weights_only=True)
            fusion_data[gene] = cls_mat[compound_idx].numpy()

    p(f"  {len(fusion_data)}/{len(gt_g)} proteins loaded")
    return fusion_data, valid_compounds


class PerTargetMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims=(512, 256)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_per_target_mlp(X_tr, y_tr, X_te, in_dim, epochs=80, lr=1e-3):
    model = PerTargetMLP(in_dim, hidden_dims=(512, 256)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pos_w = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=DEVICE))

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=DEVICE)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(X_te_t)).cpu().numpy()
    return pred


def main():
    p("=" * 70)
    p("PER-TARGET AUROC WITH FULL 1024-DIM FUSION CLS")
    p("=" * 70)

    t0 = time.time()

    # Load data
    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores = data[:7]
    cg_binary = gt["cg_binary_top1"]

    p("\nLoading fusion CLS...")
    fusion_data, valid_compounds = load_fusion_cls(gt_c, gt_g)
    n_valid = len(valid_compounds)

    # Build the full CLS matrix: [n_compounds, n_genes, 1024]
    hidden_dim = next(iter(fusion_data.values())).shape[1]
    p(f"  Hidden dim: {hidden_dim}, compounds: {n_valid}, genes: {len(fusion_data)}")

    known_idx_150 = select_known(cg_scores, 150)
    # Map known_idx to valid_compounds indices
    valid_set = set(valid_compounds.tolist())
    known_valid = [i for i, k in enumerate(valid_compounds) if k in set(known_idx_150.tolist())]
    known_valid = np.array(known_valid)
    p(f"  Known compounds in fusion data: {len(known_valid)}")

    # ================================================================
    # EXPERIMENT 1: Zero-shot with CLS direction per target
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 1: Zero-shot — CLS cosine with active centroid")
    p(f"{'='*70}")

    target_aucs_zs = []
    target_names_zs = []

    for gi, gene in enumerate(gt_g):
        if gene not in fusion_data:
            continue
        cls_mat = fusion_data[gene]  # [n_valid, 1024]
        cls_known = cls_mat[known_valid]  # [n_known, 1024]

        y = cg_binary[valid_compounds[known_valid], gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue

        # Active centroid: mean CLS of compounds that phenocopy this target
        active_mask = y == 1
        inactive_mask = y == 0
        active_centroid = cls_known[active_mask].mean(axis=0)
        active_centroid /= np.linalg.norm(active_centroid) + 1e-8

        # Score: cosine similarity to active centroid
        norms = np.linalg.norm(cls_known, axis=1, keepdims=True) + 1e-8
        scores = (cls_known / norms) @ active_centroid

        try:
            auc = roc_auc_score(y, scores)
            target_aucs_zs.append(auc)
            target_names_zs.append(gene)
        except:
            pass

    median_zs = np.median(target_aucs_zs)
    p(f"  Zero-shot CLS cosine: median={median_zs:.4f} ({len(target_aucs_zs)} targets)")

    # LOO variant: leave-one-out centroid to avoid info leak
    p("\n  LOO variant (leave-one-out centroid)...")
    target_aucs_loo = []

    for gi, gene in enumerate(gt_g):
        if gene not in fusion_data:
            continue
        cls_mat = fusion_data[gene]
        cls_known = cls_mat[known_valid]

        y = cg_binary[valid_compounds[known_valid], gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue

        active_mask = y == 1
        n_active = active_mask.sum()
        if n_active < 2:
            continue

        # LOO: for each compound, compute centroid without it
        scores = np.zeros(len(y))
        for i in range(len(y)):
            if active_mask[i]:
                others = cls_known[active_mask & (np.arange(len(y)) != i)]
            else:
                others = cls_known[active_mask]
            centroid = others.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            norm_i = np.linalg.norm(cls_known[i]) + 1e-8
            scores[i] = (cls_known[i] / norm_i) @ centroid

        try:
            auc = roc_auc_score(y, scores)
            target_aucs_loo.append(auc)
        except:
            pass

    median_loo = np.median(target_aucs_loo)
    p(f"  LOO CLS cosine: median={median_loo:.4f} ({len(target_aucs_loo)} targets)")

    # ================================================================
    # EXPERIMENT 2: CLS norm (baseline — what we had before)
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 2: CLS norm baseline (scalar)")
    p(f"{'='*70}")

    target_aucs_norm = []
    for gi, gene in enumerate(gt_g):
        if gene not in fusion_data:
            continue
        cls_known = fusion_data[gene][known_valid]
        y = cg_binary[valid_compounds[known_valid], gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        scores = np.linalg.norm(cls_known, axis=1)
        try:
            auc = roc_auc_score(y, scores)
            target_aucs_norm.append(auc)
        except:
            pass

    median_norm = np.median(target_aucs_norm)
    p(f"  CLS norm: median={median_norm:.4f} ({len(target_aucs_norm)} targets)")

    # ================================================================
    # EXPERIMENT 3: CG score baseline (best single config)
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 3: CG score baseline (proj_raw, alpha=0.7)")
    p(f"{'='*70}")

    mat = propagate_ppi(cg_scores["proj_raw"], ppi, alpha=0.7, threshold=0.0)
    target_aucs_cg = []
    for gi in range(n_g):
        y = cg_binary[known_idx_150, gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        try:
            auc = roc_auc_score(y, mat[known_idx_150, gi])
            target_aucs_cg.append(auc)
        except:
            pass

    median_cg = np.median(target_aucs_cg)
    p(f"  CG score (proj_raw a=0.7): median={median_cg:.4f} ({len(target_aucs_cg)} targets)")

    # ================================================================
    # EXPERIMENT 4: Global MLP trained on (CLS → phenocopy)
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 4: Global MLP on 1024-dim CLS")
    p(f"{'='*70}")

    # Build global dataset: each sample is (CLS_vector, phenocopy_label)
    all_X = []
    all_y = []
    all_gene_idx = []

    for gi, gene in enumerate(gt_g):
        if gene not in fusion_data:
            continue
        cls_known = fusion_data[gene][known_valid]
        y = cg_binary[valid_compounds[known_valid], gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        all_X.append(cls_known)
        all_y.append(y)
        all_gene_idx.extend([gi] * len(y))

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    all_gene_idx = np.array(all_gene_idx)

    p(f"  Global dataset: {len(all_X)} samples, {len(np.unique(all_gene_idx))} targets")
    p(f"  Positive rate: {all_y.mean():.4f}")

    # Target-level 5-fold CV (split by target, not sample)
    unique_genes = np.unique(all_gene_idx)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_target_aucs = []
    for fold, (tr_genes, te_genes) in enumerate(kf.split(unique_genes)):
        tr_gene_set = set(unique_genes[tr_genes])
        te_gene_set = set(unique_genes[te_genes])

        tr_mask = np.isin(all_gene_idx, list(tr_gene_set))
        te_mask = np.isin(all_gene_idx, list(te_gene_set))

        scaler = StandardScaler().fit(all_X[tr_mask])
        X_tr = scaler.transform(all_X[tr_mask])
        X_te = scaler.transform(all_X[te_mask])

        pred = train_per_target_mlp(X_tr, all_y[tr_mask], X_te, hidden_dim)

        # Evaluate per target in test set
        te_gene_ids = all_gene_idx[te_mask]
        te_y = all_y[te_mask]
        target_aucs_fold = []
        for g in te_gene_set:
            g_mask = te_gene_ids == g
            if g_mask.sum() < 5:
                continue
            y_g = te_y[g_mask]
            if y_g.sum() < 1 or y_g.sum() >= len(y_g):
                continue
            try:
                auc = roc_auc_score(y_g, pred[g_mask])
                target_aucs_fold.append(auc)
            except:
                pass

        fold_median = np.median(target_aucs_fold) if target_aucs_fold else 0
        p(f"  Fold {fold}: median={fold_median:.4f} ({len(target_aucs_fold)} targets)")
        fold_target_aucs.extend(target_aucs_fold)

    global_mlp_median = np.median(fold_target_aucs)
    p(f"  Global MLP (target-CV): median={global_mlp_median:.4f} ({len(fold_target_aucs)} targets)")

    # ================================================================
    # EXPERIMENT 5: CLS + CG combined features
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 5: CLS 1024-dim + CG scores (5 keys) combined")
    p(f"{'='*70}")

    cg_keys = ["proj_raw", "proj_cos", "cb_pa_raw", "cb_pa_cos", "cb_tok_raw"]
    all_X_combined = []
    all_y_combined = []
    all_gene_idx_combined = []

    for gi, gene in enumerate(gt_g):
        if gene not in fusion_data:
            continue
        cls_known = fusion_data[gene][known_valid]
        y = cg_binary[valid_compounds[known_valid], gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue

        # CG scores for this target (with PPI propagation)
        cg_feats = []
        for key in cg_keys:
            mat = propagate_ppi(cg_scores[key], ppi, alpha=0.5, threshold=0.0)
            cg_feats.append(mat[valid_compounds[known_valid], gi:gi+1])
        cg_feat_mat = np.hstack(cg_feats)  # [n_known, 5]

        combined = np.hstack([cls_known, cg_feat_mat])  # [n_known, 1029]
        all_X_combined.append(combined)
        all_y_combined.append(y)
        all_gene_idx_combined.extend([gi] * len(y))

    all_X_combined = np.vstack(all_X_combined)
    all_y_combined = np.concatenate(all_y_combined)
    all_gene_idx_combined = np.array(all_gene_idx_combined)

    p(f"  Combined dataset: {len(all_X_combined)} samples, {all_X_combined.shape[1]} features")

    unique_genes_c = np.unique(all_gene_idx_combined)
    fold_target_aucs_c = []
    for fold, (tr_genes, te_genes) in enumerate(kf.split(unique_genes_c)):
        tr_gene_set = set(unique_genes_c[tr_genes])
        te_gene_set = set(unique_genes_c[te_genes])

        tr_mask = np.isin(all_gene_idx_combined, list(tr_gene_set))
        te_mask = np.isin(all_gene_idx_combined, list(te_gene_set))

        scaler = StandardScaler().fit(all_X_combined[tr_mask])
        X_tr = scaler.transform(all_X_combined[tr_mask])
        X_te = scaler.transform(all_X_combined[te_mask])

        pred = train_per_target_mlp(X_tr, all_y_combined[tr_mask], X_te, all_X_combined.shape[1])

        te_gene_ids = all_gene_idx_combined[te_mask]
        te_y = all_y_combined[te_mask]
        target_aucs_fold = []
        for g in te_gene_set:
            g_mask = te_gene_ids == g
            if g_mask.sum() < 5:
                continue
            y_g = te_y[g_mask]
            if y_g.sum() < 1 or y_g.sum() >= len(y_g):
                continue
            try:
                auc = roc_auc_score(y_g, pred[g_mask])
                target_aucs_fold.append(auc)
            except:
                pass

        fold_median = np.median(target_aucs_fold) if target_aucs_fold else 0
        p(f"  Fold {fold}: median={fold_median:.4f} ({len(target_aucs_fold)} targets)")
        fold_target_aucs_c.extend(target_aucs_fold)

    combined_median = np.median(fold_target_aucs_c)
    p(f"  CLS+CG combined (target-CV): median={combined_median:.4f} ({len(fold_target_aucs_c)} targets)")

    # ================================================================
    # EXPERIMENT 6: Per-target CLS cosine (rank compounds by CLS similarity
    #               to other actives for the SAME target)
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 6: Per-target pairwise CLS cosine (mean sim to actives)")
    p(f"{'='*70}")

    target_aucs_pairwise = []
    for gi, gene in enumerate(gt_g):
        if gene not in fusion_data:
            continue
        cls_known = fusion_data[gene][known_valid]
        y = cg_binary[valid_compounds[known_valid], gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue

        # Normalize
        norms = np.linalg.norm(cls_known, axis=1, keepdims=True) + 1e-8
        cls_normed = cls_known / norms

        # Pairwise cosine
        cos_sim = cls_normed @ cls_normed.T

        # LOO: for each compound, mean cosine to OTHER actives
        active_mask = y == 1
        scores = np.zeros(len(y))
        for i in range(len(y)):
            other_actives = active_mask.copy()
            other_actives[i] = False
            if other_actives.sum() == 0:
                scores[i] = 0
                continue
            scores[i] = cos_sim[i, other_actives].mean()

        try:
            auc = roc_auc_score(y, scores)
            target_aucs_pairwise.append(auc)
        except:
            pass

    median_pairwise = np.median(target_aucs_pairwise)
    p(f"  Pairwise CLS cosine (LOO): median={median_pairwise:.4f} ({len(target_aucs_pairwise)} targets)")

    # ================================================================
    # SUMMARY
    # ================================================================
    p(f"\n{'='*70}")
    p("SUMMARY — Per-target median AUROC")
    p(f"{'='*70}")

    results = {
        "cg_score_baseline": median_cg,
        "cls_norm_scalar": median_norm,
        "cls_cosine_centroid": median_zs,
        "cls_cosine_loo": median_loo,
        "cls_pairwise_cosine_loo": median_pairwise,
        "global_mlp_1024": global_mlp_median,
        "cls_cg_combined_mlp": combined_median,
    }

    p(f"\n  {'Method':45s} {'Median':>8s}")
    p(f"  {'-'*53}")
    for name, val in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ***" if val > median_cg else ""
        p(f"  {name:45s} {val:8.4f}{marker}")

    p(f"\n  CG baseline (proj_raw a=0.7): {median_cg:.4f}")
    p(f"  Time: {time.time() - t0:.0f}s")

    out = os.path.join(RESULTS_DIR, "per_target_fusion_cls_results.json")
    json.dump(results, open(out, "w"), indent=2)
    p(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
