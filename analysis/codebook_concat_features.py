"""Test concatenated codebook features for CC AUROC and per-target.

Instead of reducing protein+molecule codebook embeddings to a scalar CG score
(dot product), concatenate them into a joint vector and feed to an MLP.

Approaches:
1. Tokens concat: [prot_tok; mol_tok] = 2560-dim per pair
2. Pooled attn product: prot_pa * mol_pa = 8192-dim interaction fingerprint
3. Pooled attn concat: [prot_pa; mol_pa] = 16384-dim per pair
4. Combined: concat tokens + pooled attention product
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.home() / "cellular_experiment" / "analysis"))
from phenomics_signature import (
    load_data, select_known, extract_pairs, propagate_ppi,
    PairMLP, train_mlp_fold,
)

RESULTS_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"
DEVICE = "cuda"


def p(*a, **k):
    print(*a, **k, flush=True)


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
        loss_fn(model(X_tr_t), y_tr_t).backward()
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(X_te_t)).cpu().numpy()


def run_per_target_cv(X_all, y_all, gene_idx_all, in_dim, label):
    """Target-level 5-fold CV."""
    unique_genes = np.unique(gene_idx_all)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_aucs = []

    for fold, (tr_genes, te_genes) in enumerate(kf.split(unique_genes)):
        tr_set = set(unique_genes[tr_genes])
        te_set = set(unique_genes[te_genes])
        tr_mask = np.isin(gene_idx_all, list(tr_set))
        te_mask = np.isin(gene_idx_all, list(te_set))

        scaler = StandardScaler().fit(X_all[tr_mask])
        X_tr = scaler.transform(X_all[tr_mask])
        X_te = scaler.transform(X_all[te_mask])

        pred = train_per_target_mlp(X_tr, y_all[tr_mask], X_te, in_dim)

        te_genes_arr = gene_idx_all[te_mask]
        te_y = y_all[te_mask]
        fold_aucs = []
        for g in te_set:
            g_mask = te_genes_arr == g
            if g_mask.sum() < 5:
                continue
            y_g = te_y[g_mask]
            if y_g.sum() < 1 or y_g.sum() >= len(y_g):
                continue
            try:
                fold_aucs.append(roc_auc_score(y_g, pred[g_mask]))
            except:
                pass

        median = np.median(fold_aucs) if fold_aucs else 0
        p(f"    Fold {fold}: median={median:.4f} ({len(fold_aucs)} targets)")
        all_aucs.extend(fold_aucs)

    overall = np.median(all_aucs)
    p(f"    {label}: median={overall:.4f} ({len(all_aucs)} targets)")
    return overall


def main():
    p("=" * 70)
    p("CONCATENATED CODEBOOK FEATURES")
    p("=" * 70)

    t0 = time.time()

    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores = data[:7]
    mol_tok, mol_pa, mol_proj = data[7:10]
    prot_tok, prot_pa, prot_proj = data[10:13]
    cc_sim = gt["cc_similarity"]
    cg_binary = gt["cg_binary_top1"]

    known_idx = select_known(cg_scores, 150)
    ki = known_idx
    n_known = len(ki)
    p(f"  {n_c} compounds, {n_g} genes, {n_known} known")
    p(f"  mol_tok: {mol_tok.shape}, prot_tok: {prot_tok.shape}")
    p(f"  mol_pa: {mol_pa.shape}, prot_pa: {prot_pa.shape}")

    # ================================================================
    # PART 1: CC AUROC with concatenated features
    # ================================================================
    p(f"\n{'='*70}")
    p("PART 1: CC AUROC — Pair-level CV @0.6")
    p(f"{'='*70}")

    # Build pair features from tokens concat
    # For CC: each pair is (compound_i, compound_j)
    # Interaction profile of compound i = [prot[1]*mol[i], prot[2]*mol[i], ...]
    # But for CC AUROC, the question is whether two compounds are similar
    # The current features already capture CG profiles, signatures, fusion

    # For CC AUROC, the codebook concat doesn't directly apply (it's per CG pair)
    # Let's focus on per-target where it's more relevant

    # ================================================================
    # PART 2: Per-target with concatenated codebook features
    # ================================================================
    p(f"\n{'='*70}")
    p("PART 2: Per-target — concatenated codebook features")
    p(f"{'='*70}")

    results = {}

    # Strategy 1: Tokens concat [prot_tok; mol_tok] = 2560-dim
    p("\n--- Strategy 1: Tokens concat (2560-dim) ---")
    all_X, all_y, all_gi = [], [], []
    for gi in range(n_g):
        y = cg_binary[ki, gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        # Each known compound gets [prot_tok[gi]; mol_tok[ci]]
        prot_vec = prot_tok[gi]  # (1280,)
        X_pairs = np.hstack([
            np.tile(prot_vec, (n_known, 1)),  # (n_known, 1280) — same protein for all
            mol_tok[ki]  # (n_known, 1280)
        ])
        all_X.append(X_pairs)
        all_y.append(y)
        all_gi.extend([gi] * n_known)

    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    all_gi = np.array(all_gi)
    p(f"  Dataset: {all_X.shape}, pos rate: {all_y.mean():.4f}")

    results["tokens_concat_2560"] = run_per_target_cv(all_X, all_y, all_gi, 2560, "Tokens concat")

    # Strategy 2: Pooled attention element-wise product (8192-dim)
    p("\n--- Strategy 2: Pooled attention product (8192-dim) ---")
    all_X2, all_y2, all_gi2 = [], [], []
    for gi in range(n_g):
        y = cg_binary[ki, gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        prot_vec = prot_pa[gi]  # (8192,)
        # Element-wise product: which codebook entries co-activate
        X_pairs = mol_pa[ki] * prot_vec[np.newaxis, :]
        all_X2.append(X_pairs)
        all_y2.append(y)
        all_gi2.extend([gi] * n_known)

    all_X2 = np.vstack(all_X2)
    all_y2 = np.concatenate(all_y2)
    all_gi2 = np.array(all_gi2)
    p(f"  Dataset: {all_X2.shape}, pos rate: {all_y2.mean():.4f}")

    results["pa_product_8192"] = run_per_target_cv(all_X2, all_y2, all_gi2, 8192, "PA product")

    # Strategy 3: Tokens concat + PA product combined
    p("\n--- Strategy 3: Tokens concat + PA product (10752-dim) ---")
    all_X3 = np.hstack([all_X, all_X2])
    p(f"  Dataset: {all_X3.shape}")
    results["combined_10752"] = run_per_target_cv(all_X3, all_y, all_gi, 10752, "Combined")

    # Strategy 4: Just molecule tokens (1280-dim) — ablation
    p("\n--- Strategy 4: Molecule tokens only (1280-dim, ablation) ---")
    all_X4, all_y4, all_gi4 = [], [], []
    for gi in range(n_g):
        y = cg_binary[ki, gi]
        if y.sum() < 2 or y.sum() > len(y) - 2:
            continue
        all_X4.append(mol_tok[ki])
        all_y4.append(y)
        all_gi4.extend([gi] * n_known)

    all_X4 = np.vstack(all_X4)
    all_y4 = np.concatenate(all_y4)
    all_gi4 = np.array(all_gi4)
    results["mol_tokens_1280"] = run_per_target_cv(all_X4, all_y4, all_gi4, 1280, "Mol tokens only")

    # ================================================================
    # SUMMARY
    # ================================================================
    p(f"\n{'='*70}")
    p("SUMMARY — Per-target median AUROC (target-CV)")
    p(f"{'='*70}")

    # Reference baselines
    p(f"\n  {'Method':45s} {'Median':>8s}")
    p(f"  {'-'*53}")
    p(f"  {'CG score baseline (from prior exp)':45s} {'0.5726':>8s}")
    p(f"  {'Fusion CLS 1024-dim MLP (from prior exp)':45s} {'0.7770':>8s}")
    p(f"  {'Fusion CLS+CG combined (from prior exp)':45s} {'0.7855':>8s}")
    for name, val in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ***" if val > 0.5726 else ""
        p(f"  {name:45s} {val:8.4f}{marker}")

    p(f"\n  Time: {time.time() - t0:.0f}s")

    out = os.path.join(RESULTS_DIR, "codebook_concat_results.json")
    json.dump(results, open(out, "w"), indent=2)
    p(f"  Saved to {out}")


if __name__ == "__main__":
    main()
