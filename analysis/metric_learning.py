"""Metric learning: project Bonbon embeddings → phenomics similarity space.

Learns W such that cos(W(emb_i), W(emb_j)) ≈ phenomics_sim(i,j).
Trained on GPU with compound-level CV to avoid leakage.
"""
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
EMB_BASE = "/opt/dlami/nvme/rxrx3_phenomics/dark-snowball-245"


def p(*a, **k):
    __builtins__["print"](*a, **k, flush=True) if isinstance(__builtins__, dict) else print(*a, **k, flush=True)


def load_compound_embeddings(gt_c_ids):
    dirs = {
        "tokens": os.path.join(EMB_BASE, "rxrx3_pairs_molecule_codebook_dark-snowball-245", "tokens"),
        "pooled": os.path.join(EMB_BASE, "rxrx3_pairs_molecule_codebook_dark-snowball-245", "pooled_attention"),
        "proj": os.path.join(EMB_BASE, "rxrx3_pairs_molecule_molecule_mean_dark-snowball-245"),
    }

    available = {}
    for etype, edir in dirs.items():
        available[etype] = {os.path.splitext(f)[0] for f in os.listdir(edir) if f.endswith(".pt")}
    common = set.intersection(*available.values())

    sample = list(common)[0]
    dims = {et: torch.load(os.path.join(d, f"{sample}.pt"), map_location="cpu").shape[0] for et, d in dirs.items()}
    p(f"  Dims: {dims}, {len(common)} compounds available")

    n = len(gt_c_ids)
    embs = {et: torch.zeros(n, dim) for et, dim in dims.items()}
    for i, cid in enumerate(gt_c_ids):
        if cid not in common:
            continue
        for et, d in dirs.items():
            embs[et][i] = torch.load(os.path.join(d, f"{cid}.pt"), map_location="cpu").float()
    return embs


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def get_pair_mask(pair_i, pair_j, compound_indices):
    """Vectorized: which pairs have BOTH compounds in the given set."""
    in_set = torch.zeros(max(pair_i.max(), pair_j.max(), compound_indices.max()) + 1, dtype=torch.bool)
    in_set[compound_indices] = True
    return in_set[pair_i] & in_set[pair_j]


def train_fold(model, X, cc_sim_flat, train_pairs_i, train_pairs_j, device, epochs=80, lr=1e-3, bs=16384):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_pairs = len(train_pairs_i)

    X_dev = X.to(device)
    target = cc_sim_flat.to(device)

    best_loss = float("inf")
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_pairs)
        epoch_loss = 0
        n_b = 0

        for s in range(0, n_pairs, bs):
            idx = perm[s:s + bs]
            pi, pj = train_pairs_i[idx], train_pairs_j[idx]
            ei = model(X_dev[pi])
            ej = model(X_dev[pj])
            pred = (ei * ej).sum(-1)
            loss = F.mse_loss(pred, target[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_b, 1)

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= 12:
                break

    return model


def eval_fold(model, X, gt, pair_idx_np, test_mask_np, device):
    model.eval()
    with torch.no_grad():
        proj = model(X.to(device)).cpu()
        sim = (proj @ proj.T).numpy()

    results = {}
    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                        ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
        if tkey not in gt:
            continue
        y = gt[tkey][pair_idx_np][test_mask_np]
        y_pred = sim[pair_idx_np[0][test_mask_np], pair_idx_np[1][test_mask_np]]
        if 0 < y.sum() < len(y):
            results[tname] = roc_auc_score(y, y_pred)
    return results


def main(results_dir=None, gt_dir=GT_DIR):
    device = torch.device("cuda")
    p(f"Device: {device}")

    gt = np.load(os.path.join(gt_dir, "ground_truth_binary.npz"))
    gt_c_ids = gt["compound_ids"].tolist()
    n = len(gt_c_ids)
    pair_idx_np = np.triu_indices(n, k=1)
    n_pairs = len(pair_idx_np[0])
    p(f"Compounds: {n}, Pairs: {n_pairs:,}")

    cc_sim = np.load(os.path.join(gt_dir, "phenomics_compound_compound_sim.npz"))["similarity"]

    # Pre-compute flat similarity for training pairs
    pair_i = torch.tensor(pair_idx_np[0], dtype=torch.long)
    pair_j = torch.tensor(pair_idx_np[1], dtype=torch.long)
    cc_sim_flat = torch.tensor(cc_sim[pair_idx_np], dtype=torch.float32)

    p("\nLoading embeddings...")
    embs = load_compound_embeddings(gt_c_ids)

    # Build input variants
    X_full = torch.cat([embs["tokens"], embs["pooled"], embs["proj"]], dim=-1)
    p(f"Full dim: {X_full.shape[1]}")

    pca200 = PCA(n_components=200)
    X_pca200 = torch.tensor(pca200.fit_transform(X_full.numpy()), dtype=torch.float32)
    p(f"PCA-200 variance: {pca200.explained_variance_ratio_.sum():.2%}")

    pca50 = PCA(n_components=50)
    X_pca50 = torch.tensor(pca50.fit_transform(X_full.numpy()), dtype=torch.float32)
    p(f"PCA-50 variance: {pca50.explained_variance_ratio_.sum():.2%}")

    configs = [
        ("Linear_128", X_pca200, lambda d: ProjectionMLP(d, [], 128, 0)),
        ("MLP_256_128", X_pca200, lambda d: ProjectionMLP(d, [256], 128, 0.3)),
        ("MLP_512_256_128", X_pca200, lambda d: ProjectionMLP(d, [512, 256], 128, 0.3)),
        ("MLP_1024_512_256", X_pca200, lambda d: ProjectionMLP(d, [1024, 512], 256, 0.3)),
        ("PCA50_MLP_128_64", X_pca50, lambda d: ProjectionMLP(d, [128], 64, 0.2)),
        ("Full_MLP_2048_512_128", X_full, lambda d: ProjectionMLP(d, [2048, 512], 128, 0.4)),
    ]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_results = {}

    for cfg_name, X_input, model_fn in configs:
        p(f"\n--- {cfg_name} (input_dim={X_input.shape[1]}) ---")
        fold_results = {t: [] for t in ["0.4", "0.6", "P90", "P95"]}
        t0 = time.time()

        for fold, (train_c, test_c) in enumerate(kf.split(range(n))):
            train_c_t = torch.tensor(train_c, dtype=torch.long)
            test_c_t = torch.tensor(test_c, dtype=torch.long)

            train_mask = get_pair_mask(pair_i, pair_j, train_c_t)
            test_mask = get_pair_mask(pair_i, pair_j, test_c_t)

            train_pi = pair_i[train_mask]
            train_pj = pair_j[train_mask]
            train_sim = cc_sim_flat[train_mask]
            test_mask_np = test_mask.numpy()

            model = model_fn(X_input.shape[1]).to(device)
            model = train_fold(model, X_input, train_sim, train_pi, train_pj, device)

            results = eval_fold(model, X_input, gt, pair_idx_np, test_mask_np, device)
            for t, v in results.items():
                fold_results[t].append(v)

        elapsed = time.time() - t0
        for tname, folds in fold_results.items():
            if folds:
                m, s = np.mean(folds), np.std(folds)
                all_results[f"{cfg_name}_{tname}"] = {"mean": float(m), "std": float(s)}
                p(f"  @{tname}: AUROC={m:.4f} ± {s:.4f}")
        p(f"  [{elapsed:.1f}s]")

    if results_dir:
        eval_dir = os.path.join(results_dir, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        with open(os.path.join(eval_dir, "metric_learning_results.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    p(f"\n{'='*70}")
    p("SUMMARY")
    p(f"{'='*70}")
    p(f"{'Config':<35} {'@0.4':>8} {'@0.6':>8} {'@P90':>8} {'@P95':>8}")
    p("-" * 70)
    for cfg_name, _, _ in configs:
        row = f"{cfg_name:<35}"
        for t in ["0.4", "0.6", "P90", "P95"]:
            key = f"{cfg_name}_{t}"
            if key in all_results:
                row += f" {all_results[key]['mean']:>7.4f}"
            else:
                row += "     N/A"
        p(row)
    p(f"\nXGBoost best: ~0.55-0.57")
    p(f"Beaini: 71-76% (PPI + transcriptomics, 246 known compounds)")


if __name__ == "__main__":
    rd = sys.argv[1] if len(sys.argv) > 1 else "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245"
    main(rd)
