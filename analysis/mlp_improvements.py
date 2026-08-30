"""MLP architecture improvements for phenomics prediction.

Tests in order:
1. Wider/shallower architectures
2. Residual blocks + GELU/SiLU
3. FT-Transformer
4. Multi-task loss (classification + regression)
5. Focal loss

All evaluated on pair-level CV @0.6 with full feature set (signatures + fusion).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.home() / "cellular_experiment" / "analysis"))
from phenomics_signature import (
    load_data, build_original_42_features, select_known,
    build_signature_features, build_cg_signature_features,
    extract_pairs,
)
from fusion_cls_integration import load_ground_truth, load_fusion_cls, build_fusion_features

RESULTS_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"
DEVICE = "cuda"


def p(*a, **k):
    print(*a, **k, flush=True)


# ============================================================
# Architecture 1: Original MLP (baseline)
# ============================================================
class OriginalMLP(nn.Module):
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


# ============================================================
# Architecture 2: Wider/shallower MLP
# ============================================================
class WiderMLP(nn.Module):
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


# ============================================================
# Architecture 3: Residual MLP with GELU
# ============================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class ResidualMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, n_blocks=3, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x).squeeze(-1)


# ============================================================
# Architecture 4: FT-Transformer
# ============================================================
class FeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        # x: [batch, n_features] → [batch, n_features, d_token]
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FTTransformer(nn.Module):
    def __init__(self, n_features, d_token=64, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=n_heads, dim_feedforward=d_token * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x):
        tokens = self.tokenizer(x)  # [batch, n_feat, d_token]
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # [batch, 1+n_feat, d_token]
        encoded = self.encoder(tokens)
        cls_out = encoded[:, 0]  # [batch, d_token]
        return self.head(cls_out).squeeze(-1)


# ============================================================
# Architecture 5: Multi-task MLP
# ============================================================
class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, dropout=0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.cls_head = nn.Linear(prev, 1)
        self.reg_head = nn.Linear(prev, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)


# ============================================================
# Losses
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ============================================================
# Training functions
# ============================================================
def train_model_fold(model, X_tr, y_tr, X_te, y_te, y_cont_tr=None,
                     epochs=120, lr=1e-3, bs=4096, loss_type="bce",
                     focal_gamma=2.0, multi_task_weight=0.5):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_w = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)],
                          dtype=torch.float32).to(DEVICE)

    if loss_type == "bce":
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    elif loss_type == "focal":
        criterion = FocalLoss(gamma=focal_gamma, pos_weight=pos_w)
    elif loss_type == "multi_task":
        cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        reg_criterion = nn.MSELoss()

    X_tr_d = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    y_tr_d = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
    X_te_d = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    y_cont_d = torch.tensor(y_cont_tr, dtype=torch.float32).to(DEVICE) if y_cont_tr is not None else None

    best_auc, best_preds = 0, None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        for s in range(0, len(X_tr), bs):
            idx = perm[s:s + bs]
            if loss_type == "multi_task":
                cls_logits, reg_pred = model(X_tr_d[idx])
                loss = cls_criterion(cls_logits, y_tr_d[idx])
                if y_cont_d is not None:
                    loss += multi_task_weight * reg_criterion(reg_pred, y_cont_d[idx])
            else:
                logits = model(X_tr_d[idx])
                loss = criterion(logits, y_tr_d[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            if loss_type == "multi_task":
                cls_logits, reg_pred = model(X_te_d)
                preds = torch.sigmoid(cls_logits).cpu().numpy()
            else:
                preds = torch.sigmoid(model(X_te_d)).cpu().numpy()
        auc = roc_auc_score(y_te, preds)
        if auc > best_auc:
            best_auc = auc
            best_preds = preds.copy()

    return best_preds, best_auc


def run_experiment(name, model_factory, X, y, y_cont, n_splits=5, **train_kwargs):
    p(f"\n  --- {name} ---")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (tr, te) in enumerate(kf.split(X)):
        y_tr, y_te = y[tr], y[te]
        if len(np.unique(y_te)) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        X_tr_s = scaler.transform(X[tr])
        X_te_s = scaler.transform(X[te])

        model = model_factory(X_tr_s.shape[1])
        y_cont_tr = y_cont[tr] if y_cont is not None else None
        _, auc = train_model_fold(model, X_tr_s, y_tr, X_te_s, y_te,
                                   y_cont_tr=y_cont_tr, **train_kwargs)
        fold_aucs.append(auc)
        p(f"    Fold {fold}: {auc:.4f}")

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    p(f"    Mean: {mean_auc:.4f} +/- {std_auc:.4f}")
    return {"mean": mean_auc, "std": std_auc}


def main():
    p("=" * 70)
    p("MLP ARCHITECTURE IMPROVEMENTS")
    p("=" * 70)

    # Load data
    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores = data[:7]
    mol_tok, mol_pa, mol_proj = data[7:10]
    cc_sim = gt["cc_similarity"]
    known_idx = select_known(cg_scores, 150)

    # Build full feature set (baseline + signatures + fusion)
    baseline = build_original_42_features(cg_scores, ppi, gt_c, n_c)
    sig = build_signature_features(mol_tok, mol_pa, mol_proj, cc_sim, known_idx)
    cg_sig = build_cg_signature_features(cg_scores, ppi, cc_sim, known_idx)
    gt2, _, _ = load_ground_truth()
    fusion_data, valid_compounds = load_fusion_cls(gt_c, gt_g)
    fusion = build_fusion_features(fusion_data, gt_g, n_c, valid_compounds)

    all_features = {**baseline, **sig, **cg_sig, **fusion}
    n_feat = len(all_features)
    p(f"  Total features: {n_feat}")

    # Extract pairs
    ki = known_idx
    X, names, pair_idx = extract_pairs(all_features, ki)
    sub_sim = cc_sim[np.ix_(ki, ki)]
    y_06 = (sub_sim[pair_idx] > 0.6).astype(np.float32)
    y_cont = sub_sim[pair_idx].astype(np.float32)
    p(f"  Pairs: {len(y_06)}, Positive rate @0.6: {y_06.mean():.3f}")

    results = {}

    # ============================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 1: Original MLP baseline")
    p(f"{'='*70}")
    results["original_1024_512_256_128"] = run_experiment(
        "Original [1024,512,256,128]",
        lambda d: OriginalMLP(d, [1024, 512, 256, 128]),
        X, y_06, y_cont,
    )

    # ============================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 2: Wider/shallower architectures")
    p(f"{'='*70}")

    for arch_name, dims in [
        ("wide_2048_1024", [2048, 1024]),
        ("wide_2048_1024_512", [2048, 1024, 512]),
        ("wide_4096_2048", [4096, 2048]),
    ]:
        results[arch_name] = run_experiment(
            f"Wider {dims}",
            lambda d, dims=dims: WiderMLP(d, dims),
            X, y_06, y_cont,
        )

    # ============================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 3: Residual + GELU")
    p(f"{'='*70}")

    for n_blocks, hdim in [(2, 1024), (3, 1024), (3, 2048)]:
        name = f"residual_{hdim}x{n_blocks}"
        results[name] = run_experiment(
            f"Residual {hdim}×{n_blocks} blocks + GELU",
            lambda d, h=hdim, n=n_blocks: ResidualMLP(d, hidden_dim=h, n_blocks=n),
            X, y_06, y_cont,
        )

    # ============================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 4: FT-Transformer")
    p(f"{'='*70}")

    for d_tok, n_heads, n_layers in [(64, 4, 3), (128, 4, 3), (64, 4, 6)]:
        name = f"ft_transformer_d{d_tok}_h{n_heads}_l{n_layers}"
        results[name] = run_experiment(
            f"FT-Transformer d={d_tok} h={n_heads} L={n_layers}",
            lambda d, dt=d_tok, nh=n_heads, nl=n_layers: FTTransformer(d, d_token=dt, n_heads=nh, n_layers=nl),
            X, y_06, y_cont,
            epochs=150, lr=5e-4,
        )

    # ============================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 5: Multi-task loss (classification + regression)")
    p(f"{'='*70}")

    for weight in [0.3, 0.5, 1.0]:
        name = f"multitask_w{weight}"
        results[name] = run_experiment(
            f"Multi-task MLP (reg weight={weight})",
            lambda d: MultiTaskMLP(d, [1024, 512, 256, 128]),
            X, y_06, y_cont,
            loss_type="multi_task", multi_task_weight=weight,
        )

    # ============================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 6: Focal loss")
    p(f"{'='*70}")

    for gamma in [1.0, 2.0, 3.0]:
        name = f"focal_g{gamma}"
        results[name] = run_experiment(
            f"Focal loss (gamma={gamma})",
            lambda d: OriginalMLP(d, [1024, 512, 256, 128]),
            X, y_06, y_cont,
            loss_type="focal", focal_gamma=gamma,
        )

    # ============================================================
    p(f"\n{'='*70}")
    p("RESULTS SUMMARY")
    p(f"{'='*70}")
    p(f"\n  {'Model':45s} {'Mean':>8s} {'Std':>8s}")
    p(f"  {'-'*61}")
    for name, res in sorted(results.items(), key=lambda x: -x[1]["mean"]):
        marker = " ***" if res["mean"] > results["original_1024_512_256_128"]["mean"] else ""
        p(f"  {name:45s} {res['mean']:8.4f} {res['std']:8.4f}{marker}")

    ensemble_baseline = results["original_1024_512_256_128"]["mean"]
    p(f"\n  Original MLP baseline: {ensemble_baseline:.4f}")
    p(f"  Target (ensemble): 0.8115")

    out_path = os.path.join(RESULTS_DIR, "mlp_improvements_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    p(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
