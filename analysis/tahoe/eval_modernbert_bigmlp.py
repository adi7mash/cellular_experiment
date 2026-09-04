"""
ModernBERT + large MLP sweep for Tahoe expression prediction.
Tests bigger models at 5%, 10%, 20%, 50%, 100% data fractions.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from transformers import ModernBertConfig, ModernBertModel
import time
import warnings
warnings.filterwarnings('ignore')

BASE = "/opt/dlami/nvme/tahoe"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def p(*a, **k):
    print(*a, **k, flush=True)

# ─── Load data ───
pb = np.load(f"{BASE}/pseudobulk_means.npz", allow_pickle=True)
means = pb['means']
drug_names = pb['drug_names']
cell_line_ids = pb['cell_line_ids']
cell_counts = pb['cell_counts']
gene_indices = pb['gene_indices']
control_means = pb['control_means']
control_cell_lines = pb['control_cell_lines']

ctrl_map = {cl: control_means[i] for i, cl in enumerate(control_cell_lines)}

cg = np.load(f"{BASE}/tahoe_cg_profiles.npz", allow_pickle=True)
cg_profiles = cg['cg_profiles']
cg_drugs = list(cg['drugs'])

drug_name_map = {}
for td in np.unique(drug_names):
    for i, cd in enumerate(cg_drugs):
        if cd.lower().strip() == td.lower().strip():
            drug_name_map[td] = i
            break

valid_mask = np.array([(dn in drug_name_map) and (cl in ctrl_map) for dn, cl in zip(drug_names, cell_line_ids)])
valid_mask &= (cell_counts >= 10)
valid_idx = np.where(valid_mask)[0]

X_cg = np.array([cg_profiles[drug_name_map[drug_names[i]]] for i in valid_idx])
Y_expr = means[valid_idx]
Y_ctrl = np.array([ctrl_map[cell_line_ids[i]] for i in valid_idx])
Y_delta = Y_expr - Y_ctrl
valid_drugs = drug_names[valid_idx]
valid_cls = cell_line_ids[valid_idx]

n = len(valid_idx)
delta_var = np.var(Y_delta, axis=0)
n_hvg = min(2000, (delta_var > 1e-10).sum())
hvg_idx = np.argsort(-delta_var)[:n_hvg]
Y_delta_hvg = Y_delta[:, hvg_idx]

# Cell-line features
unique_cls_list = sorted(set(valid_cls))
cl_to_idx = {cl: i for i, cl in enumerate(unique_cls_list)}
cl_onehot = np.zeros((n, len(unique_cls_list)))
for i, cl in enumerate(valid_cls):
    cl_onehot[i, cl_to_idx[cl]] = 1.0
X_cg_cl = np.hstack([X_cg, cl_onehot])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

p(f"Data: {n} conditions, {n_hvg} HVGs, {X_cg.shape[1]} CG dims")
p(f"Device: {device}")

def eval_pearson_delta(pred, true):
    corrs = []
    for i in range(pred.shape[0]):
        if np.std(pred[i]) < 1e-8 or np.std(true[i]) < 1e-8:
            continue
        r, _ = pearsonr(pred[i], true[i])
        if not np.isnan(r):
            corrs.append(r)
    return np.mean(corrs) if corrs else float('nan')

def eval_top_deg(pred, true, top_k=50):
    corrs = []
    for i in range(pred.shape[0]):
        abs_delta = np.abs(true[i])
        top_idx = np.argsort(-abs_delta)[:top_k]
        if np.std(pred[i, top_idx]) < 1e-8 or np.std(true[i, top_idx]) < 1e-8:
            continue
        r, _ = pearsonr(pred[i, top_idx], true[i, top_idx])
        if not np.isnan(r):
            corrs.append(r)
    return np.mean(corrs) if corrs else float('nan')

def report(name, n_train, pd, deg, elapsed):
    p(f"  {name:60s} | n={n_train:4d} | Pearson Δ: {pd:.4f} | DEG-50: {deg:.4f} | {elapsed:.1f}s")

# ═══════════════════════════════════════════════════════════════
# ModernBERT wrapper: chunk CG profile into tokens, run through
# ModernBERT encoder, decode CLS to expression deltas
# ═══════════════════════════════════════════════════════════════
class ModernBERTRegressor(nn.Module):
    def __init__(self, cg_dim, n_out, hidden_size=256, num_layers=4, num_heads=4,
                 intermediate_size=1024, chunk_size=63, dropout=0.3):
        super().__init__()
        self.chunk_size = chunk_size
        n_chunks = (cg_dim + chunk_size - 1) // chunk_size
        self.pad_to = n_chunks * chunk_size
        self.n_chunks = n_chunks

        config = ModernBertConfig(
            vocab_size=2,  # not used — we project directly
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=n_chunks + 1,
            hidden_dropout_prob=dropout,
            attention_dropout=dropout,
            classifier_dropout=dropout,
        )
        self.encoder = ModernBertModel(config)
        # Replace the embedding layer with our projection
        self.input_proj = nn.Linear(chunk_size, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, n_out),
        )

    def forward(self, x):
        B = x.size(0)
        if x.size(1) < self.pad_to:
            x = F.pad(x, (0, self.pad_to - x.size(1)))
        x = x.view(B, self.n_chunks, self.chunk_size)
        x = self.input_proj(x)  # [B, n_chunks, hidden_size]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, n_chunks+1, hidden_size]

        # ModernBERT expects input_ids but we bypass embeddings
        # Use the encoder layers directly
        out = self.encoder(inputs_embeds=x).last_hidden_state
        return self.head(out[:, 0])  # CLS token


def train_model(X_feat, Y_target, frac, model_fn, epochs, lr, wd, name, batch_size=None):
    """Generic trainer with optional mini-batching for larger models."""
    X_in = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(Y_target, dtype=torch.float32)
    all_preds = torch.zeros_like(Y_t)
    t0 = time.time()

    for fold, (tr, te) in enumerate(kf.split(X_feat)):
        n_sub = max(2, int(len(tr) * frac))
        np.random.seed(fold + 42)
        sub = np.random.choice(tr, size=n_sub, replace=False)

        x_sub = X_in[sub].to(device)
        y_sub = Y_t[sub].to(device)
        x_te = X_in[te].to(device)

        mu, std = x_sub.mean(0), x_sub.std(0).clamp(min=1e-8)
        x_sub = (x_sub - mu) / std
        x_te = (x_te - mu) / std

        model = model_fn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        if batch_size and len(sub) > batch_size:
            for ep in range(epochs):
                perm = torch.randperm(len(sub))
                for start in range(0, len(sub), batch_size):
                    idx = perm[start:start+batch_size]
                    pred = model(x_sub[idx])
                    loss = nn.MSELoss()(pred, y_sub[idx])
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                scheduler.step()
        else:
            for ep in range(epochs):
                pred = model(x_sub)
                loss = nn.MSELoss()(pred, y_sub)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            if batch_size and len(te) > batch_size:
                preds_list = []
                for start in range(0, len(te), batch_size):
                    preds_list.append(model(x_te[start:start+batch_size]).cpu())
                all_preds[te] = torch.cat(preds_list, dim=0)
            else:
                all_preds[te] = model(x_te).cpu()

    preds_np = all_preds.numpy()
    pd = eval_pearson_delta(preds_np, Y_target)
    deg = eval_top_deg(preds_np, Y_target)
    elapsed = time.time() - t0
    avg_n = max(2, int(len(tr) * frac))
    report(name, avg_n, pd, deg, elapsed)
    return pd, deg


# ═══════════════════════════════════════════════════════════════
p(f"\n{'='*110}")
p(f"MODERNBERT + BIG MLP SWEEP")
p(f"{'='*110}")

results = []

for frac in [0.05, 0.10, 0.20, 0.50, 1.0]:
    pct = int(frac * 100)
    p(f"\n{'─'*110}")
    p(f"  TRAINING FRACTION: {pct}%")
    p(f"{'─'*110}")

    # ─── Ridge baseline ───
    p(f"\n  Ridge baselines:")
    for alpha in [0.5, 1.0, 5.0]:
        preds = np.zeros_like(Y_delta_hvg)
        for fold, (tr, te) in enumerate(kf.split(X_cg)):
            n_sub = max(2, int(len(tr) * frac))
            np.random.seed(fold + 42)
            sub = np.random.choice(tr, size=n_sub, replace=False)
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_cg[sub])
            X_te = sc.transform(X_cg[te])
            model = Ridge(alpha=alpha)
            model.fit(X_tr, Y_delta_hvg[sub])
            preds[te] = model.predict(X_te)
        pd = eval_pearson_delta(preds, Y_delta_hvg)
        deg = eval_top_deg(preds, Y_delta_hvg)
        report(f"Ridge(α={alpha})", max(2, int(len(tr)*frac)), pd, deg, 0)
        results.append((f'Ridge(α={alpha})', pct, pd, deg))

    # ─── Big MLPs ───
    p(f"\n  Big MLPs:")
    configs = [
        # (hidden_dims, dropout, lr, wd, epochs, activation, name)
        ([2048], 0.4, 5e-4, 1e-2, 600, 'gelu', 'MLP[2048]'),
        ([4096], 0.5, 3e-4, 1e-2, 600, 'gelu', 'MLP[4096]'),
        ([2048, 1024], 0.4, 5e-4, 1e-2, 600, 'gelu', 'MLP[2048,1024]'),
        ([4096, 2048], 0.5, 3e-4, 1e-2, 600, 'gelu', 'MLP[4096,2048]'),
        ([2048, 1024, 512], 0.5, 3e-4, 1e-2, 800, 'gelu', 'MLP[2048,1024,512]'),
        ([4096, 2048, 1024], 0.5, 2e-4, 1e-2, 800, 'gelu', 'MLP[4096,2048,1024]'),
        # SiLU variants
        ([2048, 1024], 0.4, 5e-4, 5e-3, 600, 'silu', 'MLP[2048,1024] SiLU'),
        ([4096, 2048], 0.5, 3e-4, 5e-3, 600, 'silu', 'MLP[4096,2048] SiLU'),
        # With cell-line conditioning
        ([2048, 1024], 0.4, 5e-4, 1e-2, 600, 'gelu', 'MLP+CL[2048,1024]'),
        ([4096, 2048], 0.5, 3e-4, 1e-2, 600, 'gelu', 'MLP+CL[4096,2048]'),
    ]

    for hid, dp, lr, wd, epochs, act, name in configs:
        use_cl = '+CL' in name
        X_use = X_cg_cl if use_cl else X_cg
        act_fn = {'gelu': nn.GELU, 'silu': nn.SiLU}[act]

        def make_mlp(hid=hid, dp=dp, act_fn=act_fn, in_dim=X_use.shape[1]):
            layers = []
            d = in_dim
            for h in hid:
                layers.extend([nn.Linear(d, h), act_fn(), nn.Dropout(dp)])
                d = h
            layers.append(nn.Linear(d, n_hvg))
            return nn.Sequential(*layers)

        pd, deg = train_model(X_use, Y_delta_hvg, frac, make_mlp, epochs, lr, wd,
                              f"{name} dp={dp}", batch_size=256)
        results.append((name, pct, pd, deg))

    # ─── ModernBERT variants ───
    p(f"\n  ModernBERT:")
    bert_configs = [
        # (hidden, layers, heads, intermediate, dropout, lr, epochs, name)
        (128, 2, 4, 512, 0.3, 3e-4, 500, 'ModernBERT-tiny(h=128,L=2)'),
        (256, 4, 4, 1024, 0.3, 3e-4, 500, 'ModernBERT-small(h=256,L=4)'),
        (384, 6, 6, 1536, 0.3, 2e-4, 600, 'ModernBERT-med(h=384,L=6)'),
        (512, 6, 8, 2048, 0.4, 2e-4, 600, 'ModernBERT-large(h=512,L=6)'),
        (768, 8, 12, 3072, 0.4, 1e-4, 600, 'ModernBERT-xl(h=768,L=8)'),
    ]

    for hidden, nlayers, nheads, inter, dp, lr, epochs, name in bert_configs:
        # Skip very large models at very low data
        n_train_est = max(2, int(n * 0.8 * frac))
        n_params_est = nlayers * (4 * hidden * hidden + hidden * inter * 2) + hidden * n_hvg * 2
        if n_train_est < 50 and n_params_est > 5_000_000:
            p(f"  {name:60s} | SKIP (too few samples for {n_params_est/1e6:.1f}M params)")
            continue

        def make_bert(h=hidden, nl=nlayers, nh=nheads, im=inter, d=dp):
            return ModernBERTRegressor(
                cg_dim=X_cg.shape[1], n_out=n_hvg,
                hidden_size=h, num_layers=nl, num_heads=nh,
                intermediate_size=im, chunk_size=63, dropout=d
            )

        try:
            pd, deg = train_model(X_cg, Y_delta_hvg, frac, make_bert, epochs, lr, 1e-2,
                                  f"{name} dp={dp}", batch_size=128)
            results.append((name, pct, pd, deg))
        except Exception as e:
            p(f"  {name:60s} | ERROR: {e}")

# ═══════════════════════════════════════════════════════════════
p(f"\n{'='*110}")
p(f"TOP RESULTS PER TRAINING FRACTION")
p(f"{'='*110}")

for pct in [5, 10, 20, 50, 100]:
    pct_results = [(n, p, d) for n, pc, p, d in results if pc == pct]
    pct_results.sort(key=lambda x: -x[1])
    p(f"\n  Top 5 at {pct}%:")
    for i, (name, pd, deg) in enumerate(pct_results[:5]):
        p(f"    {i+1}. Pearson Δ={pd:.4f} DEG-50={deg:.4f} | {name}")

p(f"\n{'='*110}")
p("THRESHOLD ANALYSIS")
p(f"{'='*110}")
for target in [0.80, 0.84, 0.85, 0.88, 0.90]:
    hits = [(pc, n, pd) for n, pc, pd, d in results if pd >= target]
    if hits:
        min_pct = min(h[0] for h in hits)
        best = max(h[2] for h in hits if h[0] == min_pct)
        method = [n for n, pc, pd, d in results if pc == min_pct and pd == best][0]
        p(f"  Pearson Δ ≥ {target:.2f}: {min_pct}% data | {best:.4f} | {method}")
    else:
        p(f"  Pearson Δ ≥ {target:.2f}: NOT achieved")

p("\nDone.")
