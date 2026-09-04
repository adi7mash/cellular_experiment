"""Fixed ModernBERT sweep — pad_token_id=0, eager attention to avoid flash_attn issues."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from transformers import ModernBertConfig, ModernBertModel
import time
import warnings
warnings.filterwarnings('ignore')

BASE = "/opt/dlami/nvme/tahoe"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def p(*a, **k):
    print(*a, **k, flush=True)

# Load data
pb = np.load(f"{BASE}/pseudobulk_means.npz", allow_pickle=True)
means = pb['means']
drug_names = pb['drug_names']
cell_line_ids = pb['cell_line_ids']
cell_counts = pb['cell_counts']
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
Y_delta = means[valid_idx] - np.array([ctrl_map[cell_line_ids[i]] for i in valid_idx])

n = len(valid_idx)
delta_var = np.var(Y_delta, axis=0)
n_hvg = min(2000, (delta_var > 1e-10).sum())
hvg_idx = np.argsort(-delta_var)[:n_hvg]
Y_delta_hvg = Y_delta[:, hvg_idx]

kf = KFold(n_splits=5, shuffle=True, random_state=42)
p(f"Data: {n} conditions, {n_hvg} HVGs, {X_cg.shape[1]} CG dims, device={device}")

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
        top_idx = np.argsort(-np.abs(true[i]))[:top_k]
        if np.std(pred[i, top_idx]) < 1e-8 or np.std(true[i, top_idx]) < 1e-8:
            continue
        r, _ = pearsonr(pred[i, top_idx], true[i, top_idx])
        if not np.isnan(r):
            corrs.append(r)
    return np.mean(corrs) if corrs else float('nan')


class ModernBERTRegressor(nn.Module):
    def __init__(self, cg_dim, n_out, hidden_size=256, num_layers=4, num_heads=4,
                 intermediate_size=1024, chunk_size=63, dropout=0.3):
        super().__init__()
        self.chunk_size = chunk_size
        n_chunks = (cg_dim + chunk_size - 1) // chunk_size
        self.pad_to = n_chunks * chunk_size
        self.n_chunks = n_chunks

        config = ModernBertConfig(
            vocab_size=50265,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=n_chunks + 2,
            hidden_dropout_prob=dropout,
            attention_dropout=dropout,
            classifier_dropout=dropout,
            pad_token_id=0,
            attn_implementation='eager',
        )
        self.bert = ModernBertModel(config)
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
        x = self.input_proj(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        out = self.bert(inputs_embeds=x).last_hidden_state
        return self.head(out[:, 0])


def train_eval(frac, model_fn, epochs, lr, wd, name, batch_size=128):
    X_in = torch.tensor(X_cg, dtype=torch.float32)
    Y_t = torch.tensor(Y_delta_hvg, dtype=torch.float32)
    all_preds = torch.zeros_like(Y_t)
    t0 = time.time()

    for fold, (tr, te) in enumerate(kf.split(X_cg)):
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
        for ep in range(epochs):
            if len(sub) > batch_size:
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
                pred = model(x_sub)
                loss = nn.MSELoss()(pred, y_sub)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, len(te), batch_size):
                preds.append(model(x_te[start:start+batch_size]).cpu())
            all_preds[te] = torch.cat(preds, dim=0)

    preds_np = all_preds.numpy()
    pd = eval_pearson_delta(preds_np, Y_delta_hvg)
    deg = eval_top_deg(preds_np, Y_delta_hvg)
    elapsed = time.time() - t0
    avg_n = max(2, int(len(tr) * frac))
    p(f"  {name:60s} | n={avg_n:4d} | Pearson Δ: {pd:.4f} | DEG-50: {deg:.4f} | {elapsed:.1f}s")
    return pd, deg


p(f"\n{'='*110}")
p(f"MODERNBERT SWEEP (fixed)")
p(f"{'='*110}")

bert_configs = [
    (128, 2, 4, 512, 0.3, 3e-4, 500, 'ModernBERT-tiny(h=128,L=2)'),
    (256, 4, 4, 1024, 0.3, 3e-4, 500, 'ModernBERT-small(h=256,L=4)'),
    (384, 6, 6, 1536, 0.3, 2e-4, 600, 'ModernBERT-med(h=384,L=6)'),
    (512, 6, 8, 2048, 0.4, 2e-4, 600, 'ModernBERT-large(h=512,L=6)'),
    (768, 8, 12, 3072, 0.4, 1e-4, 600, 'ModernBERT-xl(h=768,L=8)'),
    (1024, 12, 16, 4096, 0.4, 1e-4, 800, 'ModernBERT-xxl(h=1024,L=12)'),
]

results = []
for frac in [0.05, 0.10, 0.20, 0.50, 1.0]:
    pct = int(frac * 100)
    p(f"\n{'─'*110}")
    p(f"  TRAINING FRACTION: {pct}%")
    p(f"{'─'*110}")

    for hidden, nlayers, nheads, inter, dp, lr, epochs, name in bert_configs:
        n_train_est = max(2, int(n * 0.8 * frac))
        n_params = nlayers * (4 * hidden**2 + hidden * inter * 2) + hidden * n_hvg * 2
        if n_train_est < 30 and n_params > 10_000_000:
            p(f"  {name:60s} | SKIP ({n_params/1e6:.1f}M params, only {n_train_est} samples)")
            continue

        def make_model(h=hidden, nl=nlayers, nh=nheads, im=inter, d=dp):
            return ModernBERTRegressor(
                cg_dim=X_cg.shape[1], n_out=n_hvg,
                hidden_size=h, num_layers=nl, num_heads=nh,
                intermediate_size=im, chunk_size=63, dropout=d
            )

        try:
            pd, deg = train_eval(frac, make_model, epochs, lr, 1e-2, f"{name} dp={dp}")
            results.append((name, pct, pd, deg))
        except Exception as e:
            p(f"  {name:60s} | ERROR: {e}")

p(f"\n{'='*110}")
p("TOP RESULTS PER FRACTION")
p(f"{'='*110}")
for pct in [5, 10, 20, 50, 100]:
    pct_r = [(n, p, d) for n, pc, p, d in results if pc == pct]
    pct_r.sort(key=lambda x: -x[1])
    p(f"\n  Top 3 at {pct}%:")
    for i, (nm, pd, deg) in enumerate(pct_r[:3]):
        p(f"    {i+1}. Pearson Δ={pd:.4f} DEG-50={deg:.4f} | {nm}")

p("\nDone.")
