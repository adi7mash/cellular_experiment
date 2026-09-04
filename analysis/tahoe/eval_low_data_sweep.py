"""
Sweep architectures/hyperparams to maximize Pearson Δ at 1-20% training data.
Goal: 0.84-0.92 range with minimal Tahoe training data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
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
cg_proteins = list(cg['proteins'])

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
unique_cls = sorted(set(valid_cls))
cl_to_idx = {cl: i for i, cl in enumerate(unique_cls)}
cl_onehot = np.zeros((n, len(unique_cls)))
for i, cl in enumerate(valid_cls):
    cl_onehot[i, cl_to_idx[cl]] = 1.0
X_cg_cl = np.hstack([X_cg, cl_onehot])

p(f"Data: {n} conditions, {n_hvg} HVGs, {X_cg.shape[1]} CG dims, {len(unique_cls)} cell lines")
p(f"Device: {device}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

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

def run_ridge(X, Y, frac, alpha, scaler_cls=StandardScaler, name=""):
    preds = np.zeros_like(Y)
    n_trains = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        n_sub = max(2, int(len(tr) * frac))
        np.random.seed(fold + 42)
        sub = np.random.choice(tr, size=n_sub, replace=False)
        n_trains.append(n_sub)
        sc = scaler_cls()
        X_tr = sc.fit_transform(X[sub])
        X_te = sc.transform(X[te])
        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y[sub])
        preds[te] = model.predict(X_te)
    pd = eval_pearson_delta(preds, Y)
    deg = eval_top_deg(preds, Y)
    avg_n = int(np.mean(n_trains))
    p(f"  {name:55s} | n={avg_n:4d} | Pearson Δ: {pd:.4f} | DEG-50: {deg:.4f}")
    return pd, deg

# ─── MLP with various configs ───
def train_mlp_lowdata(X_feat, Y_target, frac, hidden_dims, epochs, lr, dropout, wd,
                      activation='gelu', batch_norm=False, name="MLP"):
    X_in = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(Y_target, dtype=torch.float32)
    all_preds = torch.zeros_like(Y_t)
    n_out = Y_target.shape[1]

    for fold, (tr, te) in enumerate(kf.split(X_feat)):
        n_sub = max(2, int(len(tr) * frac))
        np.random.seed(fold + 42)
        sub = np.random.choice(tr, size=n_sub, replace=False)

        x_sub, y_sub = X_in[sub].to(device), Y_t[sub].to(device)
        x_te = X_in[te].to(device)

        mu, std = x_sub.mean(0), x_sub.std(0).clamp(min=1e-8)
        x_sub = (x_sub - mu) / std
        x_te = (x_te - mu) / std

        act_fn = {'gelu': nn.GELU, 'silu': nn.SiLU, 'relu': nn.ReLU, 'mish': nn.Mish}[activation]

        layers = []
        in_dim = x_sub.shape[1]
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_out))
        model = nn.Sequential(*layers).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for ep in range(epochs):
            pred = model(x_sub)
            loss = nn.MSELoss()(pred, y_sub)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            all_preds[te] = model(x_te).cpu()

    preds_np = all_preds.numpy()
    pd = eval_pearson_delta(preds_np, Y_target)
    deg = eval_top_deg(preds_np, Y_target)
    avg_n = max(2, int(len(tr) * frac))
    p(f"  {name:55s} | n={avg_n:4d} | Pearson Δ: {pd:.4f} | DEG-50: {deg:.4f}")
    return pd, deg

# ─── Transformer for expression prediction ───
class CGTransformer(nn.Module):
    def __init__(self, cg_dim, n_out, d_model=256, nhead=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(cg_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_out)
        )

    def forward(self, x):
        # x: [B, cg_dim] -> reshape to [B, num_chunks, chunk_size]
        x = self.input_proj(x).unsqueeze(1)  # [B, 1, d_model]
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 2, d_model]
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token output

class CGChunkedTransformer(nn.Module):
    """Split 945-dim CG into chunks, treat as sequence tokens."""
    def __init__(self, cg_dim, n_out, chunk_size=63, d_model=128, nhead=4, n_layers=2, dropout=0.3):
        super().__init__()
        self.chunk_size = chunk_size
        n_chunks = (cg_dim + chunk_size - 1) // chunk_size
        self.pad_to = n_chunks * chunk_size
        self.input_proj = nn.Linear(chunk_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, n_chunks + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, n_out)
        )

    def forward(self, x):
        B = x.size(0)
        if x.size(1) < self.pad_to:
            x = F.pad(x, (0, self.pad_to - x.size(1)))
        x = x.view(B, -1, self.chunk_size)  # [B, n_chunks, chunk_size]
        x = self.input_proj(x)  # [B, n_chunks, d_model]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, n_chunks+1, d_model]
        x = x + self.pos_emb[:, :x.size(1)]
        x = self.encoder(x)
        return self.head(x[:, 0])

def train_transformer(X_feat, Y_target, frac, model_cls, model_kwargs, epochs, lr, wd, name="Transformer"):
    X_in = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(Y_target, dtype=torch.float32)
    all_preds = torch.zeros_like(Y_t)

    for fold, (tr, te) in enumerate(kf.split(X_feat)):
        n_sub = max(2, int(len(tr) * frac))
        np.random.seed(fold + 42)
        sub = np.random.choice(tr, size=n_sub, replace=False)

        x_sub, y_sub = X_in[sub].to(device), Y_t[sub].to(device)
        x_te = X_in[te].to(device)

        mu, std = x_sub.mean(0), x_sub.std(0).clamp(min=1e-8)
        x_sub = (x_sub - mu) / std
        x_te = (x_te - mu) / std

        model = model_cls(cg_dim=x_sub.shape[1], n_out=Y_target.shape[1], **model_kwargs).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
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
            all_preds[te] = model(x_te).cpu()

    preds_np = all_preds.numpy()
    pd = eval_pearson_delta(preds_np, Y_target)
    deg = eval_top_deg(preds_np, Y_target)
    avg_n = max(2, int(len(tr) * frac))
    p(f"  {name:55s} | n={avg_n:4d} | Pearson Δ: {pd:.4f} | DEG-50: {deg:.4f}")
    return pd, deg

# ─── Bottleneck MLP (encode CG → low dim → decode to expression) ───
class BottleneckMLP(nn.Module):
    def __init__(self, in_dim, n_out, bottleneck=64, hidden=512, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, bottleneck),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_bottleneck(X_feat, Y_target, frac, bottleneck, hidden, epochs, lr, dropout, wd, name="Bottleneck"):
    X_in = torch.tensor(X_feat, dtype=torch.float32)
    Y_t = torch.tensor(Y_target, dtype=torch.float32)
    all_preds = torch.zeros_like(Y_t)

    for fold, (tr, te) in enumerate(kf.split(X_feat)):
        n_sub = max(2, int(len(tr) * frac))
        np.random.seed(fold + 42)
        sub = np.random.choice(tr, size=n_sub, replace=False)

        x_sub, y_sub = X_in[sub].to(device), Y_t[sub].to(device)
        x_te = X_in[te].to(device)

        mu, std = x_sub.mean(0), x_sub.std(0).clamp(min=1e-8)
        x_sub = (x_sub - mu) / std
        x_te = (x_te - mu) / std

        model = BottleneckMLP(x_sub.shape[1], Y_target.shape[1], bottleneck, hidden, dropout).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for ep in range(epochs):
            pred = model(x_sub)
            loss = nn.MSELoss()(pred, y_sub)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            all_preds[te] = model(x_te).cpu()

    preds_np = all_preds.numpy()
    pd = eval_pearson_delta(preds_np, Y_target)
    deg = eval_top_deg(preds_np, Y_target)
    avg_n = max(2, int(len(tr) * frac))
    p(f"  {name:55s} | n={avg_n:4d} | Pearson Δ: {pd:.4f} | DEG-50: {deg:.4f}")
    return pd, deg

# ═══════════════════════════════════════════════════════════════════
p(f"\n{'='*100}")
p(f"LOW-DATA SWEEP: Target 0.84-0.92 Pearson Δ with 1-20% training data")
p(f"{'='*100}")

results = []

for frac in [0.01, 0.05, 0.10, 0.15, 0.20]:
    pct = int(frac * 100)
    p(f"\n{'─'*100}")
    p(f"  TRAINING FRACTION: {pct}%")
    p(f"{'─'*100}")

    # ─── Ridge sweep ───
    p(f"\n  Ridge variants:")
    for alpha in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]:
        pd, deg = run_ridge(X_cg, Y_delta_hvg, frac, alpha, StandardScaler,
                           f"Ridge(α={alpha}) StdScaler")
        results.append(('Ridge', f'α={alpha}', pct, pd, deg))

    # Ridge with RobustScaler
    for alpha in [0.1, 1.0, 10.0]:
        pd, deg = run_ridge(X_cg, Y_delta_hvg, frac, alpha, RobustScaler,
                           f"Ridge(α={alpha}) RobustScaler")
        results.append(('Ridge-Robust', f'α={alpha}', pct, pd, deg))

    # Ridge + cell line
    for alpha in [0.1, 1.0, 10.0]:
        pd, deg = run_ridge(X_cg_cl, Y_delta_hvg, frac, alpha, StandardScaler,
                           f"Ridge+CL(α={alpha})")
        results.append(('Ridge+CL', f'α={alpha}', pct, pd, deg))

    # ─── MLP sweep ───
    p(f"\n  MLP variants:")

    # Small MLPs for low data
    for hid, dp, lr, wd, act in [
        ([256], 0.5, 1e-3, 1e-2, 'gelu'),
        ([512], 0.5, 1e-3, 1e-2, 'gelu'),
        ([512], 0.3, 5e-4, 5e-3, 'silu'),
        ([1024], 0.5, 5e-4, 1e-2, 'gelu'),
        ([512, 256], 0.5, 5e-4, 1e-2, 'gelu'),
        ([512, 256], 0.4, 1e-3, 5e-3, 'mish'),
        ([1024, 512], 0.5, 5e-4, 1e-2, 'gelu'),
        ([1024, 512, 256], 0.5, 3e-4, 1e-2, 'gelu'),
    ]:
        hstr = ','.join(map(str, hid))
        name = f"MLP[{hstr}] dp={dp} lr={lr} wd={wd} {act}"
        pd, deg = train_mlp_lowdata(X_cg, Y_delta_hvg, frac, hid, 500, lr, dp, wd, act, name=name)
        results.append(('MLP', name, pct, pd, deg))

    # MLP with batch norm
    for hid, dp in [([512], 0.3), ([1024, 512], 0.4)]:
        hstr = ','.join(map(str, hid))
        name = f"MLP[{hstr}]+BN dp={dp}"
        pd, deg = train_mlp_lowdata(X_cg, Y_delta_hvg, frac, hid, 500, 5e-4, dp, 1e-2,
                                    'gelu', batch_norm=True, name=name)
        results.append(('MLP+BN', name, pct, pd, deg))

    # MLP + cell line features
    for hid, dp in [([512], 0.4), ([1024, 512], 0.5)]:
        hstr = ','.join(map(str, hid))
        name = f"MLP+CL[{hstr}] dp={dp}"
        pd, deg = train_mlp_lowdata(X_cg_cl, Y_delta_hvg, frac, hid, 500, 5e-4, dp, 1e-2,
                                    'gelu', name=name)
        results.append(('MLP+CL', name, pct, pd, deg))

    # ─── Bottleneck MLP ───
    p(f"\n  Bottleneck MLP:")
    for bn, hid, dp in [(32, 512, 0.3), (64, 512, 0.3), (128, 1024, 0.4)]:
        name = f"Bottleneck(bn={bn},h={hid}) dp={dp}"
        pd, deg = train_bottleneck(X_cg, Y_delta_hvg, frac, bn, hid, 500, 5e-4, dp, 1e-2, name=name)
        results.append(('Bottleneck', name, pct, pd, deg))

    # ─── Transformer ───
    p(f"\n  Transformer:")
    # Only run transformers for >= 5% data (too few samples otherwise)
    if frac >= 0.05:
        for d_model, nhead, nlayers, dp in [(128, 4, 2, 0.3), (256, 4, 2, 0.4)]:
            name = f"ChunkedTF(d={d_model},h={nhead},L={nlayers}) dp={dp}"
            pd, deg = train_transformer(
                X_cg, Y_delta_hvg, frac,
                CGChunkedTransformer,
                dict(chunk_size=63, d_model=d_model, nhead=nhead, n_layers=nlayers, dropout=dp),
                500, 3e-4, 1e-2, name=name
            )
            results.append(('Transformer', name, pct, pd, deg))
    else:
        p(f"    (skipped — too few samples for transformer at {pct}%)")

# ─── Summary ───
p(f"\n{'='*100}")
p(f"TOP RESULTS PER TRAINING FRACTION")
p(f"{'='*100}")

for pct in [1, 5, 10, 15, 20]:
    pct_results = [(t, n, p, d) for t, n, pc, p, d in results if pc == pct]
    pct_results.sort(key=lambda x: -x[2])
    p(f"\n  Top 5 at {pct}%:")
    for i, (typ, name, pd, deg) in enumerate(pct_results[:5]):
        p(f"    {i+1}. Pearson Δ={pd:.4f} DEG-50={deg:.4f} | {name}")

# Also show: what's the minimum data to reach 0.84?
p(f"\n{'='*100}")
p(f"THRESHOLD ANALYSIS: Minimum data to reach targets")
p(f"{'='*100}")
for target in [0.70, 0.75, 0.80, 0.84, 0.85, 0.90]:
    hits = [(pc, n, pd) for t, n, pc, pd, d in results if pd >= target]
    if hits:
        min_pct = min(h[0] for h in hits)
        best_at_min = max(h[2] for h in hits if h[0] == min_pct)
        method = [n for t, n, pc, pd, d in results if pc == min_pct and pd == best_at_min][0]
        p(f"  Pearson Δ ≥ {target:.2f}: achievable at {min_pct}% | best={best_at_min:.4f} | {method}")
    else:
        p(f"  Pearson Δ ≥ {target:.2f}: NOT achieved in sweep")

p("\nDone.")
