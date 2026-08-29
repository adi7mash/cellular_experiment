"""Build phenomics ground truth using EFAAR TVN preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"
GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"

print("Loading metadata...")
meta = pd.read_csv(f"{DATA_DIR}/metadata_rxrx3_core.csv", low_memory=False)

print("Loading OpenPhenom embeddings...")
emb_df = pd.read_parquet(f"{DATA_DIR}/OpenPhenom_rxrx3_core_embeddings.parquet")
print(f"Embeddings shape: {emb_df.shape}")

emb_cols = [c for c in emb_df.columns if c != "well_id"]
n_dims = len(emb_cols)
print(f"Embedding dimensions: {n_dims}")

merged = meta.merge(emb_df, on="well_id", how="inner")
print(f"Merged: {len(merged)} wells")

is_control = (
    merged["gene"].isin(["EMPTY_control"]) |
    merged["treatment"].isin(["DMSO", "EMPTY_control"])
)

embeddings = merged[emb_cols].values.copy().astype(np.float64)

print("\n--- Step 1: TVN (Typical Variation Normalization) ---")

# Step 1a: Center and scale on controls (per experiment batch)
print("  1a: Center+scale on controls per experiment...")
for exp_name in merged["experiment_name"].unique():
    exp_mask = (merged["experiment_name"] == exp_name).values
    ctrl_mask = (exp_mask & is_control.values)
    if ctrl_mask.sum() < 2:
        scaler = StandardScaler().fit(embeddings[exp_mask])
    else:
        scaler = StandardScaler().fit(embeddings[ctrl_mask])
    embeddings[exp_mask] = scaler.transform(embeddings[exp_mask])

# Step 1b: PCA on controls (rotation to decorrelate)
print("  1b: PCA on controls (whitening)...")
ctrl_emb = embeddings[is_control.values]
pca = PCA().fit(ctrl_emb)
embeddings = pca.transform(embeddings)

# Step 1c: Center and scale on controls again (in PCA space)
print("  1c: Center+scale on controls in PCA space...")
ctrl_mask_all = is_control.values
ctrl_emb_pca = embeddings[ctrl_mask_all]
scaler_pca = StandardScaler().fit(ctrl_emb_pca)
embeddings = scaler_pca.transform(embeddings)

# Step 1d: CORAL per experiment batch
print("  1d: CORAL normalization per batch...")
for exp_name in merged["experiment_name"].unique():
    exp_mask = (merged["experiment_name"] == exp_name).values
    ctrl_mask = exp_mask & is_control.values
    if ctrl_mask.sum() < n_dims + 1:
        continue
    ctrl_batch = embeddings[ctrl_mask]
    cov = np.cov(ctrl_batch, rowvar=False, ddof=1) + 0.5 * np.eye(embeddings.shape[1])
    cov_half_inv = linalg.fractional_matrix_power(cov, -0.5).real
    embeddings[exp_mask] = embeddings[exp_mask] @ cov_half_inv

embeddings = embeddings.astype(np.float32)
print("TVN done.")

# Check similarity distribution after TVN
print("\n--- Verifying preprocessing ---")
ctrl_sims = cosine_similarity(embeddings[is_control.values][:100])
upper_ctrl = ctrl_sims[np.triu_indices(100, k=1)]
print(f"Control cosine sim: mean={upper_ctrl.mean():.4f}, std={upper_ctrl.std():.4f}")

print("\n--- Step 2: Aggregate gene KO embeddings ---")
gene_mask = (
    (merged["perturbation_type"] == "CRISPR") &
    (~merged["gene"].isin(["EMPTY_control", "CRISPR_control", ""])) &
    merged["gene"].notna()
).values

gene_wells = merged.loc[gene_mask, ["gene"]].copy()
gene_wells["_idx"] = np.where(gene_mask)[0]

gene_agg = {}
for gene, group in gene_wells.groupby("gene"):
    gene_agg[gene] = embeddings[group["_idx"].values].mean(axis=0)

gene_ids = sorted(gene_agg.keys())
gene_embeddings = np.array([gene_agg[g] for g in gene_ids], dtype=np.float32)
print(f"Gene embeddings: {gene_embeddings.shape}")

np.savez_compressed(f"{GT_DIR}/phenomics_gene_embeddings.npz",
    embeddings=gene_embeddings, gene_ids=np.array(gene_ids))

print("\n--- Step 3: Aggregate compound embeddings ---")
compound_mask = (
    (merged["perturbation_type"] == "COMPOUND") &
    (~merged["treatment"].isin(["EMPTY_control", "DMSO", "CRISPR_control", ""])) &
    merged["treatment"].notna()
).values

compound_wells = merged.loc[compound_mask, ["treatment"]].copy()
compound_wells["_idx"] = np.where(compound_mask)[0]

compound_agg = {}
for treatment, group in compound_wells.groupby("treatment"):
    compound_agg[treatment] = embeddings[group["_idx"].values].mean(axis=0)

compound_ids = sorted(compound_agg.keys())
compound_embeddings = np.array([compound_agg[c] for c in compound_ids], dtype=np.float32)
print(f"Compound embeddings: {compound_embeddings.shape}")

np.savez_compressed(f"{GT_DIR}/phenomics_compound_embeddings.npz",
    embeddings=compound_embeddings, compound_ids=np.array(compound_ids))

print("\n--- Step 4: Similarity matrices ---")
cc_sim = cosine_similarity(compound_embeddings).astype(np.float32)
upper = cc_sim[np.triu_indices(len(compound_ids), k=1)]
print(f"CC sim: mean={upper.mean():.4f}, std={upper.std():.4f}, "
      f"min={upper.min():.4f}, max={upper.max():.4f}")

np.savez_compressed(f"{GT_DIR}/phenomics_compound_compound_sim.npz",
    similarity=cc_sim, compound_ids=np.array(compound_ids))

cg_sim = cosine_similarity(compound_embeddings, gene_embeddings).astype(np.float32)
np.savez_compressed(f"{GT_DIR}/phenomics_compound_gene_sim.npz",
    similarity=cg_sim, compound_ids=np.array(compound_ids), gene_ids=np.array(gene_ids))
print(f"CG sim: {cg_sim.shape}")

print("\n--- Step 5: Binarize ---")
# Use percentile-based thresholds that match Beaini's methodology
# Also include absolute thresholds for comparison
for thresh_name, thresh_val in [("04", 0.4), ("06", 0.6)]:
    binary = (cc_sim > thresh_val).astype(np.int8)
    np.fill_diagonal(binary, 0)
    rate = binary[np.triu_indices(len(compound_ids), k=1)].mean()
    print(f"  CC >{thresh_val}: {rate:.4%} positive")

# Percentile thresholds
p90_thresh = np.percentile(upper, 90)
p95_thresh = np.percentile(upper, 95)
cc_binary_04 = (cc_sim > 0.4).astype(np.int8)
cc_binary_06 = (cc_sim > 0.6).astype(np.int8)
cc_binary_p90 = (cc_sim > p90_thresh).astype(np.int8)
cc_binary_p95 = (cc_sim > p95_thresh).astype(np.int8)
for mat in [cc_binary_04, cc_binary_06, cc_binary_p90, cc_binary_p95]:
    np.fill_diagonal(mat, 0)

print(f"  CC >P90 ({p90_thresh:.4f}): {cc_binary_p90[np.triu_indices(len(compound_ids), k=1)].mean():.4%}")
print(f"  CC >P95 ({p95_thresh:.4f}): {cc_binary_p95[np.triu_indices(len(compound_ids), k=1)].mean():.4%}")

n_compounds = len(compound_ids)
n_genes = len(gene_ids)
cg_binary_top1 = np.zeros_like(cg_sim, dtype=np.int8)
top_k = max(1, int(n_compounds * 0.01))
for j in range(n_genes):
    col = cg_sim[:, j]
    threshold = np.sort(col)[-top_k]
    cg_binary_top1[:, j] = (col >= threshold).astype(np.int8)

np.savez_compressed(f"{GT_DIR}/ground_truth_binary.npz",
    cc_binary_04=cc_binary_04, cc_binary_06=cc_binary_06,
    cc_binary_p90=cc_binary_p90, cc_binary_p95=cc_binary_p95,
    cg_binary_top1=cg_binary_top1,
    compound_ids=np.array(compound_ids), gene_ids=np.array(gene_ids),
    p90_threshold=np.float32(p90_thresh), p95_threshold=np.float32(p95_thresh))

print(f"\nCG top 1%: {cg_binary_top1.mean():.4%}")

# Quick ECFP sanity check
ecfp = np.load(f"{GT_DIR}/ecfp_tanimoto.npz")
ecfp_ids = ecfp["compound_ids"].tolist()
id_map = {c: i for i, c in enumerate(ecfp_ids)}
idx = [id_map[c] for c in compound_ids if c in id_map]
ecfp_sub = ecfp["tanimoto"][np.ix_(idx, idx)]
ecfp_upper = ecfp_sub[np.triu_indices(len(idx), k=1)]
from scipy.stats import spearmanr
rho, _ = spearmanr(ecfp_upper, upper[:len(ecfp_upper)])
print(f"\nSpearman(ECFP, phenomics): {rho:.4f}")

from sklearn.metrics import roc_auc_score
for name, binary in [("P90", cc_binary_p90), ("P95", cc_binary_p95)]:
    gt_upper = binary[np.triu_indices(len(idx), k=1)]
    auroc = roc_auc_score(gt_upper, ecfp_upper[:len(gt_upper)])
    print(f"ECFP AUROC ({name}): {auroc:.4f}")

print("\nDone!")
