"""Beaini-matched evaluation pipeline.

Replicates Beaini's methodology as closely as possible:
1. PPI network propagation (STRING database)
2. Known compound selection (top 246 by Bonbon confidence)
3. Modified Jaccard similarity
4. 6-parameter model
5. Evaluation on CC AUROC and per-target AUROC
"""
import json
import os
import sys
import time
import numpy as np
import requests
from io import StringIO
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"
IP_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/interaction_prints"


def p(*a, **k):
    __builtins__["print"](*a, **k, flush=True) if isinstance(__builtins__, dict) else print(*a, **k, flush=True)


def download_string_ppi(gene_symbols, output_path):
    """Download STRING PPI network for our gene set."""
    if os.path.exists(output_path):
        p(f"  Loading cached PPI from {output_path}")
        return np.load(output_path, allow_pickle=True)

    p("  Downloading STRING PPI network...")
    # STRING API: get network for our genes
    # API has a limit, batch into groups of 200
    all_interactions = []
    batch_size = 200

    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i:i + batch_size]
        identifiers = "%0d".join(batch)
        url = f"https://string-db.org/api/tsv/network?identifiers={identifiers}&species=9606&required_score=700"
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code == 200:
                for line in resp.text.strip().split("\n")[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 6:
                        gene_a = parts[2]  # preferredName_A
                        gene_b = parts[3]  # preferredName_B
                        score = float(parts[5])  # score
                        all_interactions.append((gene_a, gene_b, score))
                p(f"    Batch {i//batch_size + 1}: {len(all_interactions)} total interactions")
        except Exception as e:
            p(f"    Batch {i//batch_size + 1} failed: {e}")
        time.sleep(0.5)

    p(f"  Total STRING interactions: {len(all_interactions)}")

    # Build adjacency matrix
    gene_to_idx = {g: i for i, g in enumerate(gene_symbols)}
    n = len(gene_symbols)
    ppi_matrix = np.zeros((n, n), dtype=np.float32)

    n_mapped = 0
    for gene_a, gene_b, score in all_interactions:
        if gene_a in gene_to_idx and gene_b in gene_to_idx:
            i, j = gene_to_idx[gene_a], gene_to_idx[gene_b]
            ppi_matrix[i, j] = score / 1000.0  # normalize to [0, 1]
            ppi_matrix[j, i] = score / 1000.0
            n_mapped += 1

    p(f"  Mapped interactions: {n_mapped}")
    p(f"  PPI density: {(ppi_matrix > 0).sum() / (n * n):.4%}")

    np.savez_compressed(output_path, ppi_matrix=ppi_matrix, gene_symbols=gene_symbols)
    return {"ppi_matrix": ppi_matrix, "gene_symbols": gene_symbols}


def propagate_ppi(score_matrix, ppi_matrix, alpha=0.3):
    """Propagate binding scores through PPI network.

    For each compound, its score for protein B gets a contribution from
    all proteins A that interact with B:
      new_score(compound, B) = score(compound, B) + alpha * sum_A(PPI(A,B) * score(compound, A))

    Args:
        score_matrix: (n_compounds, n_genes) binding scores
        ppi_matrix: (n_genes, n_genes) PPI interaction strengths [0,1]
        alpha: propagation strength

    Returns: propagated score matrix (n_compounds, n_genes)
    """
    # Normalize PPI by row (each protein's neighbors sum to 1)
    ppi_norm = ppi_matrix.copy()
    row_sums = ppi_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ppi_norm = ppi_norm / row_sums

    # One-step propagation
    propagated = score_matrix + alpha * (score_matrix @ ppi_norm)
    return propagated


def select_known_compounds(score_matrix, compound_ids, n_select=246):
    """Select 'known' compounds = those where Bonbon has strongest signal.

    Uses max binding score across all proteins as confidence proxy.
    """
    max_scores = np.max(score_matrix, axis=1)
    score_range = np.max(score_matrix, axis=1) - np.min(score_matrix, axis=1)

    # Rank by score range (how discriminative the compound's profile is)
    # High range = Bonbon clearly distinguishes which proteins this compound binds
    ranking = np.argsort(-score_range)
    selected = ranking[:n_select]

    sel_ids = [compound_ids[i] for i in selected]
    p(f"  Selected {n_select} 'known' compounds")
    p(f"  Score range: selected mean={score_range[selected].mean():.4f} "
      f"vs all mean={score_range.mean():.4f}")
    return selected, sel_ids


def modified_jaccard(binary_a, binary_b, noise=0.0):
    """Modified Jaccard with noise dampening.

    jaccard(a,b) = |a ∩ b| / (|a ∪ b| + noise)
    """
    intersection = np.sum(binary_a & binary_b, axis=-1).astype(np.float32)
    union = np.sum(binary_a | binary_b, axis=-1).astype(np.float32)
    return intersection / (union + noise + 1e-8)


def build_jaccard_cc_similarity(binary_prints, noise=0.0):
    """Build compound-compound Jaccard similarity matrix from binary prints (vectorized)."""
    B = binary_prints.astype(np.float32)
    # intersection = B @ B.T (counts shared 1s)
    intersection = B @ B.T
    # |A| + |B| for all pairs
    counts = B.sum(axis=1)
    sum_pairs = counts[:, None] + counts[None, :]
    # union = |A| + |B| - |A ∩ B|
    union = sum_pairs - intersection
    sim = intersection / (union + noise + 1e-8)
    return sim.astype(np.float32)


def virtual_cell_model(scores, ppi_matrix, threshold, ppi_alpha, noise, scale):
    """Beaini's 6-parameter virtual cell model (simplified).

    Parameters:
        threshold: binding score threshold for binary conversion
        ppi_alpha: PPI propagation strength
        noise: Jaccard noise dampening
        scale: output rescaling

    (We skip transcriptomics and activator/inhibitor since we don't have that data)
    """
    # Step 1: PPI propagation
    propagated = propagate_ppi(scores, ppi_matrix, alpha=ppi_alpha)

    # Step 2: Threshold to binary
    binary = (propagated > threshold).astype(np.int8)

    # Step 3: Modified Jaccard CC similarity
    n = binary.shape[0]
    cc_sim = build_jaccard_cc_similarity(binary, noise=noise)

    # Step 4: Rescale
    cc_sim = cc_sim * scale

    return cc_sim, binary, propagated


def optimize_model(scores, ppi_matrix, gt_cc_sim, pair_idx, compound_mask=None):
    """Optimize the 4 free parameters (we skip transcriptomics + activator)."""
    if compound_mask is not None:
        scores_sub = scores[compound_mask]
        gt_sub = gt_cc_sim[np.ix_(compound_mask, compound_mask)]
        sub_pair_idx = np.triu_indices(len(scores_sub), k=1)
    else:
        scores_sub = scores
        gt_sub = gt_cc_sim
        sub_pair_idx = pair_idx

    gt_flat = gt_sub[sub_pair_idx]

    def objective(params):
        threshold, ppi_alpha, noise, scale = params
        ppi_alpha = max(0, ppi_alpha)
        noise = max(0, noise)
        scale = max(0.01, scale)

        cc_pred, _, _ = virtual_cell_model(scores_sub, ppi_matrix, threshold, ppi_alpha, noise, scale)
        pred_flat = cc_pred[sub_pair_idx]

        # Negative correlation as loss
        if pred_flat.std() < 1e-8:
            return 1.0
        corr = np.corrcoef(pred_flat, gt_flat)[0, 1]
        return -corr

    # Grid search for initial point
    p("  Grid search for initial parameters...")
    best_loss = float("inf")
    best_init = [0.0, 0.3, 5.0, 1.0]

    for thresh in np.linspace(np.percentile(scores_sub, 50), np.percentile(scores_sub, 95), 5):
        for alpha in [0.0, 0.2, 0.5, 1.0]:
            for noise in [0, 5, 20, 50]:
                params = [thresh, alpha, noise, 1.0]
                loss = objective(params)
                if loss < best_loss:
                    best_loss = loss
                    best_init = params

    p(f"  Best grid point: loss={best_loss:.4f}, params={best_init}")

    # Refine with Nelder-Mead
    result = minimize(objective, best_init, method="Nelder-Mead",
                      options={"maxiter": 500, "xatol": 1e-4})
    best_params = result.x
    p(f"  Optimized: loss={result.fun:.4f}, params=[thresh={best_params[0]:.3f}, "
      f"ppi_alpha={best_params[1]:.3f}, noise={best_params[2]:.1f}, scale={best_params[3]:.3f}]")

    return best_params


def evaluate_cc_auroc(pred_sim, gt_binary, pair_idx, label=""):
    """Compute CC AUROC at all thresholds."""
    results = {}
    for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                        ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
        if tkey not in gt_binary:
            continue
        y = gt_binary[tkey][pair_idx]
        y_pred = pred_sim[pair_idx]
        if 0 < y.sum() < len(y):
            auroc = roc_auc_score(y, y_pred)
            results[tname] = float(auroc)
            p(f"  {label} @{tname}: AUROC={auroc:.4f}")
    return results


def main():
    p("=" * 70)
    p("BEAINI-MATCHED EVALUATION PIPELINE")
    p("=" * 70)

    # Load data
    gt = np.load(os.path.join(GT_DIR, "ground_truth_binary.npz"))
    gt_c_ids = gt["compound_ids"].tolist()
    gt_g_ids = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_c_ids), len(gt_g_ids)
    pair_idx = np.triu_indices(n_c, k=1)

    cc_sim_data = np.load(os.path.join(GT_DIR, "phenomics_compound_compound_sim.npz"))
    gt_cc_sim = cc_sim_data["similarity"]

    p(f"Compounds: {n_c}, Genes: {n_g}")

    # Load binding scores (projection raw — closest to Boltz-2 binary affinity)
    cg_data = np.load(os.path.join(IP_DIR, "cg_projection.npz"))
    score_raw = np.array(cg_data["score_raw"])  # materialize from NPZ
    pred_c = cg_data["compound_ids"].tolist()
    pred_g = cg_data["protein_ids"].tolist()
    c_map = {c: i for i, c in enumerate(pred_c)}
    g_map = {g: i for i, g in enumerate(pred_g)}

    # Vectorized alignment
    c_idx = np.array([c_map.get(c, -1) for c in gt_c_ids])
    g_idx = np.array([g_map.get(g, -1) for g in gt_g_ids])
    scores = np.zeros((n_c, n_g), dtype=np.float32)
    vc = c_idx >= 0
    vg = g_idx >= 0
    scores[np.ix_(np.where(vc)[0], np.where(vg)[0])] = score_raw[np.ix_(c_idx[vc], g_idx[vg])]

    p(f"Score matrix: {scores.shape}, range=[{scores.min():.4f}, {scores.max():.4f}]")

    # Step 1: Download PPI network
    p("\n--- Step 1: PPI Network ---")
    ppi_path = os.path.join(GT_DIR, "string_ppi.npz")
    ppi_data = download_string_ppi(gt_g_ids, ppi_path)
    ppi_matrix = ppi_data["ppi_matrix"]
    n_edges = (ppi_matrix > 0).sum() // 2
    p(f"  PPI edges: {n_edges}, density: {(ppi_matrix > 0).mean():.4%}")

    # Step 2: Select known compounds
    p("\n--- Step 2: Known Compound Selection ---")
    known_idx, known_ids = select_known_compounds(scores, gt_c_ids, n_select=246)

    # Step 3: Evaluate multiple approaches
    all_results = {}

    # === Approach A: Raw cosine (baseline, no PPI) ===
    p("\n--- Approach A: Raw Cosine (no PPI, no Jaccard) ---")
    from sklearn.metrics.pairwise import cosine_similarity
    cc_cosine = cosine_similarity(scores).astype(np.float32)
    p("  All 1674 compounds:")
    all_results["cosine_all"] = evaluate_cc_auroc(cc_cosine, gt, pair_idx, "cosine_all")
    # Known subset
    known_mask = np.array(known_idx)
    known_pair_idx = np.triu_indices(len(known_idx), k=1)
    cc_cosine_known = cc_cosine[np.ix_(known_idx, known_idx)]
    gt_known = {k: gt[k][np.ix_(known_idx, known_idx)] for k in gt if k.startswith("cc_binary")}
    gt_known_wrap = type("", (), {"__contains__": lambda s, k: k in gt_known, "__getitem__": lambda s, k: gt_known[k]})()
    p("  Known 246 compounds:")
    all_results["cosine_known"] = evaluate_cc_auroc(cc_cosine_known, gt_known_wrap, known_pair_idx, "cosine_known")

    # === Approach B: PPI propagation + cosine ===
    p("\n--- Approach B: PPI Propagation + Cosine ---")
    for alpha in [0.1, 0.3, 0.5, 1.0]:
        propagated = propagate_ppi(scores, ppi_matrix, alpha=alpha)
        cc_prop = cosine_similarity(propagated).astype(np.float32)
        p(f"  alpha={alpha}, All 1674:")
        all_results[f"ppi_{alpha}_cosine_all"] = evaluate_cc_auroc(cc_prop, gt, pair_idx, f"ppi_{alpha}")
        cc_prop_known = cc_prop[np.ix_(known_idx, known_idx)]
        p(f"  alpha={alpha}, Known 246:")
        all_results[f"ppi_{alpha}_cosine_known"] = evaluate_cc_auroc(cc_prop_known, gt_known_wrap, known_pair_idx, f"ppi_{alpha}_known")

    # === Approach C: PPI + Jaccard ===
    p("\n--- Approach C: PPI + Jaccard ---")
    best_alpha = max([k for k in all_results if "ppi" in k and "all" in k],
                     key=lambda k: all_results[k].get("0.6", 0))
    best_ppi_alpha = float(best_alpha.split("_")[1])
    p(f"  Using best PPI alpha: {best_ppi_alpha}")
    propagated = propagate_ppi(scores, ppi_matrix, alpha=best_ppi_alpha)

    for threshold_pct in [50, 70, 80, 90, 95]:
        thresh = np.percentile(propagated, threshold_pct)
        binary = (propagated > thresh).astype(np.int8)
        for noise in [0, 5, 20]:
            cc_jacc = build_jaccard_cc_similarity(binary, noise=noise)
            label = f"jacc_p{threshold_pct}_n{noise}"
            p(f"  {label}, All 1674:")
            all_results[f"{label}_all"] = evaluate_cc_auroc(cc_jacc, gt, pair_idx, label)
            cc_jacc_known = cc_jacc[np.ix_(known_idx, known_idx)]
            p(f"  {label}, Known 246:")
            all_results[f"{label}_known"] = evaluate_cc_auroc(cc_jacc_known, gt_known_wrap, known_pair_idx, f"{label}_known")

    # === Approach D: Optimized virtual cell model ===
    p("\n--- Approach D: Optimized Virtual Cell Model ---")
    p("  Optimizing on all 1674 compounds...")
    best_params_all = optimize_model(scores, ppi_matrix, gt_cc_sim, pair_idx)
    cc_pred_all, _, _ = virtual_cell_model(scores, ppi_matrix, *best_params_all)
    p("  Results on all:")
    all_results["vcell_all"] = evaluate_cc_auroc(cc_pred_all, gt, pair_idx, "vcell_all")

    p("  Optimizing on known 246 compounds...")
    gt_cc_sim_known = gt_cc_sim[np.ix_(known_idx, known_idx)]
    scores_known = scores[known_idx]
    best_params_known = optimize_model(scores_known, ppi_matrix, gt_cc_sim_known, known_pair_idx)
    cc_pred_known, _, _ = virtual_cell_model(scores_known, ppi_matrix, *best_params_known)
    p("  Results on known:")
    all_results["vcell_known"] = evaluate_cc_auroc(cc_pred_known, gt_known_wrap, known_pair_idx, "vcell_known")

    # === Approach E: Optimized model with XGBoost on top ===
    p("\n--- Approach E: XGBoost on propagated features ---")
    import xgboost as xgb
    # Build features: raw + PPI-propagated cosine + Jaccard at multiple thresholds
    propagated = propagate_ppi(scores, ppi_matrix, alpha=best_ppi_alpha)

    # CC features from propagated scores
    cc_cos_prop = cosine_similarity(propagated).astype(np.float32)

    # Jaccard at a few thresholds
    jaccard_features = {}
    for pct in [70, 85, 95]:
        thresh = np.percentile(propagated, pct)
        binary = (propagated > thresh).astype(np.int8)
        jaccard_features[f"jacc_p{pct}"] = build_jaccard_cc_similarity(binary, noise=5)

    # Stack features
    ecfp = np.load(os.path.join(GT_DIR, "ecfp_tanimoto.npz"))
    ecfp_ids = ecfp["compound_ids"].tolist()
    ecfp_map = {c: i for i, c in enumerate(ecfp_ids)}
    ecfp_tanimoto = np.array(ecfp["tanimoto"])
    ecfp_idx = np.array([ecfp_map.get(c, -1) for c in gt_c_ids])
    ecfp_valid = ecfp_idx >= 0
    ecfp_aligned = np.zeros((n_c, n_c), dtype=np.float32)
    ev = np.where(ecfp_valid)[0]
    ecfp_aligned[np.ix_(ev, ev)] = ecfp_tanimoto[np.ix_(ecfp_idx[ecfp_valid], ecfp_idx[ecfp_valid])]

    feature_matrices = {"ecfp": ecfp_aligned, "cos_prop": cc_cos_prop}
    feature_matrices.update(jaccard_features)

    # Also add non-propagated cosine
    cc_cos_raw = cosine_similarity(scores).astype(np.float32)
    feature_matrices["cos_raw"] = cc_cos_raw

    # For known subset
    for subset_name, subset_idx, subset_pair_idx, gt_wrap in [
        ("all", np.arange(n_c), pair_idx, gt),
        ("known", np.array(known_idx), known_pair_idx, gt_known_wrap),
    ]:
        n_sub = len(subset_idx)
        sub_pairs = np.triu_indices(n_sub, k=1)
        feat_names = list(feature_matrices.keys())
        X = np.column_stack([feature_matrices[k][np.ix_(subset_idx, subset_idx)][sub_pairs]
                             for k in feat_names])
        p(f"\n  XGBoost on {subset_name}: {X.shape[0]:,} pairs, {X.shape[1]} features")

        for tname, tkey in [("0.4", "cc_binary_04"), ("0.6", "cc_binary_06"),
                            ("P90", "cc_binary_p90"), ("P95", "cc_binary_p95")]:
            if tkey not in gt:
                continue
            if subset_name == "known":
                y = gt[tkey][np.ix_(known_idx, known_idx)][sub_pairs].astype(np.int32)
            else:
                y = gt[tkey][sub_pairs].astype(np.int32)

            if y.sum() == 0 or y.sum() == len(y):
                continue

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            fold_aurocs = []
            for tr, te in kf.split(X):
                model = xgb.XGBClassifier(
                    objective="binary:logistic", tree_method="hist", device="cuda",
                    n_estimators=500, max_depth=5, learning_rate=0.03,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                    verbosity=0)
                model.fit(X[tr], y[tr])
                fold_aurocs.append(roc_auc_score(y[te], model.predict_proba(X[te])[:, 1]))

            mean_a = np.mean(fold_aurocs)
            std_a = np.std(fold_aurocs)
            all_results[f"xgb_{subset_name}_{tname}"] = {"mean": float(mean_a), "std": float(std_a)}
            p(f"    @{tname}: AUROC={mean_a:.4f} ± {std_a:.4f}")

    # Per-target evaluation
    p("\n--- Per-Target AUROC ---")
    cg_binary = gt["cg_binary_top1"]
    from sklearn.model_selection import StratifiedKFold

    for subset_name, subset_idx in [("all", np.arange(n_c)), ("known", np.array(known_idx))]:
        propagated_sub = propagated[subset_idx]
        cg_binary_sub = cg_binary[subset_idx]
        n_sub = len(subset_idx)

        target_aurocs = []
        for j in range(n_g):
            y = cg_binary_sub[:, j].astype(np.int32)
            if y.sum() < 2 or y.sum() == n_sub:
                continue
            y_pred = propagated_sub[:, j]
            try:
                target_aurocs.append(roc_auc_score(y, y_pred))
            except ValueError:
                pass

        aurocs = np.array(target_aurocs)
        all_results[f"per_target_prop_{subset_name}"] = {
            "median": float(np.median(aurocs)),
            "mean": float(np.mean(aurocs)),
            "p25": float(np.percentile(aurocs, 25)),
            "p75": float(np.percentile(aurocs, 75)),
            "n_targets": len(aurocs),
        }
        p(f"  {subset_name}: median={np.median(aurocs):.4f}  mean={np.mean(aurocs):.4f}  "
          f"({len(aurocs)} targets)")

    # Summary
    p(f"\n{'='*70}")
    p("FINAL SUMMARY")
    p(f"{'='*70}")

    p("\nCC AUROC comparison:")
    p(f"{'Method':<40} {'@0.4':>8} {'@0.6':>8} {'@P95':>8}")
    p("-" * 70)
    for key in sorted(all_results):
        if isinstance(all_results[key], dict) and "0.4" in all_results[key]:
            v04 = all_results[key].get("0.4", "N/A")
            v06 = all_results[key].get("0.6", "N/A")
            v95 = all_results[key].get("P95", "N/A")
            if isinstance(v04, float):
                p(f"  {key:<40} {v04:>7.4f} {v06 if isinstance(v06, str) else v06:>7.4f} "
                  f"{v95 if isinstance(v95, str) else v95:>7.4f}")
        elif isinstance(all_results[key], dict) and "mean" in all_results[key]:
            mean = all_results[key]["mean"]
            std = all_results[key].get("std", 0)
            p(f"  {key:<40} {mean:.4f} ± {std:.4f}")

    p(f"\n  Beaini: 71% (@0.4), 76% (@0.6)  [PPI + transcriptomics, 246 known, 11 cell lines]")

    p("\nPer-target AUROC:")
    for key in all_results:
        if "per_target" in key:
            r = all_results[key]
            p(f"  {key:<40} median={r['median']:.4f}")
    p(f"  Beaini: 53.9% median (7K proteins)")

    # Save
    eval_dir = os.path.join(os.path.dirname(IP_DIR), "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "beaini_matched_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
