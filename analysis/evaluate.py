"""Evaluate all methods against phenomics ground truth (Phase 5)."""
import json
import os
import sys
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate_cc(pred_sim: np.ndarray, gt_binary: np.ndarray, method_name: str) -> dict:
    """Evaluate compound-compound similarity prediction."""
    n = pred_sim.shape[0]
    mask = np.triu_indices(n, k=1)
    y_true = gt_binary[mask]
    y_score = pred_sim[mask]

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"method": method_name, "auroc": float("nan"), "n_positive": int(y_true.sum()), "n_total": len(y_true)}

    auroc = roc_auc_score(y_true, y_score)
    return {
        "method": method_name,
        "auroc": auroc,
        "n_positive": int(y_true.sum()),
        "n_total": len(y_true),
        "positive_rate": float(y_true.mean()),
    }


def evaluate_per_target(pred_cg: np.ndarray, gt_binary_cg: np.ndarray, method_name: str) -> dict:
    """Evaluate per-target AUROC (compound-gene)."""
    n_compounds, n_genes = pred_cg.shape
    aurocs = []

    for j in range(n_genes):
        y_true = gt_binary_cg[:, j]
        y_score = pred_cg[:, j]

        if y_true.sum() == 0 or y_true.sum() == n_compounds:
            continue

        try:
            aurocs.append(roc_auc_score(y_true, y_score))
        except ValueError:
            continue

    aurocs = np.array(aurocs)
    return {
        "method": method_name,
        "n_targets_evaluated": len(aurocs),
        "median_auroc": float(np.median(aurocs)),
        "mean_auroc": float(np.mean(aurocs)),
        "p25_auroc": float(np.percentile(aurocs, 25)),
        "p75_auroc": float(np.percentile(aurocs, 75)),
        "std_auroc": float(np.std(aurocs)),
        "per_target_aurocs": aurocs.tolist(),
    }


def align_matrices(pred_ids: list, gt_ids: list, pred_matrix: np.ndarray, axis: int) -> np.ndarray:
    """Reorder pred_matrix rows/cols to match gt_ids ordering."""
    pred_id_to_idx = {pid: i for i, pid in enumerate(pred_ids)}
    indices = [pred_id_to_idx[gid] for gid in gt_ids if gid in pred_id_to_idx]

    if axis == 0:
        return pred_matrix[indices]
    else:
        return pred_matrix[:, indices]


def main(results_dir: str, gt_dir: str):
    os.makedirs(os.path.join(results_dir, "evaluation"), exist_ok=True)

    gt_binary = np.load(os.path.join(gt_dir, "ground_truth_binary.npz"))
    cc_binary_04 = gt_binary["cc_binary_04"]
    cc_binary_06 = gt_binary["cc_binary_06"]
    cc_binary_p90 = gt_binary["cc_binary_p90"] if "cc_binary_p90" in gt_binary else None
    cc_binary_p95 = gt_binary["cc_binary_p95"] if "cc_binary_p95" in gt_binary else None
    cg_binary = gt_binary["cg_binary_top1"]
    gt_compound_ids = gt_binary["compound_ids"].tolist()
    gt_gene_ids = gt_binary["gene_ids"].tolist()

    ecfp_data = np.load(os.path.join(gt_dir, "ecfp_tanimoto.npz"))
    ecfp_tanimoto = ecfp_data["tanimoto"]
    ecfp_ids = ecfp_data["compound_ids"].tolist()

    ip_dir = os.path.join(results_dir, "interaction_prints")

    thresholds = {"threshold_04": cc_binary_04, "threshold_06": cc_binary_06}
    if cc_binary_p90 is not None:
        thresholds["threshold_p90"] = cc_binary_p90
    if cc_binary_p95 is not None:
        thresholds["threshold_p95"] = cc_binary_p95
    cc_results = {k: [] for k in thresholds}
    cg_results = []

    # --- ECFP baseline ---
    ecfp_id_map = {cid: i for i, cid in enumerate(ecfp_ids)}
    ecfp_indices = [ecfp_id_map[cid] for cid in gt_compound_ids if cid in ecfp_id_map]
    ecfp_aligned = ecfp_tanimoto[np.ix_(ecfp_indices, ecfp_indices)]

    for thresh_key, gt_mat in thresholds.items():
        n = min(ecfp_aligned.shape[0], gt_mat.shape[0])
        r = evaluate_cc(ecfp_aligned[:n, :n], gt_mat[:n, :n], "ECFP Tanimoto")
        cc_results[thresh_key].append(r)
        print(f"ECFP Tanimoto ({thresh_key}): AUROC = {r['auroc']:.4f}")

    # --- Binary interaction-prints ---
    for fname, method_name in [
        ("binary_projection.npz", "Projection dot product"),
        ("binary_codebook_tokens.npz", "Codebook token dot product"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            print(f"Skipping {method_name}: {fname} not found")
            continue

        data = np.load(fpath)
        for score_key, score_label in [("score_raw", "raw"), ("score_cosine", "cosine")]:
            if score_key not in data:
                continue
            scores = data[score_key]
            pred_compound_ids = data["compound_ids"].tolist()
            pred_protein_ids = data["protein_ids"].tolist()

            # Build compound-compound similarity from protein-compound score matrix
            # scores is (n_prot, n_mol), transpose to get compound-compound
            # For binary IP: compound j's fingerprint is scores[:, j]
            # CC similarity = cosine between columns
            scores_T = scores.T  # (n_mol, n_prot)
            from sklearn.metrics.pairwise import cosine_similarity
            cc_pred = cosine_similarity(scores_T).astype(np.float32)

            # Align to ground truth ordering
            pred_id_map = {cid: i for i, cid in enumerate(pred_compound_ids)}
            common = [cid for cid in gt_compound_ids if cid in pred_id_map]
            gt_idx = [gt_compound_ids.index(c) for c in common]
            pred_idx = [pred_id_map[c] for c in common]

            cc_pred_aligned = cc_pred[np.ix_(pred_idx, pred_idx)]

            full_name = f"{method_name} ({score_label})"
            for thresh_key, gt_full in thresholds.items():
                gt_aligned = gt_full[np.ix_(gt_idx, gt_idx)]
                r = evaluate_cc(cc_pred_aligned, gt_aligned, full_name)
                cc_results[thresh_key].append(r)
                print(f"{full_name} ({thresh_key}): AUROC = {r['auroc']:.4f}")

    # --- Codebook interaction-prints ---
    for fname in sorted(os.listdir(ip_dir)):
        if not fname.endswith("_concat.npz") and not fname.endswith("_product.npz"):
            continue

        fpath = os.path.join(ip_dir, fname)
        data = np.load(fpath)
        if "similarity" not in data:
            continue

        method_name = fname.replace(".npz", "").replace("_", " ").title()
        cc_pred = data["similarity"]
        pred_compound_ids = data["compound_ids"].tolist()

        pred_id_map = {cid: i for i, cid in enumerate(pred_compound_ids)}
        common = [cid for cid in gt_compound_ids if cid in pred_id_map]
        gt_idx = [gt_compound_ids.index(c) for c in common]
        pred_idx = [pred_id_map[c] for c in common]

        cc_pred_aligned = cc_pred[np.ix_(pred_idx, pred_idx)]

        for thresh_key, gt_full in thresholds.items():
            gt_aligned = gt_full[np.ix_(gt_idx, gt_idx)]
            r = evaluate_cc(cc_pred_aligned, gt_aligned, method_name)
            cc_results[thresh_key].append(r)
            print(f"{method_name} ({thresh_key}): AUROC = {r['auroc']:.4f}")

    # --- Per-target evaluation ---
    print("\n=== Per-target AUROC ===")
    for fname, method_name in [
        ("cg_projection.npz", "Projection"),
        ("cg_codebook_tokens.npz", "Codebook tokens"),
        ("cg_pooled_attention.npz", "Pooled attention"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue

        data = np.load(fpath)
        for score_key, score_label in [("score_raw", "raw"), ("score_cosine", "cosine")]:
            if score_key not in data:
                continue
            cg_scores = data[score_key]
            pred_compound_ids = data["compound_ids"].tolist()
            pred_protein_ids = data["protein_ids"].tolist()

            # Align compounds and genes to ground truth ordering
            c_map = {cid: i for i, cid in enumerate(pred_compound_ids)}
            g_map = {gid: i for i, gid in enumerate(pred_protein_ids)}

            common_c = [cid for cid in gt_compound_ids if cid in c_map]
            common_g = [gid for gid in gt_gene_ids if gid in g_map]

            c_pred_idx = [c_map[c] for c in common_c]
            g_pred_idx = [g_map[g] for g in common_g]
            c_gt_idx = [gt_compound_ids.index(c) for c in common_c]
            g_gt_idx = [gt_gene_ids.index(g) for g in common_g]

            cg_aligned = cg_scores[np.ix_(c_pred_idx, g_pred_idx)]
            cg_gt_aligned = cg_binary[np.ix_(c_gt_idx, g_gt_idx)]

            full_name = f"{method_name} ({score_label})"
            r = evaluate_per_target(cg_aligned, cg_gt_aligned, full_name)
            cg_results.append(r)
            per_target = r.pop("per_target_aurocs")
            print(f"{full_name}: median={r['median_auroc']:.4f}, mean={r['mean_auroc']:.4f}, "
                  f"p25={r['p25_auroc']:.4f}, p75={r['p75_auroc']:.4f} ({r['n_targets_evaluated']} targets)")
            r["per_target_aurocs"] = per_target

    # --- Save results ---
    all_results = {
        "compound_compound_auroc": cc_results,
        "per_target_auroc": [
            {k: v for k, v in r.items() if k != "per_target_aurocs"} for r in cg_results
        ],
    }
    with open(os.path.join(results_dir, "evaluation", "compound_compound_auroc.json"), "w") as f:
        json.dump(cc_results, f, indent=2)
    with open(os.path.join(results_dir, "evaluation", "per_target_auroc.json"), "w") as f:
        json.dump(cg_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("SUMMARY: Compound-Compound AUROC")
    print("=" * 80)
    print(f"{'Method':<45} {'Threshold 0.4':>15} {'Threshold 0.6':>15}")
    print("-" * 80)
    methods_04 = {r["method"]: r["auroc"] for r in cc_results["threshold_04"]}
    methods_06 = {r["method"]: r["auroc"] for r in cc_results["threshold_06"]}
    all_methods = list(dict.fromkeys(list(methods_04.keys()) + list(methods_06.keys())))
    for m in all_methods:
        a04 = methods_04.get(m, float("nan"))
        a06 = methods_06.get(m, float("nan"))
        print(f"{m:<45} {a04:>15.4f} {a06:>15.4f}")

    print(f"\nBeaini baselines: 71-76% (CC), 63-64% (ECFP)")

    print("\n" + "=" * 80)
    print("SUMMARY: Per-Target AUROC")
    print("=" * 80)
    print(f"{'Method':<35} {'Median':>8} {'Mean':>8} {'P25':>8} {'P75':>8} {'N':>5}")
    print("-" * 80)
    for r in cg_results:
        print(f"{r['method']:<35} {r['median_auroc']:>8.4f} {r['mean_auroc']:>8.4f} "
              f"{r['p25_auroc']:>8.4f} {r['p75_auroc']:>8.4f} {r['n_targets_evaluated']:>5}")

    print(f"\nBeaini baseline: 53.9% median per-target")

    return all_results


if __name__ == "__main__":
    results_dir = sys.argv[1]
    gt_dir = sys.argv[2] if len(sys.argv) > 2 else "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
    main(results_dir, gt_dir)
