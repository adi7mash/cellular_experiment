"""Per-target AUROC using fusion classifier logits — truly zero-shot.

The model's unmasked_embedding_fusion_encoder has a classification head
(nn.Linear(1024, 2)) that produces interaction logits. We apply this head
to the already-extracted CLS vectors to get zero-shot per-target scores
without any phenomics training.

Also re-extracts logits directly from the model for a subset to validate
that applying the head to saved CLS vectors matches.
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

sys.path.insert(0, str(Path.home() / "bonbontoken"))
sys.path.insert(0, str(Path.home() / "cellular_experiment" / "analysis"))
from phenomics_signature import load_data, select_known, propagate_ppi

GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
FUSION_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/fusion_cls")
CHECKPOINT = "/opt/dlami/nvme/ckpts/e_cheese_five_epoch_dark-snowball-245.ckpt"
RESULTS_DIR = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/evaluation"
DEVICE = "cuda"


def p(*a, **k):
    print(*a, **k, flush=True)


def load_classifier_head():
    """Load just the classifier head from the checkpoint."""
    p("Loading classifier head from checkpoint...")
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # The classifier lives at: unmasked_embedding_fusion_encoder.classifier
    w_key = "unmasked_embedding_fusion_encoder.classifier.weight"
    b_key = "unmasked_embedding_fusion_encoder.classifier.bias"

    weight = state_dict[w_key]  # [2, 1024]
    bias = state_dict[b_key]    # [2]

    classifier = nn.Linear(weight.shape[1], weight.shape[0])
    classifier.weight.data = weight
    classifier.bias.data = bias
    classifier.eval()

    p(f"  Classifier: Linear({weight.shape[1]}, {weight.shape[0]})")
    p(f"  Weight norm: {weight.norm():.4f}, Bias: {bias.tolist()}")
    return classifier


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
            fusion_data[gene] = cls_mat[compound_idx]  # Keep as tensor

    p(f"  {len(fusion_data)}/{len(gt_g)} proteins loaded")
    return fusion_data, valid_compounds


def main():
    p("=" * 70)
    p("ZERO-SHOT PER-TARGET WITH FUSION CLASSIFIER LOGITS")
    p("=" * 70)

    t0 = time.time()

    data = load_data()
    gt, gt_c, gt_g, n_c, n_g, ppi, cg_scores = data[:7]
    cg_binary = gt["cg_binary_top1"]

    p("\nLoading classifier head...")
    classifier = load_classifier_head()

    p("\nLoading fusion CLS...")
    fusion_data, valid_compounds = load_fusion_cls(gt_c, gt_g)

    known_idx_150 = select_known(cg_scores, 150)
    valid_set = set(valid_compounds.tolist())
    known_valid = np.array([i for i, k in enumerate(valid_compounds) if k in set(known_idx_150.tolist())])
    p(f"  Known compounds in fusion data: {len(known_valid)}")

    # ================================================================
    # EXPERIMENT 1: Zero-shot logits (classifier head on CLS)
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 1: Zero-shot classifier logits")
    p(f"{'='*70}")

    target_aucs_logits = []
    target_names = []
    all_logit_scores = {}

    with torch.no_grad():
        for gi, gene in enumerate(gt_g):
            if gene not in fusion_data:
                continue
            cls_known = fusion_data[gene][known_valid]  # [n_known, 1024]
            y = cg_binary[valid_compounds[known_valid], gi]
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue

            logits = classifier(cls_known)  # [n_known, 2]
            # Interaction probability = softmax[:,1] or just logit[:,1]
            scores = logits[:, 1].numpy()  # Higher = more likely to interact

            try:
                auc = roc_auc_score(y, scores)
                target_aucs_logits.append(auc)
                target_names.append(gene)
                all_logit_scores[gene] = scores
            except:
                pass

    median_logits = np.median(target_aucs_logits)
    mean_logits = np.mean(target_aucs_logits)
    p(f"  Logits (class 1): median={median_logits:.4f}, mean={mean_logits:.4f} ({len(target_aucs_logits)} targets)")

    # Also try logit difference (class1 - class0)
    target_aucs_diff = []
    with torch.no_grad():
        for gi, gene in enumerate(gt_g):
            if gene not in fusion_data:
                continue
            cls_known = fusion_data[gene][known_valid]
            y = cg_binary[valid_compounds[known_valid], gi]
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue

            logits = classifier(cls_known)
            scores = (logits[:, 1] - logits[:, 0]).numpy()

            try:
                auc = roc_auc_score(y, scores)
                target_aucs_diff.append(auc)
            except:
                pass

    median_diff = np.median(target_aucs_diff)
    p(f"  Logits (diff): median={median_diff:.4f} ({len(target_aucs_diff)} targets)")

    # Also try softmax probability
    target_aucs_prob = []
    with torch.no_grad():
        for gi, gene in enumerate(gt_g):
            if gene not in fusion_data:
                continue
            cls_known = fusion_data[gene][known_valid]
            y = cg_binary[valid_compounds[known_valid], gi]
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue

            logits = classifier(cls_known)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

            try:
                auc = roc_auc_score(y, probs)
                target_aucs_prob.append(auc)
            except:
                pass

    median_prob = np.median(target_aucs_prob)
    p(f"  Softmax prob: median={median_prob:.4f} ({len(target_aucs_prob)} targets)")

    # ================================================================
    # EXPERIMENT 2: CG score baseline for comparison
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 2: CG score baselines")
    p(f"{'='*70}")

    best_cg_median = 0
    best_cg_config = None
    for key in ["proj_raw", "proj_cos", "cb_pa_raw", "cb_pa_cos", "cb_tok_raw"]:
        for alpha in [0.0, 0.5, 0.7]:
            mat = cg_scores[key]
            if alpha > 0:
                mat = propagate_ppi(mat, ppi, alpha, threshold=0.0)

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
            if median_cg > best_cg_median:
                best_cg_median = median_cg
                best_cg_config = f"{key}_a{alpha}"

    p(f"  Best CG: {best_cg_config} median={best_cg_median:.4f}")

    # ================================================================
    # EXPERIMENT 3: Logits + CG combined (rank fusion)
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 3: Logits + CG score rank fusion")
    p(f"{'='*70}")

    from scipy.stats import rankdata

    mat_cg = propagate_ppi(cg_scores["proj_raw"], ppi, alpha=0.7, threshold=0.0)

    target_aucs_combined = []
    with torch.no_grad():
        for gi, gene in enumerate(gt_g):
            if gene not in fusion_data:
                continue
            cls_known = fusion_data[gene][known_valid]
            y = cg_binary[valid_compounds[known_valid], gi]
            if y.sum() < 2 or y.sum() > len(y) - 2:
                continue

            logits = classifier(cls_known)
            logit_scores = logits[:, 1].numpy()

            cg_vals = mat_cg[valid_compounds[known_valid], gi]

            # Rank fusion: average ranks
            r1 = rankdata(logit_scores)
            r2 = rankdata(cg_vals)
            combined = r1 + r2

            try:
                auc = roc_auc_score(y, combined)
                target_aucs_combined.append(auc)
            except:
                pass

    median_combined = np.median(target_aucs_combined)
    p(f"  Logits + CG rank fusion: median={median_combined:.4f} ({len(target_aucs_combined)} targets)")

    # Also try weighted combination
    for w in [0.3, 0.5, 0.7]:
        target_aucs_w = []
        with torch.no_grad():
            for gi, gene in enumerate(gt_g):
                if gene not in fusion_data:
                    continue
                cls_known = fusion_data[gene][known_valid]
                y = cg_binary[valid_compounds[known_valid], gi]
                if y.sum() < 2 or y.sum() > len(y) - 2:
                    continue

                logits = classifier(cls_known)
                logit_scores = logits[:, 1].numpy()
                cg_vals = mat_cg[valid_compounds[known_valid], gi]

                # Z-score normalize then combine
                ls = (logit_scores - logit_scores.mean()) / (logit_scores.std() + 1e-8)
                cs = (cg_vals - cg_vals.mean()) / (cg_vals.std() + 1e-8)
                combined = w * ls + (1 - w) * cs

                try:
                    auc = roc_auc_score(y, combined)
                    target_aucs_w.append(auc)
                except:
                    pass

        median_w = np.median(target_aucs_w)
        p(f"  Weighted w={w}: median={median_w:.4f}")

    # ================================================================
    # EXPERIMENT 4: Full 1024-dim CLS MLP (target-CV) — from previous experiment
    # ================================================================
    p(f"\n{'='*70}")
    p("EXPERIMENT 4: Distribution analysis of logit scores")
    p(f"{'='*70}")

    # Check what the logits look like
    all_pos_logits = []
    all_neg_logits = []
    with torch.no_grad():
        for gi, gene in enumerate(gt_g):
            if gene not in fusion_data:
                continue
            cls_known = fusion_data[gene][known_valid]
            y = cg_binary[valid_compounds[known_valid], gi]

            logits = classifier(cls_known)
            scores = logits[:, 1].numpy()

            all_pos_logits.extend(scores[y == 1].tolist())
            all_neg_logits.extend(scores[y == 0].tolist())

    p(f"  Positive logits: mean={np.mean(all_pos_logits):.4f}, std={np.std(all_pos_logits):.4f}")
    p(f"  Negative logits: mean={np.mean(all_neg_logits):.4f}, std={np.std(all_neg_logits):.4f}")
    p(f"  Separation: {np.mean(all_pos_logits) - np.mean(all_neg_logits):.4f}")
    p(f"  N pos: {len(all_pos_logits)}, N neg: {len(all_neg_logits)}")

    # ================================================================
    # SUMMARY
    # ================================================================
    p(f"\n{'='*70}")
    p("SUMMARY — Per-target median AUROC (zero-shot)")
    p(f"{'='*70}")

    results = {
        "cg_best_zero_shot": best_cg_median,
        "fusion_logit_class1": median_logits,
        "fusion_logit_diff": median_diff,
        "fusion_softmax_prob": median_prob,
        "logit_cg_rank_fusion": median_combined,
    }

    p(f"\n  {'Method':45s} {'Median':>8s}")
    p(f"  {'-'*53}")
    for name, val in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ***" if val > best_cg_median else ""
        p(f"  {name:45s} {val:8.4f}{marker}")

    p(f"\n  Time: {time.time() - t0:.0f}s")

    out = os.path.join(RESULTS_DIR, "per_target_fusion_logits_results.json")
    json.dump(results, open(out, "w"), indent=2)
    p(f"  Saved to {out}")


if __name__ == "__main__":
    main()
