"""Active Moiety Attention Analysis — Phase 16.

Extract which molecular fragments drive the phenomics signal by analyzing
Bonbon's codebook sequence_attention over molecule tokens.

For each gene target, compare attention patterns of compounds that are
phenomically active (top cosine similarity) vs inactive (bottom). Identify
SAFE tokens (molecular fragments) that are consistently high-attention
across active compounds.
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu, rankdata

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/opt/dlami/nvme/rxrx3_phenomics")
GT_DIR = BASE / "ground_truth"
DATA_DIR = BASE / "data"
SA_DIR = BASE / "dark-snowball-245/rxrx3_pairs_molecule_codebook_seqattn_dark-snowball-245/rxrx3_pairs_molecule_codebook_dark-snowball-245/sequence_attention"
MOL_TOK_DIR = BASE / "dark-snowball-245/rxrx3_pairs_molecule_codebook_dark-snowball-245/tokens"
PROT_TOK_DIR = BASE / "dark-snowball-245/rxrx3_pairs_protein_codebook_dark-snowball-245/tokens"
MOL_PROJ_DIR = BASE / "dark-snowball-245/rxrx3_pairs_molecule_molecule_mean_dark-snowball-245"
PROT_PROJ_DIR = BASE / "dark-snowball-245/rxrx3_pairs_protein_protein_mean_dark-snowball-245"
OUT_DIR = BASE / "results/dark-snowball-245/evaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def p(*a, **k):
    print(*a, **k, flush=True)


def load_tokenizer():
    from bonbontoken.tokenizer import SAFETokenizer
    tok = SAFETokenizer.from_pretrained("datamol-io/safe-gpt")
    return tok.tokenizer


def tokenize_safe(tokenizer, safe_str):
    encoded = tokenizer.encode(safe_str)
    return encoded.tokens, encoded.ids


def get_safe_fragments(safe_str):
    """Split SAFE string on '.' to get molecular fragments."""
    return safe_str.split(".")


def map_tokens_to_fragments(tokens, safe_str):
    """Map each token index to its SAFE fragment index.

    SAFE fragments are separated by '.'. We track which fragment
    each token belongs to by counting '.' tokens.
    """
    fragments = get_safe_fragments(safe_str)
    frag_idx = []
    current_frag = 0

    for tok in tokens:
        if tok == ".":
            current_frag += 1
            frag_idx.append(-1)  # separator, not part of any fragment
        elif tok in ("[CLS]", "[SEP]", "[PAD]"):
            frag_idx.append(-1)
        else:
            frag_idx.append(min(current_frag, len(fragments) - 1))

    return frag_idx, fragments


def compute_token_importance(seq_attn):
    """Compute per-token importance from sequence_attention [seq_len, 8192].

    Importance = L2 norm of the attention vector for each token position.
    Higher norm → the codebook attends more strongly to that token.
    """
    return torch.norm(seq_attn.float(), dim=1).numpy()


def compute_fragment_importance(token_importance, frag_indices, n_fragments):
    """Aggregate token importance to fragment level."""
    frag_importance = np.zeros(n_fragments)
    frag_counts = np.zeros(n_fragments)

    for ti, fi in zip(token_importance, frag_indices):
        if fi >= 0:
            frag_importance[fi] += ti
            frag_counts[fi] += 1

    # Mean importance per fragment
    mask = frag_counts > 0
    frag_importance[mask] /= frag_counts[mask]
    return frag_importance


def main():
    t0 = time.time()
    p("=" * 70)
    p("ACTIVE MOIETY ATTENTION ANALYSIS — Phase 16")
    p("=" * 70)

    # Load tokenizer
    p("\n1. Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Load ground truth and compound-gene similarities
    p("2. Loading phenomics data...")
    gt = dict(np.load(GT_DIR / "ground_truth_binary.npz", allow_pickle=True))
    gt_compounds = gt["compound_ids"].tolist()
    gt_genes = gt["gene_ids"].tolist()
    n_c, n_g = len(gt_compounds), len(gt_genes)
    p(f"   {n_c} compounds, {n_g} genes")

    # Load compound SAFE strings
    comp_safe = pd.read_csv(DATA_DIR / "compound_to_safe.tsv", sep="\t")
    safe_map = dict(zip(comp_safe["compound_id"], comp_safe["SAFE"]))

    # Load Bonbon projector CG scores (best single predictor for per-target)
    p("3. Loading Bonbon embeddings and computing CG similarities...")

    mol_embeds = {}
    prot_embeds = {}
    for c in gt_compounds:
        f = MOL_TOK_DIR / f"{c}.pt"
        if f.exists():
            mol_embeds[c] = torch.load(f, weights_only=True).float().numpy()
    for g in gt_genes:
        f = PROT_TOK_DIR / f"{g}.pt"
        if f.exists():
            prot_embeds[g] = torch.load(f, weights_only=True).float().numpy()
    p(f"   Loaded {len(mol_embeds)} molecule, {len(prot_embeds)} protein codebook tokens")

    # Compute CG cosine similarity (codebook tokens are unit-normed)
    valid_compounds = [c for c in gt_compounds if c in mol_embeds]
    valid_genes = [g for g in gt_genes if g in prot_embeds]
    mol_mat = np.stack([mol_embeds[c] for c in valid_compounds])
    prot_mat = np.stack([prot_embeds[g] for g in valid_genes])
    cg_sim = mol_mat @ prot_mat.T  # [n_c_valid, n_g_valid]
    p(f"   CG similarity: {cg_sim.shape}")

    # Load CG binary ground truth (top 1% per target)
    cg_binary = np.array(gt["cg_binary_top1"])
    vc_idx = {c: i for i, c in enumerate(gt_compounds)}
    vg_idx = {g: i for i, g in enumerate(gt_genes)}

    # ── Per-target attention analysis ──────────────────────────────────────
    p("\n4. Analyzing attention patterns per target...")

    # Load sequence_attention tensors
    sa_cache = {}
    loaded = 0
    for c in valid_compounds:
        f = SA_DIR / f"{c}.pt"
        if f.exists():
            sa_cache[c] = torch.load(f, weights_only=True)
            loaded += 1
    p(f"   Loaded {loaded} sequence_attention tensors")

    # For each gene target, find active vs inactive compounds
    results_per_target = {}
    n_targets_analyzed = 0
    global_fragment_stats = defaultdict(lambda: {"active_importance": [], "inactive_importance": [], "count": 0})

    for gi, gene in enumerate(valid_genes):
        gt_gi = vg_idx[gene]

        # Get active/inactive compounds using Bonbon CG similarity
        scores = cg_sim[:, gi]
        n_active = max(int(0.05 * len(valid_compounds)), 5)  # top 5%
        active_idx = np.argsort(-scores)[:n_active]
        inactive_idx = np.argsort(scores)[:n_active]

        active_compounds = [valid_compounds[i] for i in active_idx]
        inactive_compounds = [valid_compounds[i] for i in inactive_idx]

        # Compute fragment-level importance for active vs inactive
        active_frag_importances = []
        inactive_frag_importances = []

        for comp_list, importance_list in [(active_compounds, active_frag_importances),
                                           (inactive_compounds, inactive_frag_importances)]:
            for c in comp_list:
                if c not in sa_cache or c not in safe_map:
                    continue

                sa = sa_cache[c]
                safe_str = safe_map[c]
                tokens, _ = tokenize_safe(tokenizer, safe_str)

                if len(tokens) != sa.shape[0]:
                    continue

                token_imp = compute_token_importance(sa)
                frag_indices, fragments = map_tokens_to_fragments(tokens, safe_str)
                n_frags = len(fragments)
                frag_imp = compute_fragment_importance(token_imp, frag_indices, n_frags)

                # Normalize fragment importance within each molecule
                total = frag_imp.sum()
                if total > 0:
                    frag_imp_norm = frag_imp / total
                else:
                    frag_imp_norm = frag_imp

                importance_list.append({
                    "compound": c,
                    "safe": safe_str,
                    "fragments": fragments,
                    "fragment_importance": frag_imp_norm.tolist(),
                    "raw_importance": frag_imp.tolist(),
                    "token_importance": token_imp.tolist(),
                    "tokens": tokens,
                })

        if len(active_frag_importances) < 3 or len(inactive_frag_importances) < 3:
            continue

        # Compute mean token importance (aggregated over codebook dim)
        active_mean_imp = np.mean([np.mean(d["token_importance"]) for d in active_frag_importances])
        inactive_mean_imp = np.mean([np.mean(d["token_importance"]) for d in inactive_frag_importances])

        # Statistical test: are active compounds' attention magnitudes different?
        active_norms = [np.mean(d["token_importance"]) for d in active_frag_importances]
        inactive_norms = [np.mean(d["token_importance"]) for d in inactive_frag_importances]

        try:
            stat, pval = mannwhitneyu(active_norms, inactive_norms, alternative="two-sided")
        except ValueError:
            pval = 1.0

        # Track the top active compound's most important fragment
        top_active = active_frag_importances[0]
        top_frag_idx = np.argmax(top_active["fragment_importance"])
        top_fragment = top_active["fragments"][top_frag_idx]

        results_per_target[gene] = {
            "gene": gene,
            "n_active": len(active_frag_importances),
            "n_inactive": len(inactive_frag_importances),
            "active_mean_attention_norm": float(active_mean_imp),
            "inactive_mean_attention_norm": float(inactive_mean_imp),
            "attention_ratio": float(active_mean_imp / (inactive_mean_imp + 1e-10)),
            "mannwhitney_pval": float(pval),
            "top_compound": top_active["compound"],
            "top_compound_safe": top_active["safe"],
            "top_fragment": top_fragment,
            "top_fragment_importance": float(top_active["fragment_importance"][top_frag_idx]),
            "top_compound_score": float(scores[active_idx[0]]),
        }

        n_targets_analyzed += 1

    p(f"   Analyzed {n_targets_analyzed} targets")

    # ── Global analysis: codebook dimension activation patterns ──────────
    p("\n5. Analyzing codebook activation patterns...")

    # For each compound, compute mean activation per codebook dimension
    # Then compare active vs inactive across all targets
    all_active_activations = []
    all_inactive_activations = []

    for gi, gene in enumerate(valid_genes[:100]):  # Sample 100 targets
        scores = cg_sim[:, gi]
        n_active = max(int(0.05 * len(valid_compounds)), 5)
        active_idx = np.argsort(-scores)[:n_active]
        inactive_idx = np.argsort(scores)[:n_active]

        for idx in active_idx:
            c = valid_compounds[idx]
            if c in sa_cache:
                mean_act = sa_cache[c].float().mean(dim=0).numpy()
                all_active_activations.append(mean_act)

        for idx in inactive_idx:
            c = valid_compounds[idx]
            if c in sa_cache:
                mean_act = sa_cache[c].float().mean(dim=0).numpy()
                all_inactive_activations.append(mean_act)

    active_mean = np.mean(all_active_activations, axis=0)
    inactive_mean = np.mean(all_inactive_activations, axis=0)
    diff = active_mean - inactive_mean

    # Top differential codebook dimensions
    top_dims = np.argsort(-np.abs(diff))[:50]
    p(f"   Top differential codebook dimensions: {top_dims[:10]}")
    p(f"   Max |diff|: {np.abs(diff).max():.6f}")
    p(f"   Mean |diff|: {np.abs(diff).mean():.6f}")

    # Codebook sparsity analysis
    p("\n6. Codebook sparsity analysis...")
    sparsity_active = []
    sparsity_inactive = []
    entropy_active = []
    entropy_inactive = []

    for gi, gene in enumerate(valid_genes[:100]):
        scores = cg_sim[:, gi]
        n_active = max(int(0.05 * len(valid_compounds)), 5)
        active_idx = np.argsort(-scores)[:n_active]
        inactive_idx = np.argsort(scores)[:n_active]

        for idx in active_idx:
            c = valid_compounds[idx]
            if c in sa_cache:
                sa = sa_cache[c].float()
                # Sparsity: fraction of near-zero entries
                sparsity = (sa.abs() < 0.01).float().mean().item()
                sparsity_active.append(sparsity)
                # Entropy of attention magnitude distribution across codebook
                mag = sa.abs().mean(dim=0)
                mag_norm = mag / (mag.sum() + 1e-10)
                ent = -(mag_norm * (mag_norm + 1e-10).log()).sum().item()
                entropy_active.append(ent)

        for idx in inactive_idx:
            c = valid_compounds[idx]
            if c in sa_cache:
                sa = sa_cache[c].float()
                sparsity = (sa.abs() < 0.01).float().mean().item()
                sparsity_inactive.append(sparsity)
                mag = sa.abs().mean(dim=0)
                mag_norm = mag / (mag.sum() + 1e-10)
                ent = -(mag_norm * (mag_norm + 1e-10).log()).sum().item()
                entropy_inactive.append(ent)

    p(f"   Active compounds:   sparsity={np.mean(sparsity_active):.4f}, entropy={np.mean(entropy_active):.2f}")
    p(f"   Inactive compounds: sparsity={np.mean(sparsity_inactive):.4f}, entropy={np.mean(entropy_inactive):.2f}")

    # ── Per-token attention heatmap for top targets ────────────────────────
    p("\n7. Generating per-token attention maps for top targets...")

    # Sort targets by attention ratio
    sorted_targets = sorted(results_per_target.values(), key=lambda x: -x["attention_ratio"])

    exemplars = []
    for target in sorted_targets[:20]:
        gene = target["gene"]
        gi = valid_genes.index(gene)
        scores = cg_sim[:, gi]
        top_idx = np.argmax(scores)
        top_compound = valid_compounds[top_idx]

        if top_compound not in sa_cache or top_compound not in safe_map:
            continue

        sa = sa_cache[top_compound]
        safe_str = safe_map[top_compound]
        tokens, _ = tokenize_safe(tokenizer, safe_str)

        if len(tokens) != sa.shape[0]:
            continue

        token_imp = compute_token_importance(sa)
        frag_indices, fragments = map_tokens_to_fragments(tokens, safe_str)
        n_frags = len(fragments)
        frag_imp = compute_fragment_importance(token_imp, frag_indices, n_frags)

        total = frag_imp.sum()
        if total > 0:
            frag_imp_norm = (frag_imp / total).tolist()
        else:
            frag_imp_norm = frag_imp.tolist()

        exemplars.append({
            "gene": gene,
            "compound": top_compound,
            "safe": safe_str,
            "score": float(scores[top_idx]),
            "fragments": fragments,
            "fragment_importance": frag_imp_norm,
            "raw_fragment_importance": frag_imp.tolist(),
            "tokens": [t for t in tokens if t not in ("[CLS]", "[SEP]", "[PAD]")],
            "token_importance": [float(x) for t, x in zip(tokens, token_imp)
                                 if t not in ("[CLS]", "[SEP]", "[PAD]")],
        })

    p(f"   Generated {len(exemplars)} exemplar attention maps")

    # ── Fragment-level differential attention ──────────────────────────────
    p("\n8. Fragment-level differential attention (top 20 targets)...")

    fragment_diffs = []
    for target in sorted_targets[:20]:
        gene = target["gene"]
        gi = valid_genes.index(gene)
        scores = cg_sim[:, gi]
        n_sel = max(int(0.05 * len(valid_compounds)), 5)
        active_idx = np.argsort(-scores)[:n_sel]
        inactive_idx = np.argsort(scores)[:n_sel]

        # For each active compound, identify its most attended fragment
        for ai in active_idx[:3]:
            c = valid_compounds[ai]
            if c not in sa_cache or c not in safe_map:
                continue
            sa = sa_cache[c]
            safe_str = safe_map[c]
            tokens, _ = tokenize_safe(tokenizer, safe_str)
            if len(tokens) != sa.shape[0]:
                continue

            token_imp = compute_token_importance(sa)
            frag_indices, fragments = map_tokens_to_fragments(tokens, safe_str)
            n_frags = len(fragments)
            frag_imp = compute_fragment_importance(token_imp, frag_indices, n_frags)

            if n_frags > 1:
                top_fi = np.argmax(frag_imp)
                fragment_diffs.append({
                    "gene": gene,
                    "compound": c,
                    "top_fragment": fragments[top_fi],
                    "top_fragment_idx": int(top_fi),
                    "n_fragments": n_frags,
                    "importance_ratio": float(frag_imp[top_fi] / (np.mean(frag_imp) + 1e-10)),
                    "score": float(scores[ai]),
                })

    p(f"   {len(fragment_diffs)} fragment-level differential records")

    # ── Summary statistics ──────────────────────────────────────────────────
    p("\n" + "=" * 70)
    p("SUMMARY")
    p("=" * 70)

    significant = sum(1 for r in results_per_target.values() if r["mannwhitney_pval"] < 0.05)
    total = len(results_per_target)
    p(f"\nTargets with significant attention difference (p<0.05): {significant}/{total} ({100*significant/total:.1f}%)")

    ratios = [r["attention_ratio"] for r in results_per_target.values()]
    p(f"Attention ratio (active/inactive): mean={np.mean(ratios):.4f}, median={np.median(ratios):.4f}")
    p(f"  Range: [{min(ratios):.4f}, {max(ratios):.4f}]")

    p(f"\nCodebook sparsity:")
    p(f"  Active:   {np.mean(sparsity_active):.4f} ± {np.std(sparsity_active):.4f}")
    p(f"  Inactive: {np.mean(sparsity_inactive):.4f} ± {np.std(sparsity_inactive):.4f}")
    p(f"Codebook entropy:")
    p(f"  Active:   {np.mean(entropy_active):.2f} ± {np.std(entropy_active):.2f}")
    p(f"  Inactive: {np.mean(entropy_inactive):.2f} ± {np.std(entropy_inactive):.2f}")

    # Top exemplars
    p(f"\nTop 10 targets by attention ratio:")
    p(f"{'Gene':12s} {'Top Compound':30s} {'Score':>7s} {'Ratio':>7s} {'Top Fragment':30s}")
    p("-" * 90)
    for target in sorted_targets[:10]:
        p(f"{target['gene']:12s} {target['top_compound'][:30]:30s} {target['top_compound_score']:7.4f} "
          f"{target['attention_ratio']:7.4f} {target['top_fragment'][:30]:30s}")

    # Save results
    p("\n9. Saving results...")
    results = {
        "metadata": {
            "checkpoint": "dark-snowball-245",
            "n_compounds": len(valid_compounds),
            "n_genes": len(valid_genes),
            "n_targets_analyzed": n_targets_analyzed,
            "n_significant_p005": significant,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "per_target": results_per_target,
        "exemplars": exemplars,
        "fragment_diffs": fragment_diffs,
        "codebook_analysis": {
            "active_sparsity": {"mean": float(np.mean(sparsity_active)), "std": float(np.std(sparsity_active))},
            "inactive_sparsity": {"mean": float(np.mean(sparsity_inactive)), "std": float(np.std(sparsity_inactive))},
            "active_entropy": {"mean": float(np.mean(entropy_active)), "std": float(np.std(entropy_active))},
            "inactive_entropy": {"mean": float(np.mean(entropy_inactive)), "std": float(np.std(entropy_inactive))},
            "top_differential_codebook_dims": top_dims.tolist(),
            "max_abs_diff": float(np.abs(diff).max()),
        },
        "summary": {
            "attention_ratio_mean": float(np.mean(ratios)),
            "attention_ratio_median": float(np.median(ratios)),
            "fraction_significant": float(significant / total),
        },
    }

    out_path = OUT_DIR / "active_moiety_attention.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    p(f"   Saved to {out_path}")

    elapsed = time.time() - t0
    p(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
