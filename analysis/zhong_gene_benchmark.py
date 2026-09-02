#!/usr/bin/env python
"""
Zhong et al. 2025 Gene Embedding Benchmark on Bonbon Protein Embeddings
========================================================================

Runs the gene-pair interaction benchmarks (SL, NG, TF) from:
  Zhong J, Li L, Dannenfelser R, Yao V. bioRxiv 2025.01.29.635607v2

Converts Bonbon dark-snowball-245 embeddings (keyed by gene symbol / UniProt)
to Entrez Gene IDs and runs the SVM-based evaluation from their codebase.

Embedding types:
  - projector:  1024-dim  (protein_protein_mean)
  - codebook_tokens: 1280-dim  (codebook tokens)
  - codebook_pa: 8192-dim  (codebook pooled attention)
  - fusion: 10496-dim  (L2-normalized concat of all three)

Usage:
  source ~/output/bin/activate
  python analysis/zhong_gene_benchmark.py
"""

import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["OPENBLAS_NUM_THREADS"] = "10"
os.environ["BLIS_NUM_THREADS"] = "10"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import json
import glob
import pickle
import time
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import mygene
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ============================================================================
# Configuration
# ============================================================================

RESULTS_BASE = "/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245"
PROTEOME_BASE = f"{RESULTS_BASE}/human_proteome"

EMBEDDING_DIRS = {
    "projector": f"{PROTEOME_BASE}/human_proteome_pairs_protein_protein_mean_dark-snowball-245",
    "codebook_tokens": f"{PROTEOME_BASE}/human_proteome_pairs_protein_codebook_dark-snowball-245/tokens",
    "codebook_pa": f"{PROTEOME_BASE}/human_proteome_pairs_protein_codebook_dark-snowball-245/pooled_attention",
}

BENCHMARK_ROOT = "/tmp/gene-embedding-benchmarks"
DATA_SPLITS = f"{BENCHMARK_ROOT}/data/data_splits/gene_pair_benchmark"

OUTPUT_DIR = f"{RESULTS_BASE}/evaluation"
CONVERTED_DIR = f"{OUTPUT_DIR}/zhong_embeddings"
RESULTS_JSON = f"{OUTPUT_DIR}/zhong_gene_benchmark_results.json"
MAPPING_CACHE = f"{OUTPUT_DIR}/gene_symbol_to_entrez.json"

C_VALUES = [0.1, 1, 10, 100, 1000]
N_JOBS = 10
OPERATIONS = ["sum", "product"]

TASKS = {
    "SL": f"{DATA_SPLITS}/sl_nested_cv_splits.pkl",
    "NG": f"{DATA_SPLITS}/ng_nested_cv_splits.pkl",
    "TF": f"{DATA_SPLITS}/tf_nested_cv_splits.pkl",
}


# ============================================================================
# Step 1: Gene ID Mapping (Gene Symbol / UniProt -> Entrez)
# ============================================================================

def build_gene_mapping(file_names):
    """Map gene symbols and UniProt IDs to Entrez Gene IDs using mygene."""
    # Check for cached mapping
    if os.path.exists(MAPPING_CACHE):
        print(f"  Loading cached mapping from {MAPPING_CACHE}")
        with open(MAPPING_CACHE) as f:
            return json.load(f)

    mg = mygene.MyGeneInfo()

    uniprot_pattern = re.compile(
        r'^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$'
    )
    gene_symbols = [f for f in file_names if not uniprot_pattern.match(f)]
    uniprot_ids = [f for f in file_names if uniprot_pattern.match(f)]

    print(f"  Mapping {len(gene_symbols)} gene symbols and {len(uniprot_ids)} UniProt IDs to Entrez...")

    mapping = {}

    batch_size = 1000
    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i:i + batch_size]
        results = mg.querymany(
            batch, scopes='symbol', fields='entrezgene',
            species='human', returnall=True
        )
        for hit in results['out']:
            query = hit.get('query')
            entrez = hit.get('entrezgene')
            if entrez and query:
                mapping[query] = str(int(entrez))

    if uniprot_ids:
        for i in range(0, len(uniprot_ids), batch_size):
            batch = uniprot_ids[i:i + batch_size]
            results = mg.querymany(
                batch, scopes='uniprot.Swiss-Prot,uniprot.TrEMBL',
                fields='entrezgene,symbol', species='human', returnall=True
            )
            for hit in results['out']:
                query = hit.get('query')
                entrez = hit.get('entrezgene')
                if entrez and query:
                    mapping[query] = str(int(entrez))

    print(f"  Successfully mapped {len(mapping)}/{len(file_names)} IDs to Entrez")

    # De-duplicate: multiple file names -> same Entrez
    entrez_to_files = defaultdict(list)
    for fname, eid in mapping.items():
        entrez_to_files[eid].append(fname)
    duplicates = {k: v for k, v in entrez_to_files.items() if len(v) > 1}
    if duplicates:
        print(f"  Warning: {len(duplicates)} Entrez IDs mapped from multiple files (keeping gene symbol)")
        for eid, fnames in duplicates.items():
            gene_sym = [f for f in fnames if not uniprot_pattern.match(f)]
            keep = gene_sym[0] if gene_sym else fnames[0]
            for f in fnames:
                if f != keep:
                    del mapping[f]

    # Cache mapping
    os.makedirs(os.path.dirname(MAPPING_CACHE), exist_ok=True)
    with open(MAPPING_CACHE, 'w') as f:
        json.dump(mapping, f, indent=2)

    return mapping


# ============================================================================
# Step 2: Load and Convert Embeddings
# ============================================================================

def load_embeddings(emb_dir, mapping):
    """Load .pt files and assemble into matrix with Entrez gene list."""
    genes_entrez = []
    vectors = []

    for fname in sorted(os.listdir(emb_dir)):
        if not fname.endswith('.pt'):
            continue
        name = fname[:-3]
        if name not in mapping:
            continue
        entrez_id = mapping[name]
        vec = torch.load(os.path.join(emb_dir, fname), map_location='cpu',
                         weights_only=True)
        if vec.dtype == torch.bfloat16:
            vec = vec.float()
        vectors.append(vec.numpy())
        genes_entrez.append(entrez_id)

    # De-duplicate
    seen = {}
    unique_genes = []
    unique_vecs = []
    for g, v in zip(genes_entrez, vectors):
        if g not in seen:
            seen[g] = True
            unique_genes.append(g)
            unique_vecs.append(v)

    matrix = np.vstack(unique_vecs).astype(np.float32)
    print(f"  Loaded {matrix.shape[0]} genes, dim={matrix.shape[1]}")
    return unique_genes, matrix


def save_embedding_format(genes, matrix, out_dir, name):
    """Save in Zhong et al. format: CSV (matrix) + TXT (gene list)."""
    os.makedirs(out_dir, exist_ok=True)
    np.savetxt(os.path.join(out_dir, f"{name}.csv"), matrix, delimiter=',',
               fmt='%.6f')
    with open(os.path.join(out_dir, f"{name}_genes.txt"), 'w') as f:
        for g in genes:
            f.write(g + '\n')
    print(f"  Saved to {out_dir}")


def create_fusion_embedding(all_genes_matrices):
    """L2-normalized concatenation of all embedding types."""
    gene_sets = [set(genes) for genes, _ in all_genes_matrices.values()]
    common_genes = sorted(gene_sets[0].intersection(*gene_sets[1:]))
    print(f"\n  Fusion: {len(common_genes)} common genes across all embedding types")

    concat_parts = []
    for emb_type, (genes, matrix) in all_genes_matrices.items():
        g2i = {g: i for i, g in enumerate(genes)}
        idx = [g2i[g] for g in common_genes]
        sub = matrix[idx]
        norms = np.linalg.norm(sub, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        sub = sub / norms
        concat_parts.append(sub)
        print(f"    {emb_type}: dim={sub.shape[1]}")

    fusion = np.hstack(concat_parts)
    print(f"    Fusion total dim: {fusion.shape[1]}")
    return common_genes, fusion


# ============================================================================
# Step 3: Run Benchmark (adapted from gene_pair_benchmarks.py)
# ============================================================================

def precision_at_k(y_true, y_scores, k=10):
    order = np.argsort(y_scores)[::-1]
    return np.mean(np.array(y_true)[order[:k]])


def filter_pairs_labels(pairs, labels, ref_genes):
    fp, fl, idx = [], [], []
    for i, ((g1, g2), lbl) in enumerate(zip(pairs, labels)):
        if g1 in ref_genes and g2 in ref_genes:
            fp.append((g1, g2))
            fl.append(lbl)
            idx.append(i)
    return fp, fl, idx


def build_features(pairs, emb, g2i, operation):
    if operation == "sum":
        return np.vstack([emb[g2i[g1]] + emb[g2i[g2]] for g1, g2 in pairs])
    elif operation == "product":
        return np.vstack([emb[g2i[g1]] * emb[g2i[g2]] for g1, g2 in pairs])
    elif operation == "concat":
        idx1 = [g2i[g1] for g1, g2 in pairs]
        idx2 = [g2i[g2] for g1, g2 in pairs]
        return np.hstack([emb[idx1], emb[idx2]])
    else:
        raise ValueError(f"Unknown operation: {operation}")


def run_gene_pair_benchmark(genes, matrix, cv_pkl_path, operation="sum",
                            use_linear=False):
    """Run the nested CV SVM benchmark for a gene-pair task.

    Args:
        use_linear: If True, use LinearSVC (much faster for large datasets
                    like NG ~113K and TF ~107K pairs). If False, use RBF SVM
                    per the original Zhong et al. protocol.
    """
    with open(cv_pkl_path, 'rb') as f:
        cv_data = pickle.load(f)
    pairs = cv_data['pairs']
    labels = cv_data['labels']
    cv_splits = cv_data['cv_splits']

    ref_genes = set(genes)
    g2i = {g: i for i, g in enumerate(genes)}

    fp, fl, idx_master = filter_pairs_labels(pairs, labels, ref_genes)
    if not fp:
        print(f"    WARNING: No pairs survive filtering!")
        return None

    X = build_features(fp, matrix, g2i, operation)
    y = np.array(fl)

    orig_pos = sum(1 for l in labels if l == 1)
    filtered_pos = sum(1 for l in fl if l == 1)
    classifier_name = "LinearSVC" if use_linear else "RBF SVM"
    print(f"    Pairs: {len(fp)} (pos: {filtered_pos}/{orig_pos} original), {classifier_name}")

    results = []

    for fold, splits in cv_splits.items():
        train_m = set(splits['train_idx'])
        outer_m = set(splits['test_idx'])
        train_idx = [i for i, m in enumerate(idx_master) if m in train_m]
        outer_idx = [i for i, m in enumerate(idx_master) if m in outer_m]

        X_train, y_train = X[train_idx], y[train_idx]
        X_outer, y_outer = X[outer_idx], y[outer_idx]

        # Map inner splits
        inner_splits_mapped = []
        train_master = [m for m in idx_master if m in train_m]
        full2train = {m: i for i, m in enumerate(train_master)}
        for tr_m, val_m in splits['inner_splits']:
            tr_idx = [full2train[m] for m in tr_m if m in full2train]
            val_idx = [full2train[m] for m in val_m if m in full2train]
            if tr_idx and val_idx:
                inner_splits_mapped.append((tr_idx, val_idx))

        print(f"      Fold {fold}: train={len(train_idx)}, test={len(outer_idx)}", end="")
        sys.stdout.flush()

        t0 = time.time()

        if use_linear:
            # LinearSVC for large datasets (NG, TF)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_outer_s = scaler.transform(X_outer)
            grid = GridSearchCV(
                LinearSVC(class_weight='balanced', max_iter=10000, dual='auto'),
                param_grid={'C': [0.001, 0.01, 0.1, 1, 10]},
                scoring='roc_auc',
                refit=True,
                cv=inner_splits_mapped,
                n_jobs=N_JOBS,
                return_train_score=False,
            )
            grid.fit(X_train_s, y_train)
            inner_time = time.time() - t0
            t0_outer = time.time()
            outer_scores = grid.decision_function(X_outer_s)
        else:
            # RBF SVM for small datasets (SL)
            grid = GridSearchCV(
                SVC(class_weight='balanced'),
                param_grid={'C': C_VALUES},
                scoring='roc_auc',
                refit=True,
                cv=inner_splits_mapped,
                n_jobs=N_JOBS,
                return_train_score=False,
            )
            grid.fit(X_train, y_train)
            inner_time = time.time() - t0
            t0_outer = time.time()
            outer_scores = grid.decision_function(X_outer)

        outer_auc = roc_auc_score(y_outer, outer_scores)
        outer_auprc = average_precision_score(y_outer, outer_scores)
        outer_pr10 = precision_at_k(y_outer, outer_scores, k=10)
        outer_time = time.time() - t0_outer

        results.append({
            'fold': fold,
            'best_C': grid.best_params_['C'],
            'inner_AUC': grid.best_score_,
            'inner_time': inner_time,
            'outer_AUC': outer_auc,
            'outer_AUPRC': outer_auprc,
            'outer_PR@10': outer_pr10,
            'outer_time': outer_time,
        })
        print(f"  AUC={outer_auc:.4f}  AUPRC={outer_auprc:.4f}  ({inner_time:.0f}s)")

    avg = {
        'outer_AUC': np.mean([r['outer_AUC'] for r in results]),
        'outer_AUPRC': np.mean([r['outer_AUPRC'] for r in results]),
        'outer_PR@10': np.mean([r['outer_PR@10'] for r in results]),
        'inner_AUC': np.mean([r['inner_AUC'] for r in results]),
        'orig_pos_pairs': orig_pos,
        'filtered_pos_pairs': filtered_pos,
        'n_genes_used': len(ref_genes & set(g for pair in fp for g in pair)),
        'n_pairs': len(fp),
        'folds': results,
    }
    print(f"    => AVG outer_AUC={avg['outer_AUC']:.4f}  outer_AUPRC={avg['outer_AUPRC']:.4f}")
    return avg


# ============================================================================
# Step 4: Load Published Baselines
# ============================================================================

def load_baselines():
    """Load Zhong et al. published results for comparison.

    Uses the '_full' TSVs which evaluate each method on its own gene set,
    rather than the 'intersect' TSVs that restrict to the common gene
    intersection across all 38 methods. This gives a fair comparison since
    Bonbon has near-complete proteome coverage (98.3%).
    """
    baselines = {}
    for task in ['sl', 'ng', 'tf']:
        # Prefer _full for fair comparison across different gene coverages
        tsv_path = f"{BENCHMARK_ROOT}/results/tsvs/paired_gene_{task}_sum_outerloop_avgs_full.tsv"
        if not os.path.exists(tsv_path):
            tsv_path = f"{BENCHMARK_ROOT}/results/tsvs/paired_gene_{task}_sum_outerloop_avgs.tsv"
        if os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep='\t')
            baselines[task.upper()] = {
                row['embedding']: {
                    'outer_AUC': row['outer_AUC'],
                    'outer_AUPRC': row['outer_AUPRC'],
                    'outer_PR@10': row.get('outer_PR@10', 0),
                    'filtered_pos_pairs': row.get('filtered_pos_pairs', 0),
                }
                for _, row in df.iterrows()
            }
    return baselines


def compute_rank(our_auc, baseline_aucs):
    """Compute rank of our AUC among baselines (1 = best)."""
    all_aucs = sorted(list(baseline_aucs) + [our_auc], reverse=True)
    return all_aucs.index(our_auc) + 1


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CONVERTED_DIR, exist_ok=True)

    print("=" * 80)
    print("Zhong et al. 2025 Gene Embedding Benchmark")
    print("Bonbon dark-snowball-245 Protein Embeddings")
    print("=" * 80)

    # --- Step 1: Build gene mapping ---
    print("\n[Step 1] Building gene symbol / UniProt -> Entrez mapping...")
    proj_dir = EMBEDDING_DIRS["projector"]
    file_names = sorted([f[:-3] for f in os.listdir(proj_dir) if f.endswith('.pt')])
    mapping = build_gene_mapping(file_names)
    print(f"  Mapping has {len(mapping)} entries")

    # --- Step 2: Load and convert all embedding types ---
    print("\n[Step 2] Loading and converting embeddings...")
    all_embeddings = {}

    for emb_name, emb_dir in EMBEDDING_DIRS.items():
        print(f"\n  Loading {emb_name}...")
        genes, matrix = load_embeddings(emb_dir, mapping)
        all_embeddings[emb_name] = (genes, matrix)
        out_dir = os.path.join(CONVERTED_DIR, f"bonbon_{emb_name}")
        save_embedding_format(genes, matrix, out_dir, f"bonbon_{emb_name}")

    print("\n  Creating fusion embedding (L2-norm concat)...")
    fusion_genes, fusion_matrix = create_fusion_embedding(all_embeddings)
    all_embeddings["fusion"] = (fusion_genes, fusion_matrix)
    out_dir = os.path.join(CONVERTED_DIR, "bonbon_fusion")
    save_embedding_format(fusion_genes, fusion_matrix, out_dir, "bonbon_fusion")

    # --- Step 3: Run benchmarks ---
    print("\n[Step 3] Running gene-pair benchmarks...")
    print("=" * 80)
    all_results = {}

    for emb_name, (genes, matrix) in all_embeddings.items():
        all_results[emb_name] = {}
        for task_name, cv_pkl in TASKS.items():
            # Use LinearSVC for large datasets (NG ~113K, TF ~107K pairs)
            # and RBF SVM for smaller SL (~22K pairs) per Zhong protocol
            use_linear = task_name in ('NG', 'TF')
            for op in OPERATIONS:
                key = f"{task_name}_{op}"
                print(f"\n  >> {emb_name} | {task_name} | {op}")
                t0_task = time.time()
                result = run_gene_pair_benchmark(
                    genes, matrix, cv_pkl, operation=op,
                    use_linear=use_linear,
                )
                elapsed = time.time() - t0_task
                print(f"    [{elapsed:.0f}s total]")
                if result:
                    all_results[emb_name][key] = result

    # --- Step 4: Load baselines and compare ---
    print("\n\n[Step 4] Loading published baselines and comparing...")
    baselines = load_baselines()

    comparison = {}
    for emb_name in all_embeddings:
        comparison[emb_name] = {}
        for task_name in ['SL', 'NG', 'TF']:
            key_sum = f"{task_name}_sum"
            if key_sum in all_results.get(emb_name, {}):
                our_auc = all_results[emb_name][key_sum]['outer_AUC']
                our_auprc = all_results[emb_name][key_sum]['outer_AUPRC']
                baseline_aucs = []
                if task_name in baselines:
                    baseline_aucs = [v['outer_AUC'] for v in baselines[task_name].values()]
                rank = compute_rank(our_auc, baseline_aucs)
                total = len(baseline_aucs) + 1
                comparison[emb_name][task_name] = {
                    'outer_AUC': round(our_auc, 4),
                    'outer_AUPRC': round(our_auprc, 4),
                    'rank': rank,
                    'total_methods': total,
                }

    # --- Step 5: Save results ---
    print("\n[Step 5] Saving results...")

    json_results = {
        "metadata": {
            "benchmark": "Zhong et al. 2025 (bioRxiv 2025.01.29.635607v2)",
            "model": "Bonbon dark-snowball-245",
            "tasks": list(TASKS.keys()),
            "operations": OPERATIONS,
            "embedding_types": list(all_embeddings.keys()),
            "embedding_dimensions": {
                "projector": 1024,
                "codebook_tokens": 1280,
                "codebook_pa": 8192,
                "fusion": int(all_embeddings["fusion"][1].shape[1]),
            },
        },
        "results": {},
        "comparison_vs_baselines": comparison,
        "key_baselines": {},
    }

    for emb_name, tasks in all_results.items():
        json_results["results"][emb_name] = {}
        for key, result in tasks.items():
            json_results["results"][emb_name][key] = {
                'outer_AUC': round(result['outer_AUC'], 4),
                'outer_AUPRC': round(result['outer_AUPRC'], 4),
                'outer_PR@10': round(result['outer_PR@10'], 4),
                'n_genes_used': result['n_genes_used'],
                'n_pairs': result['n_pairs'],
                'filtered_pos_pairs': result['filtered_pos_pairs'],
                'orig_pos_pairs': result['orig_pos_pairs'],
                'per_fold': [
                    {
                        'fold': r['fold'],
                        'outer_AUC': round(r['outer_AUC'], 4),
                        'outer_AUPRC': round(r['outer_AUPRC'], 4),
                        'best_C': r['best_C'],
                    }
                    for r in result['folds']
                ],
            }

    key_methods = ['ESM2', 'GENEPT-MODEL3', 'ESMB1', 'T5', 'ALBERT', 'PPI-RAW',
                   'NODE2VEC', 'MASHUP', 'GENE2VEC', 'SCGPT-HUMAN', 'SCGPT-PANCANCER',
                   'SEQVEC', 'BIOCONCEPTVEC-SKIP-GRAM', 'FROGS-ARCHS4']
    for task_name in ['SL', 'NG', 'TF']:
        if task_name in baselines:
            json_results["key_baselines"][task_name] = {}
            for method in key_methods:
                if method in baselines[task_name]:
                    json_results["key_baselines"][task_name][method] = {
                        'outer_AUC': round(baselines[task_name][method]['outer_AUC'], 4),
                    }

    with open(RESULTS_JSON, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_JSON}")

    # --- Print summary table ---
    print("\n" + "=" * 80)
    print("SUMMARY: Bonbon vs Published Baselines (outer_AUC, sum operation)")
    print("=" * 80)

    for task_name in ['SL', 'NG', 'TF']:
        print(f"\n--- {task_name} ---")
        print(f"{'Method':<30} {'AUC':>8} {'AUPRC':>8} {'Rank':>6}")
        print("-" * 55)

        for emb_name in ['projector', 'codebook_tokens', 'codebook_pa', 'fusion']:
            if task_name in comparison.get(emb_name, {}):
                c = comparison[emb_name][task_name]
                label = f"Bonbon-{emb_name}"
                print(f"{label:<30} {c['outer_AUC']:>8.4f} {c['outer_AUPRC']:>8.4f} {c['rank']:>3}/{c['total_methods']}")

        print()
        if task_name in baselines:
            sorted_baselines = sorted(
                baselines[task_name].items(),
                key=lambda x: x[1]['outer_AUC'], reverse=True
            )
            for method, vals in sorted_baselines[:8]:
                print(f"{method:<30} {vals['outer_AUC']:>8.4f} {vals['outer_AUPRC']:>8.4f}")

    # Also print product results
    print("\n" + "=" * 80)
    print("PRODUCT OPERATION RESULTS (outer_AUC)")
    print("=" * 80)
    for task_name in ['SL', 'NG', 'TF']:
        print(f"\n--- {task_name} ---")
        for emb_name in ['projector', 'codebook_tokens', 'codebook_pa', 'fusion']:
            key = f"{task_name}_product"
            if key in all_results.get(emb_name, {}):
                r = all_results[emb_name][key]
                print(f"  Bonbon-{emb_name:<20} AUC={r['outer_AUC']:.4f}  AUPRC={r['outer_AUPRC']:.4f}")

    print("\n" + "=" * 80)
    print("Done!")


if __name__ == "__main__":
    main()
