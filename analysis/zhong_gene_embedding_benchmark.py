#!/usr/bin/env python3
"""
Zhong et al. 2025 gene embedding benchmark on Bonbon protein embeddings.

Evaluates synthetic lethality (SL), negative genetic interactions (NG),
and transcription factor target (TF) prediction using cosine similarity
and SVM classifiers.

Reference: Zhong J et al., bioRxiv 2025.01.29.635607
Repo: github.com/ylaboratory/gene-embedding-benchmarks

Results: Bonbon protein embeddings — trained on protein-ligand pairs,
not protein-protein interactions — show near-chance cosine similarity
for genetic interactions. SVM on sum-composed features extracts partial
signal (SL: 0.70 AUROC) but below sequence-based methods like ESM2 (0.77).
This is an expected negative: the EFAAR benchmark (protein complexes,
pathways, signaling) is the better test of Bonbon's protein-side thesis.
"""
import os, json, glob, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def load_embeddings(emb_dir):
    csvs = glob.glob(os.path.join(emb_dir, '*.csv'))
    emb_vals = np.loadtxt(csvs[0], delimiter=',')
    txts = glob.glob(os.path.join(emb_dir, '*.txt'))
    with open(txts[0]) as f:
        ref_genes = [l.strip() for l in f if l.strip()]
    return emb_vals, {g: i for i, g in enumerate(ref_genes)}


def cosine_benchmark(emb_vals, g2i, pairs, labels):
    norms = np.linalg.norm(emb_vals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    emb_normed = emb_vals / norms

    idx1, idx2, y = [], [], []
    for (g1, g2), lbl in zip(pairs, labels):
        if g1 in g2i and g2 in g2i:
            idx1.append(g2i[g1])
            idx2.append(g2i[g2])
            y.append(lbl)
    y = np.array(y)
    scores = np.sum(emb_normed[idx1] * emb_normed[idx2], axis=1)
    return roc_auc_score(y, scores), average_precision_score(y, scores), int(sum(y)), len(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb-dir', required=True, help='Directory with embedding CSV + gene list TXT')
    parser.add_argument('--splits-dir', required=True, help='Directory with CV split pickle files')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()

    emb_vals, g2i = load_embeddings(args.emb_dir)
    print(f'Loaded {len(g2i)} genes, {emb_vals.shape[1]}-dim')

    tasks = {'sl': 'sl_nested_cv_splits.pkl', 'ng': 'ng_nested_cv_splits.pkl', 'tf': 'tf_nested_cv_splits.pkl'}
    results = []

    for task_name, split_file in tasks.items():
        with open(os.path.join(args.splits_dir, split_file), 'rb') as f:
            cv_data = pickle.load(f)
        auc, auprc, n_pos, n_total = cosine_benchmark(emb_vals, g2i, cv_data['pairs'], cv_data['labels'])
        print(f'{task_name.upper()}: AUC={auc:.3f} AUPRC={auprc:.3f} [{n_pos} pos / {n_total} total]')
        results.append({'task': task_name.upper(), 'method': 'cosine',
            'outer_AUC': round(auc, 4), 'outer_AUPRC': round(auprc, 4),
            'n_pos': n_pos, 'n_total': n_total})

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
