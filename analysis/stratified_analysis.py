"""Stratified analysis: per-protein-family and GO-term AUROC (Phase 5c)."""
import json
import os
import sys
import time
import requests
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def fetch_uniprot_annotations(gene_to_protein_path: str) -> pd.DataFrame:
    """Fetch protein family and GO annotations from UniProt."""
    genes = pd.read_csv(gene_to_protein_path, sep="\t")
    print(f"Fetching annotations for {len(genes)} proteins...")

    rows = []
    batch_size = 50

    for i in range(0, len(genes), batch_size):
        batch = genes.iloc[i:i + batch_size]
        accessions = batch["uniprot_id"].tolist()
        query = " OR ".join(f"(accession:{a})" for a in accessions)

        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://rest.uniprot.org/uniprotkb/search",
                    params={
                        "query": query,
                        "format": "json",
                        "size": 500,
                        "fields": "accession,keyword,go_c,go_f,protein_families",
                    },
                )
                resp.raise_for_status()
                entries = resp.json().get("results", [])
                break
            except Exception as e:
                print(f"  Batch {i // batch_size + 1} attempt {attempt + 1} failed: {e}")
                time.sleep(3)
        else:
            continue

        for entry in entries:
            accession = entry.get("primaryAccession", "")

            keywords = []
            for kw in entry.get("keywords", []):
                keywords.append(kw.get("name", ""))

            go_components = []
            go_functions = []
            for xref in entry.get("uniProtKBCrossReferences", []):
                if xref.get("database") == "GO":
                    go_id = xref.get("id", "")
                    props = {p["key"]: p["value"] for p in xref.get("properties", [])}
                    term = props.get("GoTerm", "")
                    if term.startswith("C:"):
                        go_components.append(term[2:])
                    elif term.startswith("F:"):
                        go_functions.append(term[2:])

            protein_family = ""
            for comment in entry.get("comments", []):
                if comment.get("commentType") == "SIMILARITY":
                    texts = comment.get("texts", [])
                    if texts:
                        protein_family = texts[0].get("value", "")

            is_kinase = any("kinase" in k.lower() for k in keywords)
            is_gpcr = any("gpcr" in k.lower() or "g-protein coupled" in k.lower() for k in keywords)
            is_protease = any("protease" in k.lower() or "peptidase" in k.lower() for k in keywords)
            is_nuclear_receptor = any("nuclear receptor" in k.lower() for k in keywords)

            family = "Other"
            if is_kinase:
                family = "Kinase"
            elif is_gpcr:
                family = "GPCR"
            elif is_protease:
                family = "Protease"
            elif is_nuclear_receptor:
                family = "Nuclear receptor"

            gene_row = genes[genes["uniprot_id"] == accession]
            gene_symbol = gene_row["gene_symbol"].iloc[0] if len(gene_row) > 0 else ""

            rows.append({
                "gene_symbol": gene_symbol,
                "uniprot_id": accession,
                "protein_family": family,
                "keywords": "; ".join(keywords),
                "go_cellular_component": "; ".join(go_components),
                "go_molecular_function": "; ".join(go_functions),
                "protein_family_detail": protein_family,
            })

        time.sleep(0.3)
        if (i // batch_size + 1) % 5 == 0:
            print(f"  Fetched {i + len(batch)}/{len(genes)}...")

    annotations = pd.DataFrame(rows)
    return annotations


def stratified_auroc(cg_scores: np.ndarray, cg_binary: np.ndarray,
                     gene_ids: list, annotations: pd.DataFrame,
                     column: str, method_name: str) -> dict:
    """Compute per-stratum median AUROC."""
    ann_map = {}
    for _, row in annotations.iterrows():
        gene = row["gene_symbol"]
        values = str(row[column]).split("; ")
        for v in values:
            v = v.strip()
            if v and v != "nan":
                if v not in ann_map:
                    ann_map[v] = []
                ann_map[v].append(gene)

    results = {}
    for stratum, stratum_genes in ann_map.items():
        gene_indices = [i for i, g in enumerate(gene_ids) if g in stratum_genes]
        if len(gene_indices) < 3:
            continue

        aurocs = []
        for j in gene_indices:
            y_true = cg_binary[:, j]
            y_score = cg_scores[:, j]
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue
            try:
                aurocs.append(roc_auc_score(y_true, y_score))
            except ValueError:
                continue

        if aurocs:
            results[stratum] = {
                "n_targets": len(gene_indices),
                "n_evaluated": len(aurocs),
                "median_auroc": float(np.median(aurocs)),
                "mean_auroc": float(np.mean(aurocs)),
            }

    return {"method": method_name, "column": column, "strata": results}


def main(results_dir: str, gt_dir: str, data_dir: str):
    os.makedirs(os.path.join(results_dir, "evaluation"), exist_ok=True)

    ann_path = os.path.join(gt_dir, "gene_annotations.tsv")
    if os.path.exists(ann_path):
        print("Loading cached annotations...")
        annotations = pd.read_csv(ann_path, sep="\t")
    else:
        print("Fetching annotations from UniProt...")
        annotations = fetch_uniprot_annotations(os.path.join(data_dir, "gene_to_protein.tsv"))
        annotations.to_csv(ann_path, sep="\t", index=False)
        print(f"Saved annotations to {ann_path}")

    print(f"\nProtein family distribution:")
    print(annotations["protein_family"].value_counts().to_string())

    gt_binary = np.load(os.path.join(gt_dir, "ground_truth_binary.npz"))
    cg_binary = gt_binary["cg_binary_top1"]
    gt_compound_ids = gt_binary["compound_ids"].tolist()
    gt_gene_ids = gt_binary["gene_ids"].tolist()

    ip_dir = os.path.join(results_dir, "interaction_prints")
    all_stratified = []

    for fname, method_name in [
        ("cg_projection.npz", "Projection"),
        ("cg_codebook_tokens.npz", "Codebook tokens"),
        ("cg_pooled_attention.npz", "Pooled attention"),
    ]:
        fpath = os.path.join(ip_dir, fname)
        if not os.path.exists(fpath):
            continue

        data = np.load(fpath)
        for score_key, score_label in [("score_cosine", "cosine")]:
            if score_key not in data:
                continue

            cg_scores = data[score_key]
            pred_c_ids = data["compound_ids"].tolist()
            pred_g_ids = data["protein_ids"].tolist()

            c_map = {cid: i for i, cid in enumerate(pred_c_ids)}
            g_map = {gid: i for i, gid in enumerate(pred_g_ids)}

            common_c = [c for c in gt_compound_ids if c in c_map]
            common_g = [g for g in gt_gene_ids if g in g_map]

            c_pred_idx = [c_map[c] for c in common_c]
            g_pred_idx = [g_map[g] for g in common_g]
            c_gt_idx = [gt_compound_ids.index(c) for c in common_c]
            g_gt_idx = [gt_gene_ids.index(g) for g in common_g]

            cg_aligned = cg_scores[np.ix_(c_pred_idx, g_pred_idx)]
            cg_gt_aligned = cg_binary[np.ix_(c_gt_idx, g_gt_idx)]

            full_name = f"{method_name} ({score_label})"

            for col in ["protein_family", "go_cellular_component", "go_molecular_function"]:
                r = stratified_auroc(cg_aligned, cg_gt_aligned, common_g, annotations, col, full_name)
                all_stratified.append(r)
                print(f"\n{full_name} — {col}:")
                sorted_strata = sorted(r["strata"].items(), key=lambda x: -x[1]["median_auroc"])
                for stratum, stats in sorted_strata[:15]:
                    print(f"  {stratum:<40} median={stats['median_auroc']:.4f}  n={stats['n_targets']}")

    with open(os.path.join(results_dir, "evaluation", "stratified_analysis.json"), "w") as f:
        json.dump(all_stratified, f, indent=2)

    print("\nStratified analysis complete!")


if __name__ == "__main__":
    results_dir = sys.argv[1]
    gt_dir = sys.argv[2] if len(sys.argv) > 2 else "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"
    data_dir = sys.argv[3] if len(sys.argv) > 3 else "/opt/dlami/nvme/rxrx3_phenomics/data"
    main(results_dir, gt_dir, data_dir)
