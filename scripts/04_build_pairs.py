"""Build the cross-product pair file for millefeuille embedding runs."""
import pandas as pd

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"

genes = pd.read_csv(f"{DATA_DIR}/gene_to_protein.tsv", sep="\t")
compounds = pd.read_csv(f"{DATA_DIR}/compound_to_safe.tsv", sep="\t")

print(f"Proteins: {len(genes)}")
print(f"Molecules: {len(compounds)}")

genes["_key"] = 1
compounds["_key"] = 1
pairs = genes[["gene_symbol", "uniprot_id", "sequence", "_key"]].merge(
    compounds[["compound_id", "SAFE", "_key"]], on="_key"
).drop("_key", axis=1)

pairs = pairs.rename(columns={
    "gene_symbol": "Protein",
    "sequence": "Sequence",
    "compound_id": "Molecule",
})
pairs = pairs[["Protein", "Sequence", "Molecule", "SAFE"]]

print(f"Pair file: {len(pairs)} rows (expected {len(genes) * len(compounds)})")

pairs.to_csv(f"{DATA_DIR}/rxrx3_pairs.tsv", sep="\t", index=False)
print(f"Saved to {DATA_DIR}/rxrx3_pairs.tsv")
