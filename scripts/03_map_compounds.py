"""Map RxRx3-core compounds (SMILES) to SAFE strings."""
import pandas as pd
from rdkit import Chem
import safe

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"

df = pd.read_csv(f"{DATA_DIR}/metadata_rxrx3_core.csv", low_memory=False)

compounds = df[df["perturbation_type"] == "COMPOUND"]
compound_smiles = compounds.groupby("treatment")["SMILES"].first().reset_index()
compound_smiles = compound_smiles[compound_smiles["SMILES"].notna() & (compound_smiles["SMILES"] != "")]
compound_smiles = compound_smiles[~compound_smiles["treatment"].isin(["EMPTY_control", "DMSO"])]
print(f"Unique compounds with SMILES: {len(compound_smiles)}")

rows = []
unmapped = []

for _, row in compound_smiles.iterrows():
    compound_id = row["treatment"]
    smiles = row["SMILES"]

    try:
        clean_smiles = smiles.split(" |")[0] if " |" in smiles else smiles
        mol = Chem.MolFromSmiles(clean_smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            unmapped.append((compound_id, smiles, "RDKit parse failed"))
            continue

        canonical = Chem.MolToSmiles(mol, isomericSmiles=False)

        try:
            safe_str = safe.encode(canonical)
            if not safe_str:
                safe_str = canonical
        except Exception:
            safe_str = canonical

        rows.append({
            "compound_id": compound_id,
            "smiles": canonical,
            "SAFE": safe_str,
        })
    except Exception as e:
        unmapped.append((compound_id, smiles, str(e)))

out_df = pd.DataFrame(rows)
out_df.to_csv(f"{DATA_DIR}/compound_to_safe.tsv", sep="\t", index=False)
print(f"\nConverted: {len(rows)} compounds")
print(f"Failed: {len(unmapped)} compounds")

safe_encoded = sum(1 for r in rows if "." in r["SAFE"])
print(f"SAFE-fragmented: {safe_encoded}, SMILES-fallback: {len(rows) - safe_encoded}")

with open(f"{DATA_DIR}/unmapped_compounds.txt", "w") as f:
    for cid, smi, reason in unmapped:
        f.write(f"{cid}\t{smi}\t{reason}\n")

if unmapped:
    print(f"\nFailed compounds:")
    for cid, smi, reason in unmapped[:10]:
        print(f"  {cid}: {reason}")
