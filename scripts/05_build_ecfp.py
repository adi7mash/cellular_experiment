"""Compute ECFP4 fingerprints and Tanimoto similarity matrix for all compounds."""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

DATA_DIR = "/opt/dlami/nvme/rxrx3_phenomics/data"
GT_DIR = "/opt/dlami/nvme/rxrx3_phenomics/ground_truth"

compounds = pd.read_csv(f"{DATA_DIR}/compound_to_safe.tsv", sep="\t")
print(f"Compounds: {len(compounds)}")

fps = []
valid_ids = []
for _, row in compounds.iterrows():
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None:
        print(f"Warning: couldn't parse {row['compound_id']}")
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fps.append(fp)
    valid_ids.append(row["compound_id"])

print(f"Valid fingerprints: {len(fps)}")

n = len(fps)
tanimoto = np.zeros((n, n), dtype=np.float32)
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    tanimoto[i] = sims

np.savez_compressed(
    f"{GT_DIR}/ecfp_tanimoto.npz",
    tanimoto=tanimoto,
    compound_ids=np.array(valid_ids),
)
print(f"Saved ECFP Tanimoto matrix: {tanimoto.shape}")
print(f"Mean self-similarity: {np.diag(tanimoto).mean():.4f}")
print(f"Mean off-diagonal: {tanimoto[np.triu_indices(n, k=1)].mean():.4f}")
