"""Extract fusion CLS tokens from dark-snowball-245 for all RxRx3 protein-molecule pairs.

The fusion CLS token is the output of the unmasked bidirectional cross-attention
encoder — a single 1024-dim vector per protein-molecule pair that encodes the
interaction between the two sequences through 16 transformer layers.

Optimization: proteins are encoded once and cached; the fusion encoder runs per
(protein, molecule-batch) to avoid re-encoding the same protein 1,674 times.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, str(Path.home() / "bonbontoken"))
from bonbontoken.data.dataset import _preprocess_sequence_by_type
from bonbontoken.models.fusion.fusion_models import (
    BonbonGlobalEmbeddingFusionModelUnmaskedCrossAttention,
)
from bonbontoken.tokenizer import MODALITY_SETTINGS, SAFETokenizer
from bonbontoken.utils import resolve_tokenizer_path

CHECKPOINT = "/opt/dlami/nvme/ckpts/e_cheese_five_epoch_dark-snowball-245.ckpt"
PAIRS_TSV = "/opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv"
OUTPUT_DIR = Path("/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/fusion_cls")

PROTEIN_MAX_LEN = 2048
MOLECULE_MAX_LEN = 256
FUSION_BATCH_SIZE = 32
DEVICE = "cuda"


def load_model():
    print("Loading model from checkpoint...")
    t0 = time.time()
    model = BonbonGlobalEmbeddingFusionModelUnmaskedCrossAttention.load_from_checkpoint(
        CHECKPOINT, strict=False
    )
    model = model.to(DEVICE)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Model loaded in {time.time() - t0:.1f}s")
    return model


def load_tokenizers():
    prot_path = resolve_tokenizer_path(
        MODALITY_SETTINGS["protein"]["tokenizer_path"]
    )
    prot_tokenizer = AutoTokenizer.from_pretrained(prot_path)

    safe_path = resolve_tokenizer_path(
        MODALITY_SETTINGS["safe"]["tokenizer_path"]
    )
    safe_tok = SAFETokenizer.from_pretrained(safe_path).get_pretrained()
    safe_tok.add_tokens(["<", ">"])
    return prot_tokenizer, safe_tok


def load_pairs():
    df = pd.read_csv(PAIRS_TSV, sep="\t")
    proteins = df.groupby("Protein").first()["Sequence"].to_dict()
    molecules = {}
    for _, row in df.drop_duplicates("Molecule").iterrows():
        molecules[row["Molecule"]] = row["SAFE"]
    molecule_order = df["Molecule"].unique().tolist()
    protein_order = df["Protein"].unique().tolist()
    return proteins, molecules, protein_order, molecule_order


def tokenize_protein(seq, tokenizer):
    preprocessed = _preprocess_sequence_by_type(seq, "protein")
    enc = tokenizer(
        preprocessed,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=PROTEIN_MAX_LEN,
    )
    return enc["input_ids"], enc["attention_mask"]


def tokenize_molecules_batch(safe_list, tokenizer):
    enc = tokenizer(
        safe_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MOLECULE_MAX_LEN,
    )
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def encode_protein(model, prot_ids, prot_mask):
    prot_ids = prot_ids.to(DEVICE)
    prot_mask = prot_mask.to(DEVICE)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        features = model.modality_1_encoder(
            input_ids=prot_ids, attention_mask=prot_mask
        )
    return features.cpu(), prot_mask.cpu()


@torch.no_grad()
def extract_fusion_cls_batch(model, prot_features, prot_mask, mol_ids, mol_mask):
    batch_size = mol_ids.shape[0]
    prot_features_exp = prot_features.expand(batch_size, -1, -1).to(DEVICE)
    prot_mask_exp = prot_mask.expand(batch_size, -1).to(DEVICE)
    mol_ids = mol_ids.to(DEVICE)
    mol_mask = mol_mask.to(DEVICE)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        mol_features = model.modality_2_encoder(
            input_ids=mol_ids, attention_mask=mol_mask
        )
        cls_output, _, _, _, _ = model.unmasked_embedding_fusion_encoder(
            modality_1_features=prot_features_exp,
            modality_1_mask=prot_mask_exp,
            modality_2_features=mol_features,
            modality_2_mask=mol_mask,
        )
    return cls_output.float().cpu()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    prot_tokenizer, safe_tokenizer = load_tokenizers()

    print("Loading pairs...")
    proteins, molecules, protein_order, molecule_order = load_pairs()
    print(f"  {len(protein_order)} proteins, {len(molecule_order)} molecules")
    print(f"  Total pairs: {len(protein_order) * len(molecule_order):,}")

    # Filter molecules with valid SAFE
    valid_mols = [(name, molecules[name]) for name in molecule_order if pd.notna(molecules.get(name))]
    valid_mol_names = [m[0] for m in valid_mols]
    valid_mol_safes = [m[1] for m in valid_mols]
    print(f"  Valid molecules (have SAFE): {len(valid_mols)}")

    json.dump(valid_mol_names, open(OUTPUT_DIR / "molecule_index.json", "w"))

    # Phase 1: pre-encode all proteins
    print("\n=== Phase 1: Encoding proteins ===")
    protein_cache = {}
    t0 = time.time()
    for i, pname in enumerate(protein_order):
        prot_ids, prot_mask = tokenize_protein(proteins[pname], prot_tokenizer)
        features, mask = encode_protein(model, prot_ids, prot_mask)
        protein_cache[pname] = (features, mask)
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Encoded {i + 1}/{len(protein_order)} proteins "
                  f"({time.time() - t0:.1f}s)")
    print(f"  All proteins encoded in {time.time() - t0:.1f}s")

    # Phase 2: extract fusion CLS for all pairs
    print("\n=== Phase 2: Extracting fusion CLS tokens ===")
    total_pairs = 0
    t0 = time.time()

    already_done = set()
    for f in OUTPUT_DIR.glob("*.pt"):
        already_done.add(f.stem)

    for pi, pname in enumerate(protein_order):
        if pname in already_done:
            total_pairs += len(valid_mol_names)
            if (pi + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{pi + 1}/{len(protein_order)}] {pname}: skipped (already done) "
                      f"[{total_pairs:,} pairs, {elapsed:.0f}s]")
            continue

        prot_features, prot_mask = protein_cache[pname]
        cls_tokens = []

        for batch_start in range(0, len(valid_mol_safes), FUSION_BATCH_SIZE):
            batch_safes = valid_mol_safes[batch_start:batch_start + FUSION_BATCH_SIZE]
            mol_ids, mol_mask = tokenize_molecules_batch(batch_safes, safe_tokenizer)
            cls = extract_fusion_cls_batch(
                model, prot_features, prot_mask, mol_ids, mol_mask
            )
            cls_tokens.append(cls)
            total_pairs += cls.shape[0]

        cls_matrix = torch.cat(cls_tokens, dim=0)  # [num_molecules, hidden_dim]
        torch.save(cls_matrix, OUTPUT_DIR / f"{pname}.pt")

        if (pi + 1) % 10 == 0 or pi == 0:
            elapsed = time.time() - t0
            pairs_per_sec = total_pairs / elapsed if elapsed > 0 else 0
            eta = (len(protein_order) - pi - 1) * len(valid_mol_names) / pairs_per_sec if pairs_per_sec > 0 else 0
            print(f"  [{pi + 1}/{len(protein_order)}] {pname}: "
                  f"{cls_matrix.shape} | "
                  f"{total_pairs:,} pairs, {pairs_per_sec:.0f} pairs/s, "
                  f"ETA {eta / 60:.1f}min")

    elapsed = time.time() - t0
    print(f"\n=== Done ===")
    print(f"  Total pairs: {total_pairs:,}")
    print(f"  Total time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Throughput: {total_pairs / elapsed:.0f} pairs/s")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  CLS dim: {cls_matrix.shape[1]}")
    print(f"  Files: {len(list(OUTPUT_DIR.glob('*.pt')))}")


if __name__ == "__main__":
    main()
