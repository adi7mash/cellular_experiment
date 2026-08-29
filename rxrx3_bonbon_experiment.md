# Bonbon Interaction-Prints vs. Boltz-2 Affinity-Prints on Phenomics Ground Truth

## Background

Read this blog for full context on what we are replicating and extending:
https://valencelabs.substack.com/p/i-ran-boltz-2-100-million-times-to

**Summary:** Beaini (Valence Labs / Recursion) built "affinity-prints" — sparse binary vectors of Boltz-2 predicted binding across ~9K proteins for each compound — and showed they correlate with phenomics similarity maps at 71–76% AUROC (known compounds, 6-parameter model) and 53.9% median per-target AUROC. He used 100M co-foldings computed over 12 months on Recursion's BioHive supercomputer.

**Our hypothesis:** Bonbon's codebook representations encode richer per-entity information (mechanism, binding mode, site type) than binary affinity. Concatenating protein + molecule codebook embeddings into "interaction-prints" should correlate more strongly with cellular phenomics than Boltz-2's binary affinity-prints. We can produce these interaction-prints in minutes, not months.

**Important:** This experiment is **evaluation only** — no model training. We compute embeddings from frozen checkpoints and evaluate their correlation with phenomics ground truth.

---

## Repository and Environment

All code for this experiment lives in:
```
https://github.com/adi7mash/cellular_experiment
```

Clone this repo to `~/cellular_experiment/` and work from there. All scripts, notebooks, and analysis code created during this experiment should be committed to this repo.

**Assumed package locations** — these are already installed on the machine in the home directory. Do NOT reinstall or clone them:
- `~/alfajor/` — Alfajor utilities
- `~/mochi/` — Data pipeline framework (tasks, S3 data manager)
- `~/bonbontoken/` — Bonbon model definitions, tokenizers, data utils
- `~/millefeuille/` — Embedding generation entry point (`~/millefeuille/main.py`)

Activate the environment before any execution:
```bash
source ~/output/bin/activate
```

The millefeuille entry point is `~/millefeuille/main.py`. Store this as `$MILLEFEUILLE` in all scripts.

---

## Checkpoint specification

All embedding computations take a checkpoint path as a parameter. The experiment MUST be structured so that swapping the checkpoint reruns only the embedding step (Phase 2), while all data preparation (Phase 1) and phenomics ground truth (Phase 3) remain cached and are never recomputed.

**Default checkpoint:** `s3://output-model-checkpoints/checkpoints/final/e_cheese_five_epoch_dark-snowball-245.ckpt`

The user will specify alternative checkpoints to compare. Store results in checkpoint-namespaced output directories so multiple runs coexist. Ask the user which checkpoint(s) to run before starting Phase 2.

---

## Phase 1: Data Acquisition (run once, cache forever)

All data acquisition and preprocessing scripts go in `~/cellular_experiment/scripts/`. Analysis and evaluation code goes in `~/cellular_experiment/analysis/`. Configuration (checkpoint paths, thresholds) goes in `~/cellular_experiment/config.py`. Commit working code to the repo as you go.

```
~/cellular_experiment/
├── config.py                   # checkpoint paths, thresholds, column names
├── scripts/
│   ├── 01_download_rxrx3.py
│   ├── 02_map_genes.py
│   ├── 03_map_compounds.py
│   ├── 04_build_pairs.py
│   ├── 05_build_ecfp.py
│   ├── 06_build_ground_truth.py
│   └── 07_run_embeddings.sh
├── analysis/
│   ├── norm_diagnostic.py
│   ├── build_interaction_prints.py
│   ├── evaluate.py
│   └── stratified_analysis.py
├── results/                    # gitignored, large files
└── README.md
```

### 1a. Download RxRx3-core from HuggingFace

```bash
pip install huggingface_hub pandas pyarrow scikit-learn rdkit-pypi safe-mol
```

```python
from huggingface_hub import hf_hub_download

metadata_path = hf_hub_download(
    "recursionpharma/rxrx3-core",
    filename="metadata_rxrx3_core.csv",
    repo_type="dataset",
)
embeddings_path = hf_hub_download(
    "recursionpharma/rxrx3-core",
    filename="OpenPhenom_rxrx3_core_embeddings.parquet",
    repo_type="dataset",
)
```

This gives 222,601 wells, 735 labeled CRISPR knockouts, 1,674 compounds at 8 doses, and OpenPhenom-S/16 embeddings per well.

### 1b. Parse metadata and extract identifiers

From `metadata_rxrx3_core.csv`:

- Extract the list of unique gene symbols from CRISPR KO wells (the `well_type` column indicates gene KO vs compound vs control). There should be 735 unique labeled genes.
- Extract the list of unique compound identifiers and their dose levels. There should be 1,674 unique compounds.
- Record the well → perturbation mapping needed for embedding aggregation in Phase 3.
- Inspect the metadata columns carefully and document: what column holds the gene symbol? What column holds the compound identifier? What format is the compound identifier (InChIKey, name, PubChem CID, SMILES)?

### 1c. Map gene symbols → protein sequences

Map each of the 735 gene symbols to its canonical human UniProt protein sequence.

Use the UniProt ID mapping API:
```
POST https://rest.uniprot.org/idmapping/run
  from: Gene_Name
  to: UniProtKB
  taxonId: 9606 (human)
  ids: <comma-separated gene symbols>
```

Then fetch reviewed (Swiss-Prot) entries, taking the canonical isoform. For genes with multiple reviewed entries, take the longest sequence.

Output: a TSV file `gene_to_protein.tsv` with columns: `gene_symbol`, `uniprot_id`, `sequence`

Track and report any genes that fail to map. These are excluded from downstream analysis.

### 1d. Map compound identifiers → SAFE strings

The 1,674 compounds are FDA-approved and commercially available bioactives. Their identifiers in RxRx3-core metadata need to be resolved to SAFE molecular strings for Bonbon encoding.

**Resolution strategy** (try in order):
1. If the metadata contains SMILES directly, convert to SAFE using `safe.encode(smiles)`.
2. If the metadata contains InChIKey or compound names, resolve to SMILES via PubChem PUG REST API (`https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON`), then convert to SAFE.
3. If compound IDs look like internal Recursion identifiers, check whether the EFAAR benchmarking repo or the RxRx3-core HuggingFace card provides a SMILES mapping file.

Convert all resolved SMILES to non-isomeric canonical SMILES first (`rdkit.Chem.MolToSmiles(mol, isomericSmiles=False)`), then to SAFE.

Output: a TSV file `compound_to_safe.tsv` with columns: `compound_id`, `smiles`, `SAFE`

Track and report compounds that fail SAFE conversion. These are excluded from downstream analysis.

### 1e. Build the pair file for millefeuille

Create a single TSV file `rxrx3_pairs.tsv` with columns:

| Protein | Sequence | Molecule | SAFE |
|---------|----------|----------|------|

This file contains the cross product: every (protein, molecule) pair. With 735 proteins × 1,674 molecules = up to 1,230,390 rows. The file must include all four columns even though millefeuille runs protein and molecule sides separately and internally calls `.drop_duplicates()` on each side.

Column names are **case-sensitive**: `Protein`, `Sequence`, `Molecule`, `SAFE`.

### 1f. Build ECFP baseline

Compute ECFP4 fingerprints (Morgan radius 2, 2048 bits) for all 1,674 compounds using RDKit. Precompute the full Tanimoto similarity matrix (1,674 × 1,674). Save as `ecfp_tanimoto.npz`. This is the structural chemistry baseline — Beaini reports ~63–64% AUROC for ECFP.

---

## Phase 2: Compute Bonbon Embeddings (rerun per checkpoint)

This phase uses `~/millefeuille/main.py` directly. The `batch-embeddings` skill documents the full invocation patterns — refer to it for edge cases (OOM recovery, batch size tuning). Model code is in `~/bonbontoken/`, data utils in `~/mochi/`.

**Before starting:** Ask the user which checkpoint(s) to run. Store each checkpoint's results under `/opt/dlami/nvme/rxrx3_phenomics/<checkpoint_label>/`. The embedding run script should be `~/cellular_experiment/scripts/07_run_embeddings.sh`, parameterized by checkpoint path and label.

### Preflight

```bash
ls /opt/dlami/nvme/
source ~/output/bin/activate
MILLEFEUILLE=~/millefeuille/main.py
ls $MILLEFEUILLE  # verify it exists
```

### 2a. Codebook embeddings — molecule side

For each checkpoint:

```bash
python $MILLEFEUILLE \
    --pair-file /opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv \
    --molecule_encoder codebook \
    --pooling none \
    --codebook-model-path <CHECKPOINT_S3_PATH> \
    --codebook-query 2 \
    --output /opt/dlami/nvme/rxrx3_phenomics/<CHECKPOINT_LABEL> \
    --no-protein \
    --molecule-sequence-column SAFE \
    --nan-filter-column SAFE \
    --modality_type safe \
    --codebook-output-types tokens,pooled_attention
```

### 2b. Codebook embeddings — protein side

```bash
python $MILLEFEUILLE \
    --pair-file /opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv \
    --protein_encoder codebook \
    --pooling none \
    --codebook-model-path <CHECKPOINT_S3_PATH> \
    --codebook-query 1 \
    --output /opt/dlami/nvme/rxrx3_phenomics/<CHECKPOINT_LABEL> \
    --no-molecule \
    --modality_type protein \
    --codebook-output-types tokens,pooled_attention
```

### 2c. Contrastive projection embeddings — molecule side

This gives us the L2-normed projected tower embeddings for computing ITC-like binding scores (the binary interaction-print baseline).

```bash
python $MILLEFEUILLE \
    --pair-file /opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv \
    --molecule_encoder molecule \
    --pooling mean \
    --molecule-model-path <CHECKPOINT_S3_PATH> \
    --modality_type modality_2 \
    --output /opt/dlami/nvme/rxrx3_phenomics/<CHECKPOINT_LABEL> \
    --no-protein \
    --molecule-sequence-column SAFE \
    --nan-filter-column SAFE \
    --use-safe-tokenizer \
    --use-contrastive-projection
```

### 2d. Contrastive projection embeddings — protein side

```bash
python $MILLEFEUILLE \
    --pair-file /opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv \
    --protein_encoder protein \
    --pooling mean \
    --protein-model-path <CHECKPOINT_S3_PATH> \
    --modality_type modality_1 \
    --output /opt/dlami/nvme/rxrx3_phenomics/<CHECKPOINT_LABEL> \
    --no-molecule \
    --use-contrastive-projection
```

### 2e. Verify outputs

After all four runs complete for a checkpoint, verify:

```bash
# Codebook outputs — should have subdirectories tokens/ and pooled_attention/
ls /opt/dlami/nvme/rxrx3_phenomics/<LABEL>/*_molecule_codebook_*/tokens/ | wc -l  # expect ~1674
ls /opt/dlami/nvme/rxrx3_phenomics/<LABEL>/*_protein_codebook_*/tokens/ | wc -l   # expect ~735
ls /opt/dlami/nvme/rxrx3_phenomics/<LABEL>/*_molecule_codebook_*/pooled_attention/ | wc -l
ls /opt/dlami/nvme/rxrx3_phenomics/<LABEL>/*_protein_codebook_*/pooled_attention/ | wc -l

# Contrastive projection outputs — flat .pt files
ls /opt/dlami/nvme/rxrx3_phenomics/<LABEL>/*_molecule_molecule_*/ | wc -l  # expect ~1674
ls /opt/dlami/nvme/rxrx3_phenomics/<LABEL>/*_protein_protein_*/ | wc -l   # expect ~735
```

Report: "✓ Completed <label>: <n_proteins> proteins, <n_molecules> molecules (codebook + projection)"

### 2f. Note on sequence_attention

Do NOT request `sequence_attention` in this experiment. It produces large per-residue tensors that are not needed for interaction-print evaluation. If results from Phase 5 are inconclusive, sequence_attention may be added in a follow-up run **only with user approval**.

---

## Phase 3: Build Phenomics Ground Truth (run once, cache forever)

### 3a. Aggregate OpenPhenom embeddings to per-perturbation level

The RxRx3-core OpenPhenom embeddings are per-well. Follow the standard RxRx3 preprocessing pipeline (documented in the EFAAR benchmarking repo at https://github.com/recursionpharma/EFAAR_benchmarking):

1. Center and scale each embedding relative to experiment-level unperturbed control wells (this is the batch correction step).
2. Average across replicates: wells → plates → experiments → per-perturbation.
3. For compounds at multiple doses: aggregate across doses to get one vector per compound (mean across dose levels), AND keep per-dose vectors for optional dose-response analysis.
4. Output: one vector per geneKO (735 vectors) and one vector per compound (1,674 vectors).

Save as `phenomics_gene_embeddings.npz` and `phenomics_compound_embeddings.npz`.

### 3b. Build compound–compound phenomics similarity matrix

Cosine similarity between all 1,674 compound pairs from their aggregated OpenPhenom embeddings. Shape: 1,674 × 1,674. This is the ground truth for the Virtual Cell test.

Save as `phenomics_compound_compound_sim.npz`.

### 3c. Build compound–gene phenomics similarity matrix

Cosine similarity between each compound and each geneKO embedding. Shape: 1,674 × 735. This is the per-target ground truth.

Save as `phenomics_compound_gene_sim.npz`.

### 3d. Binarize ground truth at evaluation thresholds

For compound–compound (Virtual Cell test):
- Binarize at similarity thresholds > 0.4 and > 0.6, matching Beaini's thresholds.

For compound–gene (per-target test):
- For each target (column), mark the top 1% of compound–gene pairs as TRUE, matching Beaini's methodology.

Save as `ground_truth_binary.npz` with separate arrays for each threshold.

---

## Phase 4: Build Interaction-Prints from Bonbon Embeddings

### 4a. Load all embeddings into indexed matrices

For each checkpoint:

**Codebook tokens:** Load all `.pt` files from the `tokens/` subdirectories. Build matrices:
- `protein_tokens[i]` = codebook token embedding for protein i
- `molecule_tokens[j]` = codebook token embedding for molecule j
- Record the shape of each tensor — this determines downstream operations.

**Codebook pooled_attention:** Load from `pooled_attention/` subdirectories:
- `protein_pooled[i]` = sparsemax vector for protein i
- `molecule_pooled[j]` = sparsemax vector for molecule j

**Contrastive projections:** Load from the projection output directories:
- `protein_proj[i]` = projected tower embedding for protein i
- `molecule_proj[j]` = projected tower embedding for molecule j

Index all matrices by entity ID to match the phenomics ground truth ordering.

### 4b. L2 normalization diagnostic (MUST run before any similarity computation)

Do NOT blindly L2-normalize any embedding type. Instead, run a diagnostic on every embedding matrix to determine its normalization state empirically. This is a control — the results determine how similarity is computed downstream.

For each of the four embedding matrices (`protein_tokens`, `molecule_tokens`, `protein_proj`, `molecule_proj`) and the two pooled_attention matrices:

```python
import torch

def norm_diagnostic(embeddings: torch.Tensor, name: str):
    """Characterize the L2 norm distribution of an embedding matrix.
    
    Args:
        embeddings: (N, D) matrix of embeddings
        name: label for reporting
    """
    norms = torch.norm(embeddings, dim=-1)
    report = {
        "name": name,
        "shape": tuple(embeddings.shape),
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std().item(),
        "norm_min": norms.min().item(),
        "norm_max": norms.max().item(),
        "norm_median": norms.median().item(),
        "is_unit_normed": bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)),
        "fraction_near_unit": (torch.abs(norms - 1.0) < 1e-3).float().mean().item(),
    }
    return report
```

Run on every embedding matrix and save the full report as `norm_diagnostics.json` in the checkpoint results directory.

**Decision logic based on diagnostics:**

- **If `is_unit_normed` is True** (all norms within 1e-4 of 1.0): embeddings are already L2-normalized. Use raw dot product as cosine similarity. Do NOT re-normalize.
- **If norms show low variance** (`norm_std / norm_mean < 0.05`): embeddings are approximately uniform in norm but not unit-normed. The norm carries little information. L2-normalizing is safe and we should compare both (raw dot product vs. normalized dot product) to verify they produce equivalent rankings.
- **If norms show meaningful variance** (`norm_std / norm_mean >= 0.05`): the norm itself may carry signal (e.g., confidence, binding strength). Run **all downstream evaluations in three conditions:**
  1. **Raw dot product** (preserves norm magnitude as part of the score)
  2. **Cosine similarity** (L2-normalize, then dot product — removes norm, tests direction only)
  3. **Norm as separate feature** (report correlation between embedding norm and phenomics signal strength)

This applies independently to each embedding type — codebook tokens, pooled_attention, and contrastive projections may have different normalization profiles.

**Expected behavior based on code inspection:**
- Codebook `tokens`: the code applies `F.normalize(tokens, dim=-1)` explicitly, so these SHOULD be unit-normed. The diagnostic will confirm.
- Codebook `pooled_attention`: these are sparsemax outputs (non-negative, sum to 1 per row by definition of sparsemax), so they are L1-normalized but NOT L2-normalized. Their L2 norms will vary — sparser distributions have higher L2 norms. This variance may be informative (sparser = more specific binding signature).
- Contrastive projections: unknown. The projection head may or may not include normalization. The diagnostic determines this.

Log the full diagnostic report and include it in the final summary. If any embedding type shows unexpected normalization behavior (e.g., codebook tokens are NOT unit-normed despite the code), flag this to the user as a potential checkpoint or code issue before proceeding.

### 4c. Binary interaction-prints (Beaini analogue)

Two variants, with normalization determined by 4b diagnostics:

**Variant 1 — Projection dot product (ITC-like):**
- Apply normalization per 4b diagnostic results for `protein_proj` and `molecule_proj`.
- If norms have meaningful variance, compute all three conditions (raw dot, cosine, norm-as-feature).
- Compute: `score_matrix = protein_proj @ molecule_proj.T` → shape (735, 1674)
- Sweep thresholds on the score to produce binary bind/don't-bind matrix.

**Variant 2 — Codebook token dot product:**
- Apply normalization per 4b diagnostic results for `protein_tokens` and `molecule_tokens`.
- Compute: `score_matrix = protein_tokens @ molecule_tokens.T` → shape (735, 1674)
- Sweep thresholds.

For each variant, produce a binary interaction-print matrix at the optimal threshold (determined by AUROC on the compound–gene test).

### 4d. Codebook interaction-prints (novel test)

For each (compound, protein) pair, construct a dense interaction representation from `pooled_attention` vectors.

**Important note on pooled_attention norms:** Sparsemax outputs are non-negative and L1-normalized (entries sum to 1) but NOT L2-normalized. Their L2 norm varies inversely with sparsity — a highly sparse vector (few active codes) has a higher L2 norm than a diffuse one. This variance may carry signal: sparser codebook activations may indicate more specific molecular recognition. The 4b diagnostic will quantify this. Do NOT L2-normalize pooled_attention vectors without first checking whether norm variance correlates with phenomics signal.

**Concat:** protein pooled_attention ⊕ molecule pooled_attention → 16,384-dim per pair. Each compound's interaction-print is its full row across all 735 targets (735 × 16,384 flattened or used as-is for compound–compound similarity).

**Element-wise product:** protein pooled_attention ⊙ molecule pooled_attention → 8,192-dim per pair. Captures code co-activation — which codebook entries are active for both the protein and molecule simultaneously. Sparser and more directly relational.

Run both variants.

**Computing compound–compound similarity from interaction-prints:**
- For each compound j, its interaction-print is a matrix of shape (735, D) where D is 16,384 (concat) or 8,192 (product).
- Flatten to a single vector per compound: (735 × D,)
- Compute cosine similarity between all compound pairs → predicted compound–compound similarity matrix.
- If 4b diagnostic shows meaningful norm variance in pooled_attention: also compute with un-normalized dot product and report both.

### 4e. Codebook token interaction-prints

Same as 4d but using `tokens` instead of `pooled_attention`. The `tokens` tensor captures the weighted codebook representations (richer per entry, lower dimensional than pooled_attention). Normalization condition determined by 4b diagnostic (expected: already unit-normed). Test both concat and element-wise product.

---

## Phase 5: Evaluation

### 5a. Compound–compound similarity prediction (Virtual Cell test)

For each method, compute AUROC for predicting phenomics compound–compound similarity above threshold:

| Method | Description |
|--------|-------------|
| ECFP Tanimoto | Structural chemistry baseline |
| Projection dot product (binary) | Beaini-analogue from contrastive tower |
| Codebook token dot product (binary) | Beaini-analogue from codebook space |
| Codebook pooled_attention concat | Interaction-print: 16,384-dim per pair |
| Codebook pooled_attention product | Interaction-print: 8,192-dim per pair |
| Codebook tokens concat | Interaction-print from token embeddings |
| Codebook tokens product | Interaction-print from token embeddings |

Report AUROC at binarization thresholds 0.4 and 0.6 (matching Beaini).

**Targets to beat:** Beaini's 71–76% AUROC (Boltz-2 + 6-param model), ECFP at 63–64%.

### 5b. Per-target compound–gene AUROC

For each of the 735 genes and each scoring method:
- Take the column of scores (all 1,674 compounds scored against this gene).
- Compare against the compound–gene phenomics ground truth (top 1% binarization from Phase 3d).
- Compute AUROC.

Report: median AUROC across targets, mean, 25th/75th percentiles, and full distribution.

**Target to beat:** Beaini's 53.9% median across all targets.

### 5c. Stratified analysis

Annotate the 735 genes with:
- **Protein family** (kinase, GPCR, protease, nuclear receptor, etc.) from UniProt keywords.
- **GO cellular component** annotations (nucleus, mitochondrion, plasma membrane, etc.).
- **GO molecular function** annotations (ATP binding, RNA binding, etc.).
- **In-training flag**: whether the protein appears in Bonbon's training data (check against UniProt IDs in the training set).

Report per-stratum median AUROC. Key comparisons against Beaini's findings:
- Does Bonbon show less protein-family-dependent variance? (Beaini found kinases best, GPCRs worst, but with huge overlap)
- Does Bonbon degrade for mitochondrial targets? (Beaini found mitochondrion/mitochondrial membrane as the worst GO cellular components)
- Does GO cellular component still dominate prediction of which targets work? (If not, this suggests Bonbon's representations encode functional context that Boltz-2's structural predictions miss)

### 5d. Cross-checkpoint comparison

For each checkpoint tested, report all metrics from 5a–5c in a unified comparison table. Identify which checkpoint produces the strongest phenomics correlation and whether scaling (if different-sized checkpoints are tested) improves the signal.

---

## Phase 6: Extended Analysis (run only if Phase 5 shows positive results)

### 6a. MoA enrichment test

For compounds in RxRx3-core with known mechanism of action (many FDA-approved drugs):
- Among compounds predicted to bind the same target, do those with more similar codebook interaction-prints have more similar phenomics profiles?
- Specifically: cluster compounds by their codebook interaction-print similarity on each target, and check whether clusters correspond to known MoA categories (agonist vs antagonist vs inhibitor).

### 6b. Scale to JUMP-CP

Repeat the full experiment on the JUMP Cell Painting dataset (7,976 labeled genes, different phenomics embeddings from CellProfiler and community models). 7,976 proteins + 1,674 molecules = 9,650 unique entities. Use EFAAR benchmarking repo for JUMP-CP preprocessing. This is an independent replication on a different phenomics ground truth.

### 6c. Sequence_attention deep dive (requires user approval)

If specific targets show strong codebook signal but weak binary score signal, re-run embeddings with `--codebook-output-types tokens,pooled_attention,sequence_attention` to investigate whether per-residue attention maps reveal binding site localization patterns that correlate with phenomics. **Do NOT run this without explicit user approval** — sequence_attention is large and expensive.

---

## Output Structure

```
/opt/dlami/nvme/rxrx3_phenomics/
├── data/                              # Phase 1 outputs (cached, shared)
│   ├── metadata_rxrx3_core.csv
│   ├── OpenPhenom_rxrx3_core_embeddings.parquet
│   ├── gene_to_protein.tsv            # gene_symbol | uniprot_id | sequence
│   ├── compound_to_safe.tsv           # compound_id | smiles | SAFE
│   ├── rxrx3_pairs.tsv               # Protein | Sequence | Molecule | SAFE
│   ├── unmapped_genes.txt            # genes that failed UniProt mapping
│   └── unmapped_compounds.txt        # compounds that failed SAFE conversion
│
├── ground_truth/                      # Phase 3 outputs (cached, shared)
│   ├── phenomics_gene_embeddings.npz
│   ├── phenomics_compound_embeddings.npz
│   ├── phenomics_compound_compound_sim.npz
│   ├── phenomics_compound_gene_sim.npz
│   ├── ground_truth_binary.npz
│   ├── ecfp_tanimoto.npz
│   └── gene_annotations.tsv          # protein family, GO terms, in-training flag
│
├── <checkpoint_label>/                # Phase 2 outputs (per checkpoint)
│   ├── rxrx3_pairs_molecule_codebook_<run_name>/
│   │   ├── tokens/                    # one .pt per molecule
│   │   └── pooled_attention/          # one .pt per molecule
│   ├── rxrx3_pairs_protein_codebook_<run_name>/
│   │   ├── tokens/                    # one .pt per protein
│   │   └── pooled_attention/          # one .pt per protein
│   ├── rxrx3_pairs_molecule_molecule_<run_name>/   # projection embeddings
│   │   └── *.pt                       # one .pt per molecule
│   └── rxrx3_pairs_protein_protein_<run_name>/     # projection embeddings
│       └── *.pt                       # one .pt per protein
│
├── results/                           # Phase 4-5 outputs (per checkpoint)
│   └── <checkpoint_label>/
│       ├── interaction_prints/
│       │   ├── binary_projection.npz
│       │   ├── binary_codebook_tokens.npz
│       │   ├── pooled_attention_concat.npz   # or memory-mapped if large
│       │   ├── pooled_attention_product.npz
│       │   ├── tokens_concat.npz
│       │   └── tokens_product.npz
│       ├── evaluation/
│       │   ├── compound_compound_auroc.json
│       │   ├── per_target_auroc.json
│       │   ├── stratified_analysis.json
│       │   └── full_auroc_distributions.npz
│       └── summary.md
│
└── comparison/
    └── cross_checkpoint_summary.md
```

---

## Execution Order Summary

1. **Phase 1** (once): Download data → map genes → map compounds → build pair file → build ECFP baseline
2. **Phase 3** (once): Aggregate phenomics embeddings → build similarity matrices → binarize
3. **Ask user for checkpoint(s)**
4. **Phase 2** (per checkpoint): Run 4 millefeuille commands (codebook mol, codebook prot, projection mol, projection prot)
5. **Phase 4** (per checkpoint): Load embeddings → build interaction-print matrices
6. **Phase 5** (per checkpoint): Evaluate all methods → report
7. **Cross-checkpoint comparison** if multiple checkpoints were run
8. **Phase 6** only if user requests after reviewing Phase 5 results

---

## Key Numbers for Sanity Checking

- 735 unique genes → ~735 unique proteins (minus any mapping failures)
- 1,674 unique compounds → ~1,674 unique SAFE strings (minus conversion failures)
- Pair file: up to 735 × 1,674 = 1,230,390 rows
- Unique entities for millefeuille: 735 + 1,674 = 2,409
- At ~2,000 entities/sec: ~1.2 seconds for codebook embedding (but model loading dominates)
- Phenomics similarity matrices: 1,674 × 1,674 (compound–compound) and 1,674 × 735 (compound–gene)
- Beaini's baselines: 71–76% AUROC (compound–compound), 53.9% median (per-target), 63–64% (ECFP)
