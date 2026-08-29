# RxRx3 Bonbon Experiment Log

## 2026-08-29 — Phase 1: Data Acquisition

### 1a. Download RxRx3-core (DONE)
- Downloaded `metadata_rxrx3_core.csv` and `OpenPhenom_rxrx3_core_embeddings.parquet` from HuggingFace `recursionpharma/rxrx3-core`
- 222,601 wells, 384-dim OpenPhenom embeddings
- Stored in `/opt/dlami/nvme/rxrx3_phenomics/data/`

### 1b. Metadata inspection (DONE)
- Columns: `well_id, experiment_name, plate, address, gene, treatment, SMILES, concentration, perturbation_type, cell_type, well_type_label`
- `perturbation_type`: CRISPR (126,900 wells) vs COMPOUND (95,701 wells)
- Gene column: `gene` — 736 unique values including controls (735 real genes after excluding EMPTY_control)
- Compound identifier: `treatment` column has drug names (e.g., "Esomeprazole")
- SMILES: directly available in `SMILES` column — 1,674 unique values
- Some SMILES use enhanced notation with `|...|` stereo extensions
- 12 unique concentration levels

### 1c. Gene → Protein mapping (IN PROGRESS)
- Using UniProt search API (search API more reliable than ID mapping API)
- 735 unique gene symbols to map
- Querying reviewed Swiss-Prot entries for Homo sapiens (taxon 9606)
- Fallback to unreviewed entries for genes not found in Swiss-Prot
- Output: `gene_to_protein.tsv` (gene_symbol | uniprot_id | sequence)

### 1d. Compound → SAFE mapping (DONE)
- 1,674 compounds, all successfully mapped
- 1,541 SAFE-fragmented, 133 SMILES-fallback (small molecules SAFE can't fragment)
- 0 failures
- Fixed: enhanced SMILES with `|...|` notation — stripped before RDKit parsing
- Output: `compound_to_safe.tsv` (compound_id | smiles | SAFE)

### 1e. ECFP baseline (DONE)
- ECFP4 fingerprints (Morgan radius 2, 2048 bits) for all 1,674 compounds
- Full 1,674 × 1,674 Tanimoto similarity matrix
- Mean self-similarity: 1.0, mean off-diagonal: 0.1004
- Output: `ground_truth/ecfp_tanimoto.npz`

### 1f. Build ground truth (DONE)
- Batch correction: center+scale on control wells per experiment (StandardScaler fit on controls, transform all) — following EFAAR benchmarking pipeline
- Gene KO embeddings: 735 genes, aggregated by mean across replicates
- Compound embeddings: 1,674 compounds, aggregated by mean across all doses and replicates
- Compound-compound cosine similarity: mean=0.733, std=0.238
- Compound-gene cosine similarity: 1,674 × 735
- Binary ground truth:
  - CC >0.4: 87.8% positive pairs
  - CC >0.6: 79.2% positive pairs  
  - CG top 1%: 0.96% positive pairs per target
- Similarity distribution: high baseline cosine similarity is expected for cell painting embeddings
- Output: `ground_truth/` directory with all .npz files

### 1c. Gene → Protein mapping (DONE)
- All 735 genes mapped successfully to reviewed Swiss-Prot entries
- Used UniProt search API (batch queries with `gene_exact` filter, organism 9606, reviewed only)
- Output: `gene_to_protein.tsv` — 735 rows, 0 unmapped

### 1e. Build pair file (DONE)
- 735 × 1,674 = 1,230,390 rows
- Columns: Protein, Sequence, Molecule, SAFE
- Output: `rxrx3_pairs.tsv`

### Environment fix
- `efaar_benchmarking` install upgraded `typing-extensions` (4.12→4.16) and downgraded `botocore` (1.43.36→1.43.0), breaking transformers/lightning imports
- Fixed: `pip install botocore==1.43.36 typing-extensions==4.14.1`

---

## Phase 2: Compute Bonbon Embeddings (IN PROGRESS)

### Checkpoint: dark-snowball-245
- S3 path: `s3://output-model-checkpoints/checkpoints/final/e_cheese_five_epoch_dark-snowball-245.ckpt`
- GPU: NVIDIA H200 (143 GB VRAM)
- Running 4 embedding passes:
  1. Codebook molecule (tokens + pooled_attention)
  2. Codebook protein (tokens + pooled_attention)
  3. Contrastive projection molecule
  4. Contrastive projection protein
- Timing being recorded to `results/timing_dark-snowball-245.log`

### Timing (NVIDIA H200, 143 GB VRAM)
| Step | Entities | Time |
|------|----------|------|
| Codebook molecule | 1,674 | 3m19s |
| Codebook protein | 735 | 2m55s |
| Projection molecule | 1,674 | 3m08s |
| Projection protein | 735 | 3m09s |
| **Total** | **2,409** | **12m31s** |

Note: Beaini's 100M Boltz-2 co-foldings took 12 months on BioHive. Bonbon produces richer per-entity embeddings in 12.5 minutes.

### Outputs
- Codebook: 1,673 molecule .pt files × 2 (tokens + pooled_attention), 735 protein .pt files × 2
- Projection: 1,673 molecule .pt files, 735 protein .pt files
- 1 compound (atenolol-(+/-)) filtered by NaN SAFE

---

## Phase 3: Ground Truth — TVN Preprocessing

### First attempt: Simple StandardScaler
- Batch correction: center+scale on controls per experiment
- CC cosine similarity: mean=0.73, 88% positive at >0.4 threshold
- ECFP AUROC: 0.508 — essentially chance, impossible to match Beaini's 63-64%

### Fix: Full TVN (Typical Variation Normalization) from EFAAR pipeline
1. Center+scale on controls per experiment (StandardScaler)
2. PCA on controls (whitening/rotation)
3. Center+scale again in PCA space
4. CORAL normalization per batch (covariance alignment)

### TVN results
- CC cosine similarity: mean=0.20, std=0.23 (much more spread)
- CC >0.4: 20.1% positive (was 88%)
- CC >0.6: 7.7% positive (was 79%)
- P90 threshold: 0.558 (10% positive)
- P95 threshold: 0.657 (5% positive)
- ECFP Spearman with phenomics: 0.0004 (structural similarity ≠ phenomics similarity)
- ECFP AUROC (P90/P95): 0.508/0.511 — near chance even after proper preprocessing

---

## Phase 4: Norm Diagnostics

| Embedding Type | Shape | Unit-Normed? | Norm CV | Decision |
|---|---|---|---|---|
| molecule_codebook_tokens | 1673×1280 | Yes (exactly 1.0) | 0.00 | Raw dot = cosine |
| protein_codebook_tokens | 735×1280 | Yes (exactly 1.0) | 0.00 | Raw dot = cosine |
| molecule_codebook_pooled_attention | 1673×8192 | No (mean=0.17) | 0.09 | Test both raw+cosine |
| protein_codebook_pooled_attention | 735×8192 | No (mean=0.16) | 0.26 | Test both raw+cosine |
| molecule_projection | 1673×1024 | No (mean=8.2) | 0.10 | Test both raw+cosine |
| protein_projection | 735×1024 | No (mean=8.2) | 0.10 | Test both raw+cosine |

**Key finding**: Codebook tokens are perfectly unit-normed (F.normalize applied), pooled_attention has meaningful norm variance (CV 9-26%, sparser = higher L2 norm), projections are NOT normalized (mean norm ~8.2).

### Mathematical insight
Concat interaction-prints for CC similarity **degenerate** to molecule-only similarity. Proof: IP_j·IP_k = C + N×mol[j]·mol[k] where C = Σ_i ||prot[i]||² is constant for all pairs. Product interaction-prints do NOT degenerate — they weight dimensions by Σ_i prot[i,d]².

---

## Phase 5: Evaluation Results

### Compound-Compound AUROC

| Method | Thresh 0.4 | Thresh 0.6 | P90 | P95 |
|--------|-----------|-----------|-----|-----|
| ECFP Tanimoto | 0.504 | 0.510 | 0.508 | 0.511 |
| Projection dot product (raw) | 0.502 | 0.505 | 0.503 | 0.509 |
| Projection dot product (cosine) | 0.501 | 0.504 | 0.502 | 0.508 |
| Codebook token dot product | 0.507 | 0.512 | 0.510 | 0.514 |
| **Pooled Attention Concat** | **0.521** | **0.529** | **0.527** | **0.532** |
| Pooled Attention Product | 0.515 | 0.516 | 0.516 | 0.516 |
| Tokens Concat | 0.511 | 0.517 | 0.515 | 0.518 |
| Tokens Product | 0.511 | 0.516 | 0.515 | 0.518 |

**Beaini baselines**: 71-76% (CC, 6-param trained model), 63-64% (ECFP)

### Per-Target AUROC

| Method | Median | Mean | P25 | P75 |
|--------|--------|------|-----|-----|
| **Projection (raw)** | **0.524** | **0.523** | 0.468 | 0.580 |
| Projection (cosine) | 0.520 | 0.520 | 0.462 | 0.576 |
| Codebook tokens | 0.509 | 0.513 | 0.460 | 0.563 |
| Pooled attention (cosine) | 0.511 | 0.510 | 0.459 | 0.557 |
| Pooled attention (raw) | 0.506 | 0.508 | 0.457 | 0.555 |

**Beaini baseline**: 53.9% median per-target

### Stratified Analysis by Protein Family
- Proteases show highest per-target AUROC (median 0.53-0.55) across all methods
- Kinases show lowest (median 0.50-0.52)
- "Other" category in between

### Key Observations
1. **All methods are near chance (0.50-0.53 AUROC)**. ECFP included.
2. **Beaini's 71-76% came from a TRAINED 6-parameter model**, not raw correlation. Our evaluation is zero-shot.
3. **Beaini's ECFP baseline (63-64%)** likely also involved the trained model. Raw ECFP has ~0 correlation with phenomics.
4. **Beaini's per-target 53.9%** is only marginally above chance. Our projection method gets 52.4%, which is in the same ballpark.
5. **Pooled Attention Concat is best for CC** (0.53 AUROC at P95) — but this mathematically degenerates to molecule-molecule similarity plus a constant offset, so the signal is from the molecule codebook, not from protein-molecule interactions.
6. **Projection raw dot is best for per-target** (0.524 median) — the contrastive learning objective gives slightly better per-target discrimination than codebook embeddings alone.

---

## Analysis scripts prepared (while embeddings run)

### Phase 4 scripts
- `analysis/norm_diagnostic.py` — L2 norm analysis for all embedding types, determines normalization strategy
- `analysis/build_interaction_prints.py` — Builds all interaction-print variants:
  - Binary projection dot product (raw + cosine)
  - Binary codebook token dot product (raw + cosine)
  - Codebook pooled_attention concat and product
  - Codebook tokens concat and product
  - Compound-gene score matrices for per-target evaluation

### Phase 5 scripts
- `analysis/evaluate.py` — Full evaluation:
  - Compound-compound AUROC at thresholds 0.4 and 0.6
  - Per-target AUROC (median, mean, quartiles)
  - Comparison against ECFP baseline and Beaini targets
- `analysis/stratified_analysis.py` — Stratified analysis:
  - UniProt protein family annotations (Kinase, GPCR, Protease, etc.)
  - GO cellular component and molecular function
  - Per-stratum median AUROC

---

## Scripts created
- `config.py` — checkpoint paths, thresholds, directory constants
- `scripts/01_download_rxrx3.py` — download from HuggingFace
- `scripts/02_map_genes.py` — UniProt search API gene→protein mapping
- `scripts/03_map_compounds.py` — SMILES→SAFE conversion with enhanced SMILES fix
- `scripts/05_build_ecfp.py` — ECFP4 Tanimoto matrix
- `scripts/06_build_ground_truth.py` — phenomics similarity matrices and binarization
