# Supplementary Material: Frozen Molecular Interaction Embeddings Encode Cellular Biology

## A. Dataset Details

### A.1 RxRx3-core

- **Source**: Recursion Pharmaceuticals (Chandrasekaran et al., 2024)
- **Compounds**: 1,674 (150 with known gene-target associations)
- **Gene knockouts**: 735 CRISPR knockouts in HUVEC cells
- **Wells**: 222,601 total
- **Embedding dim**: 384 (OpenPhenom, pre-extracted CNN features)
- **Cell line**: HUVEC (human umbilical vein endothelial cells)
- **Perturbation type**: Small-molecule compounds + CRISPR knockouts

### A.2 JUMP-CP (cpg0016)

- **Source**: JUMP-Cell Painting consortium (Chandrasekaran et al., 2023)
- **Gene knockouts**: 7,976 CRISPR knockouts
- **Plates**: 148
- **Wells**: 50,032 (after quality control)
- **Feature dim**: 3,671 (CellProfiler morphological features after QC)
- **Cell line**: A549 (human lung carcinoma) and U2OS (human osteosarcoma)
- **Perturbation type**: CRISPR gene knockouts

### A.3 Bonbon Checkpoint

- **Checkpoint**: dark-snowball-245
- **Parameters**: 1.6B
- **Training data**: Protein-ligand sequence pairs (self-supervised)
- **No cellular data** in training: no microscopy, no Cell Painting, no transcriptomics, no phenomics

## B. Preprocessing Details

### B.1 RxRx3 OpenPhenom Baseline (TVN Pipeline)

Typical Variation Normalization (TVN) on RxRx3 OpenPhenom embeddings:

1. **Center-scale on controls**: StandardScaler fit on DMSO/untreated wells per experiment, applied to all wells
2. **PCA on controls**: PCA fit on control wells, transform all wells (retain all 384 components)
3. **Center-scale again**: Second StandardScaler on PCA-transformed controls
4. **CORAL batch correction**: `scipy.linalg.fractional_matrix_power` alignment across batches
5. **Mean aggregation**: Per-gene and per-compound mean pooling of well-level embeddings

### B.2 JUMP-CP CellProfiler Baseline (TVN Pipeline)

1. **Feature QC**: Remove features with >50% missing values, near-zero variance, or >0.95 pairwise correlation
2. **Intensity filtering**: Remove wells with abnormal staining intensity (z-score > 3 on any channel)
3. **Cell count filtering**: Remove wells with cell count outside [50, 5000]
4. **TVN normalization**: Same 4-step pipeline as above, fit on DMSO controls per batch
5. **Mean aggregation**: Per-gene mean pooling across wells and plates
6. **Final dimensions**: 3,671 features after QC, 7,976 genes

### B.3 Bonbon Protein Embeddings

- **Projector**: 1,024-dim contrastive projection, mean-pooled across residues
- **Codebook tokens**: 1,280-dim sparse codebook activations, mean-pooled
- **Pooled attention**: 8,192-dim attention-weighted codebook activations
- **Fusion**: L2-normalized concatenation of projector + tokens + pooled attention (11,296-dim total)
- **Proteome coverage**: 20,337 human proteins from UniProt reviewed set
- **JUMP-CP coverage**: 7,802/7,976 genes (97.8%) matched by gene symbol

## C. Per-Target AUROC Distributions

### C.1 Zero-Shot CG Score Methods

**Table S1.** Per-target zero-shot AUROC distribution across 735 gene targets.

| Method | Median | Mean | Std | P5 | P25 | P75 | P95 | Min | Max |
|--------|--------|------|-----|----|-----|-----|-----|-----|-----|
| Projection (raw) | .524 | .523 | .083 | .381 | .468 | .580 | .661 | .285 | .760 |
| Projection (cosine) | .520 | .520 | .084 | .378 | .462 | .576 | .655 | .292 | .757 |
| Codebook tokens (raw) | .509 | .513 | .080 | .386 | .460 | .563 | .643 | .289 | .807 |
| Codebook tokens (cosine) | .509 | .513 | .080 | .386 | .460 | .563 | .643 | .289 | .807 |
| Pooled attention (raw) | .506 | .508 | .078 | .385 | .457 | .555 | .643 | .263 | .751 |
| Pooled attention (cosine) | .511 | .510 | .078 | .390 | .459 | .557 | .646 | .255 | .749 |

### C.2 Trained Per-Target Models

**Table S2.** Per-target AUROC with trained MLP models (target-level CV, 257 evaluable targets).

| Input representation | Dim | Median AUROC |
|---------------------|-----|-------------|
| Molecule codebook tokens | 1,280 | **89.9%** |
| Molecule + protein tokens concat | 2,560 | 84.4% |
| Combined (tokens + PA) | 10,752 | 80.6% |
| Fusion CLS norm | 1,024 | 51.7% |
| Pooled attention product | 8,192 | 58.8% |

The molecule codebook tokens alone are the best single representation, confirming that the compound-side codebook is the primary carrier of phenomics-relevant signal. Adding protein tokens (concat to 2,560-dim) reduces performance by 5.5 points, suggesting the protein-side signal is complementary but introduces noise at the per-target level.

## D. CC AUROC Architecture Search

**Table S3.** Full architecture comparison (pair-level CV @0.6, 57 features). All MLPs trained with Adam (lr=1e-3), cosine annealing, 120 epochs, 30% dropout.

| Architecture | Variant | AUROC @0.6 | Std |
|---|---|---|---|
| **Wider MLP** | [2048, 1024] | **81.1%** | 1.70% |
| Focal loss | [1024,512,256,128] γ=3.0 | 81.0% | 1.69% |
| Original MLP | [1024,512,256,128] | 80.9% | 1.96% |
| Wider MLP | [4096, 2048] | 80.9% | 1.56% |
| Focal loss | [1024,512,256,128] γ=1.0 | 80.7% | 2.04% |
| Wider MLP | [2048, 1024, 512] | 80.6% | 1.81% |
| Focal loss | [1024,512,256,128] γ=2.0 | 80.4% | 1.50% |
| Multi-task | w=0.3 | 79.9% | 1.77% |
| Multi-task | w=0.5 | 79.7% | 1.45% |
| Multi-task | w=1.0 | 79.1% | 1.51% |
| Residual + GELU | 1024×2 blocks | 78.4% | 1.57% |
| Residual + GELU | 1024×3 blocks | 78.3% | 1.24% |
| Residual + GELU | 2048×3 blocks | 76.6% | 1.35% |
| FT-Transformer | d=128, h=4, L=3 | 72.0% | 1.35% |
| FT-Transformer | d=64, h=4, L=6 | 70.6% | 1.96% |
| FT-Transformer | d=64, h=4, L=3 | 69.7% | 2.48% |

Deeper architectures (residual blocks, FT-Transformer) consistently underperform. With 57 input features, the inductive biases of skip connections and self-attention find no exploitable structure. The ensemble's 82.0% advantage comes from model diversity (tree + neural + regression), not architecture capacity.

## E. Model Comparison at Multiple Scales

**Table S4.** Full model comparison at @0.4 and @0.6 thresholds (pair-level CV, 57 features).

| Model | @0.4 | @0.6 |
|-------|------|------|
| XGB (3000, d7, lr=0.03) | 72.3% | 73.8% |
| XGB (5000, d7, lr=0.02) | 72.3% | 73.9% |
| XGB (5000, d10, lr=0.01) | 72.4% | 73.6% |
| MLP [512,256,128] | 73.3% | 76.0% |
| MLP [512,256,128,64] | 73.3% | 75.7% |
| MLP [1024,512,256] | 74.0% | 75.9% |
| MLP [1024,512,256,128] | 74.2% | 75.9% |
| Logistic regression | 72.5% | 74.0% |
| Ensemble (XGB + MLP avg) | 74.9% | 76.5% |
| **Ensemble (all 3 avg)** | **75.2%** | **77.5%** |
| Ensemble (stacked) | 74.8% | 76.5% |
| Seed ensemble | 72.9% | 74.1% |

Note: These are early-stage results with 42 baseline features. The final models with 57 features (including signatures and fusion CLS) achieve 81.1% @0.6 (Table 1 in main paper).

## F. Compound-Level Cross-Validation Details

**Table S5.** Compound-level CV results with and without signature leakage.

| Configuration | @0.4 | @0.6 |
|--------------|------|------|
| Per-fold signature, XGBoost | 66.6% | 68.1% |
| Per-fold signature, regression | 65.4% | 67.0% |
| Per-fold signature, ensemble | 65.2% | 67.0% |
| Global signature (leaky), XGBoost | 68.1% | 70.3% |
| Global signature (leaky), MLP | 68.9% | 71.4% |
| Global signature (leaky), regression | 67.6% | 69.0% |
| Global signature (leaky), ensemble | 68.3% | 70.1% |

The ~2–3 point gap between per-fold and global signature quantifies the inflation from learning the phenomics signature on all data before cross-validation.

## G. Ablation: Fusion Cross-Attention Features

**Table S6.** Effect of adding fusion CLS features to the baseline (XGBoost, without signatures).

| Feature set | Pair CV @0.6 | Compound CV @0.6 |
|------------|-------------|-----------------|
| Baseline (42 features) | 72.7% | 58.5% |
| + Fusion CLS (47 features) | **76.0% (+3.3)** | **61.7% (+3.2)** |

Fusion CLS per-target performance (as a standalone CG score):
- Fusion norm median AUROC: 51.0% (257 targets)
- Fusion norm with PPI propagation: 51.7%

The CLS token encodes interaction quality and mode, not target specificity — it adds complementary signal to the entity-level codebook embeddings but cannot replace them for per-target discrimination.

## H. Ablation: Transcriptomic Information

**Table S7.** Effect of gene expression weighting (HUVEC RNA-seq from Human Protein Atlas).

| Expression weighting | Best single per-target | CC AUROC @0.6 |
|---------------------|----------------------|---------------|
| None (baseline) | 58.8% | 69.7% |
| Log expression | 57.1% | — |
| Binary (expressed/not) | 57.4% | — |
| Sqrt expression | 56.8% | — |
| Sigmoid expression | 58.0% | — |
| Rank expression | 58.5% | — |
| PPI + expression | 58.1% | — |
| All combined | — | 71.2% |

Expression weighting provides marginal CC AUROC improvement (+1.5 pts) but no per-target benefit. The frozen embeddings already capture most of the signal that transcriptomics provides.

## I. EFAAR Known-Relationship Benchmark: All Embedding Types

### I.1 735-Gene RxRx3 Subset

**Table S8.** Known-relationship recall@0.05/0.95, 735-gene RxRx3 subset.

| Method | CORUM | HuMAP | Reactome | SIGNOR | StringDB |
|--------|-------|-------|----------|--------|----------|
| OpenPhenom (Cell Painting, TVN) | .288 | .330 | .135 | .129 | .263 |
| Bonbon projector (1,024-dim) | .432 | .473 | .264 | .160 | .398 |
| Bonbon cb_tokens (1,280-dim) | .439 | .475 | .269 | .220 | .418 |
| Bonbon cb_pa (8,192-dim) | .371 | .410 | .239 | .196 | .371 |
| **Bonbon fusion (11,296-dim)** | **.451** | **.509** | **.271** | **.200** | **.447** |

### I.2 Full JUMP-CP (7,976 Genes)

**Table S9.** Known-relationship recall@0.05/0.95, JUMP-CP cpg0016.

| Method | CORUM | HuMAP | Reactome | SIGNOR | StringDB |
|--------|-------|-------|----------|--------|----------|
| CellProfiler (3,671-dim, TVN) | .154 | .123 | .095 | .102 | .126 |
| Published MAE-G/8 (Kraus 2025) | .264 | .215 | .165 | — | .235 |
| Bonbon projector (1,024-dim) | .316 | .298 | .241 | .168 | .304 |
| Bonbon cb_tokens (1,280-dim) | .270 | .311 | .223 | .183 | .347 |
| Bonbon cb_pa (8,192-dim) | .242 | .299 | .221 | .221 | .341 |
| **Bonbon fusion (11,296-dim)** | **.316** | **.338** | **.234** | **.193** | **.373** |

### I.3 Full Proteome (20,337 Proteins)

**Table S10.** Known-relationship recall@0.05/0.95, full human proteome.

| Method | CORUM | HuMAP | Reactome | SIGNOR | StringDB |
|--------|-------|-------|----------|--------|----------|
| Bonbon projector (1,024-dim) | .260 | .236 | .200 | .124 | .281 |
| Bonbon cb_tokens (1,280-dim) | .317 | .297 | .204 | .169 | .344 |
| Bonbon cb_pa (8,192-dim) | .302 | .282 | .228 | .237 | .338 |
| Bonbon fusion (11,296-dim) | .329 | .311 | .208 | .151 | .356 |

### I.4 Published JUMP-CP Baselines (ViTally Consistent, Kraus et al. 2025)

**Table S11.** Published baselines from ViTally Consistent (ICML 2025), ~7,976 genes.

| Method | CORUM | HuMAP | Reactome | StringDB |
|--------|-------|-------|----------|----------|
| CellProfiler | .219 | .184 | .131 | .191 |
| MAE-G/8 (best) | .264 | .215 | .165 | .235 |

Note: Our CellProfiler baseline (.154 CORUM) differs from the published value (.219) likely due to differences in QC filtering thresholds. Both are well below the learned representations.

## J. Active Moiety Attention Analysis

### J.1 Global Statistics

**Table S12.** Attention ratio distribution across 735 gene targets.

| Statistic | Value |
|-----------|-------|
| Mean attention ratio (active/inactive) | 1.083 |
| Median attention ratio | 1.074 |
| Std | 0.061 |
| Min | 0.933 |
| Max | 1.280 |
| P5 | 0.998 |
| P25 | 1.039 |
| P75 | 1.125 |
| P95 | 1.189 |

### J.2 Significance

| Threshold | N targets | % |
|-----------|-----------|---|
| p < 0.05 | 654 | 89.0% |
| p < 0.01 | 619 | 84.2% |
| p < 0.001 | 580 | 78.9% |
| p < 6.8e-05 (Bonferroni) | 534 | 72.7% |

### J.3 Top 50 Most Significant Targets

**Table S13.** Top 50 gene targets by significance of codebook attention difference between active and inactive compounds.

| Rank | Gene | Att. Ratio | p-value | Top Compound | Top Fragment (SAFE) |
|------|------|-----------|---------|-------------|-------------------|
| 1 | TMEM11 | 1.247 | 1.3e-28 | Mesna | O=S(=O)([O-])CCS |
| 2 | COPZ1 | 1.260 | 1.5e-28 | pantothenic-acid | N12 |
| 3 | DMAC2L | 1.195 | 3.4e-28 | Choline | C1CO |
| 4 | ATP5F1C | 1.239 | 4.1e-28 | Mesna | O=S(=O)([O-])CCS |
| 5 | OGDH | 1.239 | 4.4e-28 | Xylose | O=CC(O)C(O)C(O)CO |
| 6 | MAGOHB | 1.275 | 4.6e-28 | Urea | NC(N)=O |
| 7 | CA7 | 1.205 | 5.9e-28 | Acetazolamide | N23 |
| 8 | EXOSC8 | 1.236 | 6.1e-28 | pantothenic-acid | N12 |
| 9 | EXOSC5 | 1.242 | 6.1e-28 | pantothenic-acid | N12 |
| 10 | CENPM | 1.246 | 6.5e-28 | Gefitinib | CO5 |
| 11 | SEC22C | 1.223 | 6.5e-28 | Choline | C1CO |
| 12 | POLD2 | 1.242 | 6.5e-28 | Urea | NC(N)=O |
| 13 | CAPZA3 | 1.280 | 6.8e-28 | fosfosal | c13ccccc12 |
| 14 | RPL7A | 1.221 | 1.1e-27 | Butanedioic acid | O=C(O)CCC(=O)O |
| 15 | EIF3E | 1.179 | 1.3e-27 | NS-1643 | N24 |
| 16 | ATP6V1F | 1.272 | 2.3e-27 | Butanedioic acid | O=C(O)CCC(=O)O |
| 17 | ACADS | 1.254 | 2.4e-27 | Uracil | O=c1cc[nH]c(=O)[nH]1 |
| 18 | MAGOH | 1.241 | 3.2e-27 | Urea | NC(N)=O |
| 19 | CA1 | 1.180 | 5.3e-27 | Acetazolamide | N23 |
| 20 | ALDH4A1 | 1.229 | 5.8e-27 | Butanedioic acid | O=C(O)CCC(=O)O |
| 21 | CA5A | 1.169 | 6.7e-27 | Brinzolamide | N4C1CN5S... |
| 22 | CA12 | 1.200 | 7.7e-27 | dorzolamide | N3C1CC(C)S... |
| 23 | POLR2L | 1.207 | 1.0e-26 | Mesna | O=S(=O)([O-])CCS |
| 24 | CENPC | 1.205 | 1.4e-26 | Mesalamine | Nc1ccc(O)c2c1 |
| 25 | TATDN2 | 1.217 | 1.4e-26 | Vidarabine | C3O |
| 26 | CA9 | 1.178 | 1.9e-26 | dorzolamide | N3C1CC(C)S... |
| 27 | EIF3H | 1.179 | 2.0e-26 | Quercetin | O=c1c(O)c3oc2... |
| 28 | SLC12A3 | 1.156 | 2.4e-26 | Furosemide | C36 |
| 29 | COPE | 1.180 | 3.3e-26 | Irbesartan | c15nnn[nH]1 |
| 30 | INTS9 | 1.193 | 3.3e-26 | Adrucil | O=c1[nH]cc(F)c(=O)[nH]1 |
| 31 | CA2 | 1.155 | 4.4e-26 | Acetazolamide | N23 |
| 32 | RPS16 | 1.189 | 5.0e-26 | Pyrithioxin | C35 |
| 33 | RPL5 | 1.186 | 5.6e-26 | fosfosal | c13ccccc12 |
| 34 | POLR2C | 1.237 | 6.2e-26 | Urea | NC(N)=O |
| 35 | CTSZ | 1.242 | 6.4e-26 | Bestatin | c14ccccc1 |
| 36 | ACADL | 1.227 | 7.8e-26 | Uracil | O=c1cc[nH]c(=O)[nH]1 |
| 37 | SLC12A1 | 1.158 | 9.6e-26 | Furosemide | C36 |
| 38 | RPS11 | 1.190 | 9.6e-26 | Choline | C1CO |
| 39 | ABCC8 | 1.126 | 1.5e-25 | Glipizide | N69 |
| 40 | LCN12 | 1.166 | 1.6e-25 | ascorbic-acid | O=C1OC2C(O)C1=O |
| 41 | RPL23A | 1.162 | 1.7e-25 | Quercetin | O=c1c(O)c3oc2... |
| 42 | NUP85 | 1.210 | 1.9e-25 | Cysteamine | NCCS |
| 43 | CA14 | 1.160 | 2.2e-25 | dorzolamide | N3C1CC(C)S... |
| 44 | POLR2I | 1.207 | 2.2e-25 | Nicotinamide | c12cccnc1 |
| 45 | PIPOX | 1.229 | 2.3e-25 | L-proline | O=C2O |
| 46 | TYR | 1.170 | 2.5e-25 | phloroglucin | Oc1cc(O)cc(O)c1 |
| 47 | RPL30 | 1.226 | 3.4e-25 | Butanedioic acid | O=C(O)CCC(=O)O |
| 48 | TTR | 1.172 | 6.0e-25 | Liothyronine | c14ccc(O)c(I)c1 |
| 49 | POLR2G | 1.203 | 6.4e-25 | Pemetrexed | C46=O |
| 50 | EXOSC4 | 1.187 | 9.3e-25 | Urea | NC(N)=O |

### J.4 Pharmacologically Validated Examples

| Target | Function | Top Compound | Known Relationship |
|--------|----------|-------------|-------------------|
| CA7, CA1, CA5A, CA12, CA9, CA2, CA14 | Carbonic anhydrases | Acetazolamide, dorzolamide, Brinzolamide | All are FDA-approved CA inhibitors; sulfonamide pharmacophore identified |
| ABCC8 | Sulfonylurea receptor | Glipizide | FDA-approved sulfonylurea antidiabetic |
| CTSZ | Cathepsin Z (protease) | Bestatin | Known aminopeptidase/protease inhibitor |
| SLC12A3, SLC12A1 | Na-K-Cl cotransporters | Furosemide | Loop diuretic targeting NKCC/NCC transporters |
| TTR | Transthyretin | Liothyronine | Thyroid hormone; TTR is its transport protein |
| TYR | Tyrosinase | phloroglucin | Polyphenol with known tyrosinase inhibitory activity |

Seven carbonic anhydrase isoforms (CA1, CA2, CA5A, CA7, CA9, CA12, CA14) all independently identify sulfonamide-containing CA inhibitors as their top compounds, with the sulfonamide zinc-binding warhead consistently highlighted in the attention maps.

## K. Compute Resources

| Operation | Hardware | Time |
|-----------|----------|------|
| Bonbon embedding extraction (1,674 compounds × 735 proteins, 3 types) | 1× NVIDIA H200 | 12.5 min |
| Fusion CLS extraction (1,230,390 pairs) | 1× NVIDIA H200 | ~70 min |
| Human proteome embeddings (20,337 proteins, 3 types) | 1× NVIDIA H200 | ~15 min |
| JUMP-CP CellProfiler download + preprocessing (148 plates) | CPU | 44 sec |
| EFAAR benchmark evaluation (per embedding type) | CPU | ~5 sec |
| Active moiety attention analysis (735 targets) | CPU | ~3 min |
| XGBoost training (5-fold CV) | 1× GPU | ~5 min |
| MLP training (5-fold CV, 120 epochs) | 1× GPU | ~20 min |
| **Total pipeline** | **1× H200** | **< 2 hours** |
| Boltz-2 (Beaini et al.) | BioHive-1 supercomputer | 12 months |
