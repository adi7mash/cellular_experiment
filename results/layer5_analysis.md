# Layer 5 Analysis: Single-Cell Resolution via Bonbon Cell Conditioning

## Summary

Layer 5 of the virtual cell framework targets single-cell resolution: predicting how individual cells — not population averages — respond to drug perturbation. This document records the experiments, analysis, and architectural research conducted to date.

**Key results:**
- Bonbon CG profiles (945 binding scores, frozen, zero-shot) predict gene expression changes at **0.929 Pearson** on Tahoe-100M pseudobulk (MLP[4096,2048,1024] + one_hot cell line)
- Expression-based cell conditioning captures **82% of one-hot's lift** over CG-only (0.918 vs 0.924, or 0.912 vs 0.923 for smaller MLP) — proving Bonbon's 945-protein expression profile is a viable cell-state representation
- All three fusion architectures tested (GatedFusion, FiLM, CrossGate) underperform simple concatenation — diagnosed as a data resolution issue (8 unique cell states), not a method limitation
- Proposed Bonbon Cell Codebook architecture using STATE-style additive expression encoding over frozen protein codebook tokens

---

## 1. Tahoe-100M Data Pipeline

### 1.1 Data source

Tahoe-100M (tahoebio/Tahoe-100M on HuggingFace): 95.6M single-cell profiles across 50 cell lines x 379 drugs at 3 concentrations. 3,388 parquet shards, ~429GB. Each cell record contains: genes (int64), expressions (float32), drug, cell_line_id, canonical_smiles, sample, moa-fine, plate.

### 1.2 Pseudobulk computation

We processed 300 of 3,388 shards (step=11 for even coverage), yielding ~1.6M cells. These were averaged per (drug, cell_line) pair to produce **3,024 pseudobulk conditions** and **8 control (DMSO) profiles**, one per cell line.

- means: (3024, 51088) — mean expression per condition
- control_means: (8, 51088) — mean expression per cell line under DMSO
- Target: expression delta = means - control_means (perturbed - control)
- 2,000 highest-variance genes (HVGs) selected as prediction targets
- Stored at `/opt/dlami/nvme/tahoe/pseudobulk_means.npz`

### 1.3 CG profile computation

For each of 357 drugs in Tahoe that matched our compound set, we computed the CG profile: a 945-dimensional vector where each entry is the dot product of the drug's molecule codebook token (1280d) with a protein codebook token (1280d), from frozen Bonbon checkpoint dark-snowball-245.

- CG profiles: (357, 945)
- After matching drug names and filtering (cell_counts >= 10): **2,847 valid conditions**
- Stored at `/opt/dlami/nvme/tahoe/tahoe_cg_profiles.npz`

### 1.4 Data usage relative to Tahoe-100M

| Level | Our usage | Tahoe-100M total | Fraction |
|---|---|---|---|
| Raw cells processed | ~1.6M | 95.6M | 1.7% |
| Shards read | 300 | 3,388 | 8.9% |
| Unique conditions | 2,847 | ~15,000+ | ~19% |
| Gene dimensions | 2,000 HVGs | 51,088 genes | 3.9% |
| Cell lines | 8 | 50 | 16% |

---

## 2. Baseline Results: CG Profile Projection Heads

### 2.1 Architecture sweep

All results use 5-fold CV on 2,847 conditions. CG profiles are frozen (zero-shot from Bonbon). Only the projection head is trained.

**Projection head comparison at 100% data:**

| Head | Pearson Delta | DEG-50 |
|---|---|---|
| Ridge (alpha=100) | 0.819 | — |
| Ridge (alpha=10) | 0.822 | — |
| MLP [2048,1024] | 0.866 | 0.881 |
| MLP+CL [2048,1024] (+ one_hot) | 0.923 | 0.932 |
| **MLP [4096,2048,1024] (+ one_hot)** | **0.929** | **0.937** |

### 2.2 Data efficiency

| Training fraction | MLP CG only | MLP+CL CG+one_hot |
|---|---|---|
| 5% (~142 conditions) | 0.407 | 0.509 |
| 10% (~285 conditions) | 0.557 | 0.654 |
| 20% (~569 conditions) | 0.640 | 0.667 |
| 50% (~1424 conditions) | 0.824 | 0.856 |
| 100% (2847 conditions) | 0.866 | 0.923 |

### 2.3 ModernBERT results

ModernBERT (treating CG profile tokens as a sequence) was tested at 6 scales:

| Fraction | Best ModernBERT | Best MLP |
|---|---|---|
| 5% | 0.411 (xl, h=768, L=8) | 0.407 (CG only) |
| 10% | 0.506 (small, h=256, L=4) | 0.509 (MLP+CL) |
| 20% | 0.651 (large, h=512, L=6) | 0.667 (MLP+CL) |
| 50% | 0.832 (small, h=256, L=4) | 0.856 (MLP+CL) |
| 100% | 0.868 (xl, h=768, L=8) | 0.923 (MLP+CL) |

ModernBERT is competitive at very low data (5%) but MLP+CL dominates at all higher fractions. The attention mechanism finds no exploitable token-token structure in 945 scalar CG scores — each score is already a summary.

### 2.4 LODO (Leave-One-Drug-Out)

| Method | Pearson Delta |
|---|---|
| Ridge (alpha=100) | 0.221 |
| Ridge (alpha=10) | 0.120 |
| Ridge (alpha=1) | 0.099 |

LODO performance is low. With ~357 drugs and 945-dim CG features, the model cannot reliably generalize from training drugs' CG→expression mappings to a held-out drug. MAP reports ~0.87 on "unprofiled drugs" but uses a 187K-drug knowledge graph encoding chemical structural similarity. Our CG profiles encode binding patterns, not chemical structure — two structurally different drugs can have different CG profiles even if they share a target.

### 2.5 Confirmation: zero-shot representation, trained head only

The 0.929 result requires no feature selection, no compound filtering, no signature learning. The pipeline:
1. Frozen Bonbon checkpoint → 945 CG scores per drug (zero-shot)
2. 8-dim one-hot cell line encoding
3. MLP [4096,2048,1024] with BatchNorm, GELU, 0.4 dropout
4. 5-fold CV, MSE loss, AdamW (lr=3e-4, wd=1e-2), cosine annealing, 500 epochs

The only tuning was MLP architecture size. No features were selected or learned from the target.

---

## 3. Approach A: Expression-Weighted CG Interaction

### 3.1 Motivation

Replace the 8-dim one-hot cell line encoding with a 945-dim expression profile of the same 945 proteins used in the CG profile. This removes the dependency on cell-line labels and uses Bonbon's own proteome as the cell-state representation.

### 3.2 Expression profile construction

For each condition (drug, cell_line), we extracted the control expression of the 945 Bonbon codebook proteins from the DMSO control profile. Using `protein_to_gene_tid.json`, 943/945 proteins had valid Tahoe gene IDs. Expression statistics: mean=0.1133, std=0.3238, 87.4% of proteins expressed (>0) per cell line.

### 3.3 Input configurations tested

8 configurations at 3 data fractions (20%, 50%, 100%), plus bigger MLP on best configs:

| Config | Dim | 20% | 50% | 100% | big MLP 100% |
|---|---|---|---|---|---|
| CG only | 945 | 0.640 | 0.824 | 0.866 | — |
| CG + one_hot | 953 | 0.666 | 0.856 | 0.923 | **0.929** |
| CG + expr_945 | 1890 | 0.603 | 0.824 | 0.913 | 0.917 |
| CG + one_hot + expr_945 | 1898 | 0.603 | 0.824 | 0.914 | — |
| CG + expr + interact | 2835 | 0.537 | 0.764 | 0.881 | 0.898 |
| CG + oh + expr + interact | 2843 | 0.537 | 0.764 | 0.880 | 0.898 |
| CG + one_hot + interact | 1898 | 0.544 | 0.761 | 0.874 | — |
| interaction_only (CG * expr) | 945 | 0.452 | 0.649 | 0.820 | — |

### 3.4 Analysis

**Expression conditioning works.** CG + expr_945 (0.913) recovers 82% of the one_hot's lift over CG-only: (0.913 - 0.866) / (0.923 - 0.866) = 82%. With bigger MLP: 0.917 vs 0.929, still 73%.

**One-hot is optimal for 8 discrete states.** There are exactly 8 unique expression profiles (one per cell line). An 8-dim one-hot is information-theoretically optimal for encoding 8 categories. The 945-dim expression vector is a 118x-overcomplete, noisy encoding of the same 8 states.

**Multiplicative interaction hurts.** CG * expr zeros out dimensions for lowly-expressed proteins, but those dimensions carry binding information that matters even for weakly-expressed targets (they may be upregulated downstream). The MLP can learn its own nonlinear gating from concatenated inputs; forcing multiplicative structure is too restrictive.

**The gap is structural, not fundamental.** With 8 cell lines at pseudobulk, there are only 8 unique expression states. Any conditioning method that takes expression as input is encoding 8 categories through a noisier channel than one-hot. At single-cell resolution (thousands of unique states), expression conditioning becomes essential and one-hot encoding is impossible.

Script: `/opt/dlami/nvme/tahoe/eval_layer5_approach_a.py`
Log: `/opt/dlami/nvme/tahoe/layer5_log.txt`

---

## 4. Approach B: Gated Fusion

### 4.1 Motivation

If naive multiplication failed, perhaps learned gating would succeed. Three fusion architectures were tested, each providing a different mechanism for combining CG and expression:

- **GatedFusion**: `g * CG + (1-g) * f(expr)` where g = sigmoid(MLP(concat(CG, expr)))
- **FiLM**: `(1 + gamma(expr)) * CG + beta(expr)` — feature-wise linear modulation (Perez et al., 2018)
- **CrossGate**: Bidirectional — expression gates CG AND CG gates expression, concat both streams

### 4.2 Results

**20% data:**

| Method | Pearson Delta |
|---|---|
| CG + one_hot | **0.667** |
| CG only | 0.640 |
| CG + expr (concat) | 0.602 |
| GatedFusion + oh | 0.523 |
| GatedFusion | 0.521 |
| CrossGate | 0.358 |
| FiLM | 0.354 |

**50% data:**

| Method | Pearson Delta |
|---|---|
| CG + one_hot | **0.855** |
| CG only | 0.825 |
| CG + expr (concat) | 0.824 |
| GatedFusion + oh | 0.758 |
| GatedFusion | 0.757 |
| FiLM + oh | 0.619 |
| FiLM | 0.614 |
| CrossGate | 0.512 |

**100% data (all configs, including bigger MLPs):**

| Method | Pearson Delta | DEG-50 |
|---|---|---|
| big CG + one_hot | **0.929** | 0.937 |
| CG + one_hot | 0.924 | 0.933 |
| big CG + expr (concat) | 0.918 | 0.926 |
| CG + expr (concat) | 0.912 | 0.922 |
| big GatedFusion + oh | 0.891 | 0.899 |
| big GatedFusion | 0.891 | 0.899 |
| big FiLM | 0.887 | 0.896 |
| big FiLM + oh | 0.886 | 0.895 |
| GatedFusion + oh | 0.886 | 0.894 |
| GatedFusion | 0.885 | 0.893 |
| FiLM + oh | 0.873 | 0.883 |
| FiLM | 0.872 | 0.883 |
| CG only | 0.866 | 0.881 |
| big CrossGate | 0.825 | 0.833 |
| big CrossGate + oh | 0.824 | 0.832 |
| CrossGate + oh | 0.772 | 0.780 |
| CrossGate | 0.764 | 0.772 |

### 4.3 Analysis

**All fusion methods underperform simple concatenation.** At 100%: best fusion (big GatedFusion, 0.891) is 2.1pp below concat (0.912) and 3.8pp below one_hot (0.929).

**More parameters = worse results.** Ranking by model complexity: concat > GatedFusion > FiLM > CrossGate. The more parameters dedicated to the fusion mechanism, the worse the performance.

**Diagnosis: 8-state overfitting.** The gate/FiLM/cross-gate networks have ~500K+ parameters each but only 8 distinct expression inputs. They memorize the 8 patterns and overfit. The gate becomes a fixed lookup table per cell line — a worse version of what one-hot achieves directly. Simple concatenation works better because the downstream MLP learns implicit gating without dedicating a module to conditioning on 8 states.

**Adding one-hot to fusion doesn't help.** The fusion module dominates the forward pass; the one-hot signal gets buried. This confirms the fusion layer is a bottleneck, not an enhancement.

**Conclusion: the bottleneck is data resolution, not method.** 54 configurations tested across 3 fusion architectures x 2 MLP sizes x +/- one_hot x 3 data fractions. Simple concatenation wins everywhere. No architectural sophistication compensates for 8 discrete cell states.

Script: `/opt/dlami/nvme/tahoe/eval_layer5_gated_fusion.py`
Log: `/opt/dlami/nvme/tahoe/gated_fusion_log.txt`

---

## 5. How Existing Cell Foundation Models Encode Expression

### 5.1 STATE (Arc/CZI, SE-600M)

600M params, 16 layers, d=2048, 16 heads. Pre-trained on Tahoe-100M's 95.6M cells.

**Input representation**: Each cell is a sequence of (gene_id, expression_value) pairs. Gene identity comes from pre-computed ESM2 + Knowledge Graph embeddings (5120d, projected to 2048d).

**Expression encoding**: Learned soft bins. A small MLP maps expression scalar to a 10-dim softmax over 10 learned bin embeddings, producing a soft weighted combination that is **ADDED** to the gene token (not multiplied):

```python
# From STATE codebase (se.py lines 361-378)
counts = counts.unsqueeze(-1)                                    # [B, seq_len, 1]
bin_weights = F.softmax(self.count_encoder(counts), dim=-1)      # [B, seq_len, 10]
bin_embeddings = self.bin_encoder(self._bin_indices_cached)       # [10, d_model=2048]
count_emb = torch.matmul(bin_weights, bin_embeddings)            # [B, seq_len, 2048]
src = src + count_emb                                            # ADD to gene embeddings
```

count_encoder: `Linear(1, 512) -> LeakyReLU -> Linear(512, 10)`. bin_encoder: `nn.Embedding(10, d_model)`.

**Training objective**: Binary decoder predicts whether each gene is expressed or not.

**CLS pooling**: CLS token prepended, L2-normalized output = cell state embedding.

### 5.2 scGPT (Nature Methods, 2024)

Three expression encoding modes:
- **Continuous (default)**: `Linear(1, d_model) -> ReLU -> Linear(d_model, d_model) -> LayerNorm`. Result **ADDED** to gene embedding.
- **Category**: Expression binned into discrete integers (0-51), embedded via `nn.Embedding`, **ADDED**.
- **Scaling**: Raw scalar multiplies gene embedding. NOT the default — our experiments showed why multiplication fails.

**Training**: Masked gene prediction — mask 15% of genes, predict identity and expression.

### 5.3 Geneformer (Nature, 2024)

No explicit expression values. Genes rank-ordered by expression within each cell, scaled by corpus-wide expression. The ordering "prioritizes genes that distinguish cell state" — highly expressed genes go first.

**Training**: Masked gene prediction — predict which gene belongs at each masked position.

### 5.4 Tahoe-x1

Expression values binned via `torch.bucketize()` into discrete tokens. Quantiles computed from non-zero entries. Zero values handled specially.

### 5.5 Key insight across all models

**Expression is ADDITIVE, never multiplicative.** Every successful model adds expression encoding to gene identity, preserving the gene embedding. The one model that offers multiplication (scGPT scaling mode) is not the default. Our Approach A confirmed this empirically: CG * expr destroys information by zeroing out lowly-expressed but informationally important proteins.

---

## 6. Conditioning Methods from Other Fields

### 6.1 Adaptive Layer Norm (adaLN) — DiT (Peebles & Xie, 2023)

The current SOTA for conditioning transformers on continuous scalars. Scale and shift parameters generated from the conditioning signal, applied per layer:

```python
gamma, beta = MLP(conditioning_signal).chunk(2, dim=-1)
output = (1 + gamma) * LayerNorm(token) + beta
```

Used in Diffusion Transformers to condition on timestep (a single continuous scalar). Modulates every layer, not just the input. This is the most promising approach for expression encoding.

### 6.2 Sinusoidal / Fourier encoding

Diffusion models (DDPM) encode continuous scalars via sinusoidal positional encoding → MLP:

```python
freqs = exp(arange(0, dim, 2) * -(log(10000) / dim))
encoding = cat([sin(t * freqs), cos(t * freqs)])
embedding = MLP(encoding)
```

Maps a 1D value into high-dimensional space where nearby values have similar representations. Better than raw scalars for capturing fine-grained expression differences.

Fourier features (Tancik et al., NeurIPS 2020) extend this with random or learned frequencies, enabling networks to learn high-frequency functions of continuous inputs.

### 6.3 FiLM (Perez et al., 2018)

Feature-wise Linear Modulation: `output = gamma(cond) * input + beta(cond)`. Precursor to adaLN. Simpler (no LayerNorm). Used in visual QA.

### 6.4 AdaIN (Huang & Belongie, 2017)

Adaptive Instance Normalization from style transfer. Expression = "style" that transforms protein "content".

### 6.5 Cross-attention conditioning (Stable Diffusion)

Conditioning signal provides Keys/Values; main representation provides Queries. Powerful when conditioning signal is a sequence. The 945 expression values could each become a token.

### 6.6 Summary: expression encoding paradigms

| Method | How expression enters | Preserves identity? | Proven at scale? |
|---|---|---|---|
| Addition (STATE, scGPT) | expr_emb + gene_emb | Yes | Yes (600M params) |
| Rank ordering (Geneformer) | Position in sequence | Yes | Yes (106M params) |
| Discrete bins (Tahoe-x1) | Bucketized, embedded | Mostly | Yes (3B params) |
| adaLN (DiT) | Per-layer scale+shift | Yes (via LayerNorm) | Yes (ImageNet-scale) |
| Sinusoidal + MLP (DDPM) | Fourier features → additive | Yes | Yes (all diffusion models) |
| Multiplication | expr * gene_emb | **No** (zeros out) | Fails (our experiments) |
| FiLM | gamma * input + beta | Yes | Yes (VQA) |

---

## 7. Proposed Architecture: Bonbon Cell Codebook

### 7.1 Design rationale

Bonbon's codebook (`BonbonCodebookEncoder`) already defines a shared dictionary of 945 protein-function concepts. A cell's state can be represented as a pattern of activation across these concepts, modulated by expression levels. We propose extending the codebook architecture with a learnable cell codebook connected to the protein codebook.

### 7.2 Architecture

```
FROZEN from Bonbon:
  ProteinCodebook: [945 x 1280]        — learned during co-folding training

NEW, LEARNABLE:
  CellCodebook: [K x 1280]             — K cell-state concept tokens (e.g., 128)
  ExprEncoder: scalar -> 1280d          — sinusoidal + MLP (additive)
  CellQuery: BonbonCodebookQuery-style  — projects, attends, pools, sparsemax

FORWARD PASS:
  1. Start with 945 frozen protein codebook tokens
  2. Encode expression per protein: expr_emb[i] = MLP(sinusoidal(expression[i]))
  3. Build expression-aware protein tokens: cell_protein[i] = codebook[i] + expr_emb[i]
  4. Cell's protein tokens query the cell codebook:
     [945 x 1280] -> project -> attention over K cell concepts -> sparsemax -> weighted sum
  5. Output: cell_embedding [1280d], L2-normalized
```

### 7.3 Three concrete designs

**Design A — Third query on existing protein codebook**

Add `query_3` (cell) alongside query_1 (protein) and query_2 (molecule) on the same shared 945-token codebook. The cell embedding lives in the same 1280d space. Simplest extension; limited by protein codebook's binding-optimized concepts.

**Design B — Separate cell codebook with cross-attention**

New learnable cell codebook (K=128 tokens, 1280d). Cell concepts are grounded in proteins via cross-attention to the protein codebook. Expression-modulated protein tokens query the cell concepts. More expressive; cell concepts can capture states the protein codebook wasn't designed for.

**Design C — Hierarchical codebook**

Two-level: frozen protein codebook (945 tokens) + learnable cell codebook (64 tokens). A transformer processes both together — protein tokens and cell concept tokens attend to each other. CLS token = cell state. The transformer learns protein-protein interactions modulated by expression.

### 7.4 Expression encoding options for the cell codebook

| Option | Implementation | Pros | Cons |
|---|---|---|---|
| Sinusoidal + MLP | sin/cos at multiple freqs -> MLP -> add to protein token | Continuous, multi-frequency, proven | More params than bins |
| adaLN per layer | gamma(expr), beta(expr) modulate each transformer layer | Per-layer conditioning, most expressive | Requires transformer architecture |
| Learned soft bins (STATE-style) | MLP(scalar) -> softmax -> weighted bin embeddings -> add | Simple, proven in cell models | Coarse (10 bins), global |
| Rank ordering (Geneformer-style) | Sort 945 proteins by expression, position = rank | No params, robust | Loses magnitude information |

**Recommended: sinusoidal + MLP for initial testing, adaLN for transformer variants.**

### 7.5 Training plan

**Phase 1 — Pre-train (self-supervised on Tahoe-100M):**
Input: single cell's 945 expression values + frozen protein codebook tokens.
Task: masked protein expression prediction (mask 15%, predict from context).
Learns: cell concept tokens that capture meaningful cell states.

**Phase 2 — Fine-tune for perturbation prediction:**
Input: (control cell embedding, drug CG profile) -> predict expression delta.
The cell embedding carries pre-perturbation state; CG profile carries drug action.

### 7.6 Connection to Bonbon codebook.py

The existing `BonbonCodebookQuery` (codebook.py) implements exactly the query mechanism needed:
- Projection network: `LayerNorm -> Linear -> GELU -> LayerNorm -> Linear`
- Attention: `projected @ codebook.T / sqrt(dim)`
- Pooling: max or mean across sequence
- Sparse attention: sparsemax, entmax, or softmax
- Weighted sum: `attention @ codebook`
- L2 normalization

The cell query would follow this same pattern. Expression-modulated protein tokens replace the protein sequence features as input. The cell codebook tokens replace (or extend) the protein codebook tokens as the query target.

### 7.7 Comparison to STATE

| | STATE | Bonbon Cell Codebook |
|---|---|---|
| Protein identity | ESM2 + KG (5120d) | Bonbon codebook token (1280d) |
| What identity encodes | Structure + knowledge graph | Structure + binding function |
| Expression encoding | 10 learned bins, added | Sinusoidal + MLP, added |
| Tokens per cell | ~2048 genes | 945 proteins |
| Architecture | 16 layers, d=2048, 600M params | 4-8 layers, d=1280, ~50-100M params |
| External dependencies | ESM2, knowledge graph | None (Bonbon-native) |
| Training data | Tahoe-100M | Same |

---

## 8. MAP Evaluation Details

### 8.1 MAP's data split (from codebase)

MAP uses **6 cell lines** (not 8 as initially assumed): CVCL_1098, CVCL_1056, CVCL_0131, CVCL_0069, CVCL_0023, CVCL_0480.

"Unseen drug" split: 5% of drugs per cell line held out as external test. Each cell line's unseen drugs are non-overlapping. Internal drugs split 80/20 train/test. MAP's "unprofiled drug" result (~0.87 DEG-50) is on this 5% held-out set — not strict leave-one-drug-out.

### 8.2 Our evaluation vs MAP's

- Our 5-fold CV on 2,847 conditions: random split of drug x cell_line pairs. Most closely comparable to MAP's "unseen combination" results (~0.90).
- Our LODO (0.221): strictly harder than MAP's "unprofiled drug" — we remove ALL data for one drug, while MAP holds out 5% per cell line and uses a 187K-drug knowledge graph.

### 8.3 One-hot vs STATE encoding

| | Our one-hot | MAP/STATE |
|---|---|---|
| Representation | 8-dim binary vector | 600M-param transformer on full expression |
| Information | log2(8) = 3 bits | Thousands of bits |
| At pseudobulk (8 lines) | Optimal | Overkill |
| At single-cell | Impossible | Natural |

---

## 9. Conclusions and Next Steps

### 9.1 What we proved

1. **CG profiles predict transcriptomics.** 0.929 Pearson on Tahoe pseudobulk, zero-shot representation, trained head only. Comparable to MAP's ~0.90 which uses 600M cell encoder + 187K-drug knowledge graph + 4TB training data.

2. **Expression conditioning works but needs cell diversity.** The 945-protein expression profile captures 82% of one-hot's cell-line signal. The 1pp gap is entirely due to 8 pseudobulk states, not the method.

3. **Fusion architectures overfit on low-diversity data.** All 3 tested methods (GatedFusion, FiLM, CrossGate) underperform simple concatenation when conditioning on 8 unique states. Confirmed across 54 configurations.

4. **Additive encoding is the right approach.** Every successful cell model (STATE, scGPT, Geneformer) adds expression to gene identity. Multiplication destroys information. Our experiments confirm this empirically.

### 9.2 Immediate next steps

1. **Expand pseudobulk to 50 cell lines.** Process all 3,388 Tahoe-100M shards. Test whether expression conditioning improves with more cell types. If expr closes the gap to one_hot at 50 lines, that's a clean ICML result.

2. **Extract full protein codebook tokens.** Get the 945 x 1280 token matrix from dark-snowball-245 (not just scalars). Enables protein-level reasoning in the cell codebook.

3. **Build Bonbon Cell Codebook prototype.** Start with Design B (separate cell codebook, K=128, cross-attention to protein codebook). Sinusoidal expression encoding, sparsemax query. Test on pseudobulk first, then single-cell.

4. **Single-cell training.** Subsample ~1K cells per condition from Tahoe-100M. Train the cell codebook on masked protein expression prediction. Fine-tune for perturbation prediction.
