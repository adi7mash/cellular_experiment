# Bonbon Virtual Cell Roadmap

## Executive Summary

This document outlines the path from Bonbon's current pair-level molecular interaction model to a ground-up virtual cell — a model that predicts cell-level behavior (expression changes, drug response, genetic interactions) by composing pretrained pairwise representations through a proteome-scale transformer.

The core insight: Bonbon already encodes functional biology at the pair level (proven by RxRx3 and EFAAR results). The next step is learning to compose those representations into cell-level predictions.

---

## 1. Current State: What Bonbon Already Encodes

### 1.1 Evidence from benchmarks

| Benchmark | What it tests | Result | vs. Best Published |
|-----------|--------------|--------|-------------------|
| RxRx3 CC AUROC | Compound → cell phenotype | 81.1% (pair CV) | Boltz-2 76.0% |
| RxRx3 per-target | Per-gene compound discrimination | 60.3% zero-shot, 89.9% trained | Boltz-2 53.9% |
| EFAAR JUMP-CP (7,976 genes) | Protein → biological networks | CORUM .316, HuMAP .338, StringDB .373 | MAE-G/8: .264, .215, .235 |
| Active moiety attention | Pharmacophore recovery | 654/735 targets significant | — |
| JUMP-lite CG→CRISPR | Cross-modal retrieval | 50.1% recall@1% | — |

### 1.2 The gap

Bonbon treats every protein-ligand pair independently. It knows what protein A does with ligands and what protein B does with ligands, but not what happens when A and B are in the same cell. It cannot predict:

- How knocking down a gene changes expression in a specific cell line
- Why the same drug kills some cell lines but not others
- Synthetic lethality (knocking down A and B together is lethal, but neither alone)
- Pathway-level cascading effects

---

## 2. Benchmark Landscape Survey

### 2.1 Benchmarks that test the virtual cell thesis

#### Tier 1 — Direct fit, published baselines, feasible now

**Zhong et al. 2025 (Rice, bioRxiv 2025.01.29.635607)** — Gene embedding benchmark
- 38 methods compared on synthetic lethality (SL), negative genetic interactions (NG), transcription factor targets (TF)
- Uses Costanzo yeast data (ortholog-mapped) + human datasets
- Top: GenePT 0.86 AUROC (uses biomedical literature), ESM2 ~0.75, PPI adjacency 0.79
- Open codebase: github.com/ylaboratory/gene-embedding-benchmarks
- Tests thesis: Do protein embeddings predict genetic interactions?

**RxRx3-core (Kraus et al. 2025, ICLR LMRL Workshop)**
- Recursion's own evaluation with published baselines: CellProfiler, OpenPhenom-S/16, MAE-G/8, MolPhenix, CLOOME
- Dataset on HuggingFace, EFAAR code public
- Near-zero additional effort — we already have the embeddings

**Chem2Gen-Bench (Lin & Chen, June 2026, arxiv 2606.21109)**
- 260K chemical + 1.1M genetic perturbation profiles
- Tests chemical-to-genetic translation
- Foundation model embeddings did NOT consistently beat gene-delta baselines
- Very new (June 2026) — risk of being too fresh for reviewers

#### Tier 2 — Adjacent but different output modality

**Arc Institute Virtual Cell Challenge 2026**
- Zero-shot CRISPRi knockdown prediction across 6 unseen cell lines
- Published in Cell, $100K prize, submissions due Nov 5, 2026
- 2025 results: No model beat naive baselines consistently; BioMap won with hybrid (100B+ params + statistics + protein embeddings)
- Requires predicting full gene expression vectors — different output modality from our embeddings
- Key insight: 2025 winners (BioMap, XLearning) both used protein embeddings (ESM-2) as features

**VCBench (June 2026, bioRxiv)** — Multi-dimensional virtual cell benchmark
- 7 capability dimensions; baselines matched or beat every foundation model on 4/5 dimensions
- Designed for single-cell foundation models (scGPT, Geneformer, UCE)

**PerturBench / scPerturBench (Nature Methods 2025)**
- 27 methods, 29 datasets for perturbation prediction
- Key finding: deep learning doesn't beat mean-of-training-examples

#### Not directly useful for our thesis

**DepMap** — Resource (gene essentiality across ~1,000 cancer cell lines), not a benchmark format
**Costanzo raw** — 23M yeast double mutants; Zhong benchmark already wraps this

### 2.2 The GeroAI observation

GeroAI_v12 currently leads the Arc 2026 leaderboard. Gero (gero.ai) is a Singapore longevity biotech company using physics-inspired AI for aging. No published paper on their challenge method yet. The 2025 challenge showed that hybrid approaches (deep learning + classical statistics + protein embeddings) consistently outperformed pure neural approaches.

---

## 3. Architecture Options Considered

### 3.1 Options evaluated and narrowed

| Option | Description | Pros | Cons | Decision |
|--------|-------------|------|------|----------|
| Perturbation decoder head | Freeze Bonbon, add MLP/cross-attention decoder → expression delta | Simplest, fastest | Doesn't learn inter-protein context | Good for baseline / Arc challenge entry |
| GNN on pathway graph | Bonbon embeddings as nodes on KEGG/STRING graph | Leverages known biology | Hardcodes potentially wrong/incomplete graph structure | **Rejected** — user prefers learning structure from data |
| Proteome transformer | Self-attention over all ~20K protein tokens | Learns wiring diagram from data; hierarchical | 20K context length; training data requirements | **Selected** as primary architecture |
| Codebook composition (linear) | Cell state = Σ expression_i × PA_i; perturbation = remove/add terms | Zero training; directly tests thesis | Linear; can't capture nonlinear interactions | Good as baseline |
| Cross-attention composition | Perturbation embedding cross-attends to proteome | Closest to existing Bonbon architecture | Single-hop attention may miss cascades | Good as intermediate step |
| Perceiver bottleneck | Learned latent tokens cross-attend to proteome | Scales better than full attention | Loses some fine-grained interactions | Alternative if full attention too expensive |
| Generative (bonbon-shu adaptation) | UDLM discrete diffusion over codebook/expression space | Can model distributions, not just point estimates | More complex; requires careful adaptation | **Can be added as generative layer on top** |

### 3.2 Selected architecture: Proteome Transformer

**Why**: It's hierarchical (Bonbon compresses molecular level → proteome transformer compresses proteome level), it learns the wiring diagram from data rather than hardcoding it, and it directly embodies the thesis — pair-level representations compose through attention to produce system-level predictions.

**Why it works at scale**: Without Bonbon, a proteome transformer would need 20K proteins × ~500 residues = 10M tokens. Bonbon compresses each protein's functional identity into a single 1024-1280 dim token via pretraining on millions of molecular interactions. The proteome transformer then operates on 20K tokens — feasible with efficient attention on A100s.

---

## 4. Proteome Transformer Architecture

### 4.1 Hierarchy

```
Level 0: Raw sequences      protein ~500 AAs, ligand ~50 atoms
         ↓ Bonbon (frozen, pretrained on pair-level interactions)
Level 1: Pair-level tokens   1 token per protein, 1024-dim (projector) or 1280-dim (codebook)
         ↓ Expression gating (cell-line specific)
Level 2: Proteome Transformer (trained on perturbation data)
         ↓ Self-attention learns inter-protein relationships
Level 3: Cell state          emergent from attention over all tokens
         ↓ Task heads
Level 4: Predictions         expression changes, viability, morphology, genetic interactions
```

### 4.2 Input representation: Expression-gated Bonbon tokens

Each protein's Bonbon embedding is modulated by its expression level in the specific cell line:

```python
# Expression gating
gate_i = σ(W_gate · log(expr_i + 1) + b_gate)
effective_token_i = gate_i × BonbonEmb_i + ExpressionLevelEmb(expr_i)
```

Three components:
1. **Expression gating**: Learned gate scales Bonbon token by expression level. Unexpressed proteins suppressed, highly expressed proteins dominate
2. **Expression level embedding**: Continuous positional-style embedding of expression level, added to token
3. **Self-attention context**: Once tokens are expression-weighted, attention patterns change per cell line — same transformer, different cell states

### 4.3 Leveraging the codebook

The codebook is Bonbon's strongest feature — a discrete, compositional vocabulary of ~8K interaction motifs. The proteome transformer should use it:

**Codebook tokens as input** (1280-dim, strongest on EFAAR):
- Each protein token is the codebook token embedding, not just the projector
- The codebook decomposition is preserved — the transformer sees the functional motif composition

**Pooled attention (PA) vectors as auxiliary signal** (8192-dim):
- PA vectors encode which codebook motifs are active for each protein
- Can be used as: (a) additional input features concatenated to tokens, (b) a separate codebook-attention channel, or (c) supervision signal (predict PA changes after perturbation)

**Codebook-aware perturbation modeling**:
- When a gene is knocked down, its codebook activations are removed from the cell state
- When a compound is added, the compound's codebook activations (from Bonbon's molecule query) are injected
- The transformer predicts how the remaining codebook activation landscape changes

### 4.4 Cell context: Different pair outcomes for different cells

The same EGFR protein has the same Bonbon embedding everywhere — but EGFR knockdown kills some cell lines and does nothing to others. Expression gating solves this:

- **Different cell lines** = different expression weights = different effective token sets = different self-attention patterns = different predictions
- **Zero-shot to new cell lines**: Any cell line with an expression profile works — no cell-line-specific parameters
- **Tissue-specific biology emerges**: In a liver cell, liver-specific proteins attend strongly to each other; in a neuron, synaptic proteins dominate

### 4.5 Perturbation modeling

| Perturbation type | Implementation |
|-------------------|---------------|
| Gene knockdown | Set expression_g → 0 (gate suppresses token) |
| Compound treatment | Add compound's Bonbon embedding as extra token |
| Overexpression | Amplify expression_g weight |
| Combinatorial (drug + KO) | Both modifications simultaneously |
| Predict outcome | Compare transformer output before/after modification |

### 4.6 Task heads

Multiple downstream heads share the same proteome transformer backbone:

1. **Expression prediction head**: Transformer output → MLP → Δexpression per gene (for Perturb-seq, Arc challenge)
2. **Viability head**: CLS token → scalar viability prediction (for PRISM/GDSC)
3. **Genetic interaction head**: Pairwise gene token comparison → epistatic effect (for DepMap, Costanzo)
4. **Morphology head**: CLS token → Cell Painting feature prediction (for JUMP-CP)

### 4.7 Codebook-native design principles

The codebook is Bonbon's strongest differentiator — a discrete, compositional vocabulary of ~8K interaction motifs. The proteome transformer must leverage it, not treat Bonbon as a generic feature extractor.

#### 4.7.1 Three-query codebook architecture (planned)

Currently `BonbonCodebookEncoder` has two queries on a shared codebook:
- **query_1** (protein → codebook): "what this protein CAN do" — its functional repertoire
- **query_2** (molecule → codebook): "what this molecule CAN do" — its chemical repertoire

**Planned addition**: **query_3** (fusion → codebook) on the same shared codebook:
- **query_3** (interaction → codebook): "what HAPPENS when they interact" — the interaction itself decomposed into the same motif vocabulary

The shared codebook constraint forces alignment: a kinase domain (query_1), an ATP-competitive scaffold (query_2), and the kinase-inhibitor interaction (query_3) all activate the same codebook entries — three views of the same functional motif in a unified vocabulary.

When PPI training is added (protein encoder for both sides), query_3 captures PPI interaction motifs in the same codebook, creating a universal interaction vocabulary spanning protein-ligand and protein-protein interactions.

#### 4.7.2 Impact on cell state representation

**Without query_3 (protein-only)**:
```
C = Σ expression_i × PA_protein_i  (8192-dim, captures what proteins CAN do)
```

**With query_3 (interaction-aware)**:
```
C = Σ expression_i × PA_interaction_i  (8192-dim, captures what proteins ARE doing)
```

Where `PA_interaction_i` is query_3 output aggregated across protein i's active interactions (endogenous ligands, protein partners). A kinase with its inhibitor bound has a different interaction PA than a free kinase — the cell state captures active functional state, not just protein identity.

#### 4.7.3 Per-protein token enrichment

With query_3, each protein's input to the proteome transformer carries three signals:

| Signal | Source | What it encodes |
|--------|--------|----------------|
| Protein identity | query_1 PA (8192-dim) | What this protein CAN do |
| Interaction profile | Aggregated query_3 PAs across partners | What this protein IS doing |
| Expression weight | Cell-line expression level | How much of it is present |

This gives the proteome transformer context about each protein's active state, not just its identity.

#### 4.7.4 Drug perturbation becomes interaction-specific

When adding a compound to the cell, inject the **specific interaction codebook activations** (query_3 PA) for that compound with each of its targets — the model sees exactly which interaction motifs the drug activates or disrupts.

#### 4.7.5 Generative layer in interaction-codebook space

bonbon-shu's UDLM denoises in query_3 codebook space — predicting which interaction motifs are active after perturbation. More biologically meaningful than expression space, since codebook entries have interpretable functional meaning (validated by pharmacophore recovery in active moiety analysis).

#### 4.7.6 Prerequisite

Train Bonbon with query_3 on the fusion transformer output. This is a bonbontoken training change. Once query_3 exists, all downstream virtual cell plans are strictly stronger. Until then, query_1 (protein) codebook tokens serve as the input — still the strongest embedding type on EFAAR.

---

## 5. Generative Layer (bonbon-shu adaptation)

### 5.1 Current bonbon-shu architecture

bonbon-shu is a **discrete diffusion molecular generator** based on UDLM (Uniform Diffusion Language Models). Key components:

- **DiT backbone**: Diffusion Transformer for iterative denoising of discrete tokens
- **Protein conditioning (two paths)**:
  - Pooled protein embedding → adaLN (adaptive LayerNorm) — global context modulates every block
  - Full protein sequence → cross-attention (ProteinCrossAttn) — molecule tokens attend to protein embeddings
- **Guidance**: CFG (classifier-free), CBG (classifier-based), retrieval-guided, negative prompting
- **Already uses Bonbon embeddings**: Loads from `.pt` files, imports BonbonProjection

### 5.2 How to add generative capability on top of the proteome transformer

The proteome transformer produces a deterministic cell state. bonbon-shu's UDLM can be added as a **generative decoder** on top:

```
Proteome Transformer → Cell state embedding
                        ↓
                  bonbon-shu UDLM (adapted)
                        ↓
              Sampled expression profile / codebook activations
```

**Adaptation of bonbon-shu for expression generation**:

| Current (molecular generation) | Virtual cell adaptation |
|-------------------------------|----------------------|
| Input conditioning: single protein embedding | Input conditioning: cell state from proteome transformer |
| Output: SAFE/SMILES tokens (~256 length) | Output: expression delta tokens or codebook activation changes |
| Vocab: ~165-300 molecular tokens | Vocab: discretized expression bins (~256 bins) or codebook entries (~8K) |
| adaLN conditioning: pooled protein emb | adaLN conditioning: cell-state CLS token |
| Cross-attention KV: protein sequence | Cross-attention KV: proteome transformer hidden states |

**Sequence length solutions** (20K genes vs. ~256 molecular tokens):
- **Sparse DE generation**: Generate only differentially expressed genes as (gene_id, direction, magnitude) tuples — natural fit for variable-length generation
- **Codebook activation diffusion**: Denoise in codebook space (~8K entries) — biologically meaningful discrete space
- **Pathway-level tokenization**: ~300 pathway activity tokens — matches molecular generation length
- **Residual generation**: Model only the perturbation delta, not the full expression

### 5.3 Value of the generative layer

- **Distribution prediction**: Perturbation responses are stochastic — different cells respond differently. Generative model captures the distribution, not just the mean
- **Sampling**: Generate multiple plausible post-perturbation states; use variance as uncertainty
- **Arc challenge**: The evaluation metrics (PDS) reward discriminating between perturbations — a generative model that produces sharp, distinct distributions per perturbation scores well

---

## 6. Training on PPI Data and Molecules

### 6.1 PPI training: Peptides-as-SAFE vs. protein encoder

**Current Bonbon**: Trained on protein-ligand pairs (protein encoder × molecule encoder). Has a separate model with peptides encoded as SAFE alongside small molecules.

**Option A: Peptides-as-SAFE for PPI**
- Encode the shorter partner in a PPI pair as SAFE (peptide representation)
- Pros: No architecture change; uses existing molecule encoder pipeline
- Cons: SAFE was designed for small molecules, not protein fragments. Loses 3D structural information. Limited to peptide-length sequences (~50 AAs max). The SAFE tokenizer may not handle amino acid sequences well

**Option B: Protein encoder for both sides of PPI**
- Use the protein encoder (ESM2/ModernBERT) for both interaction partners
- Pros: Full-length proteins; native protein representation; captures structural features
- Cons: Needs architecture modification — currently assumes asymmetric encoders (protein × molecule)

**Recommendation**: Option B is better for PPI. The protein encoder already handles full-length sequences and captures structural/evolutionary features that SAFE cannot. The codebook is shared between both queries (query_1 for protein, query_2 for molecule) — for PPI, both partners would use query_1, or a new query_3 could be added for the second protein. The BonbonCodebookEncoder already supports this architecturally — it has two independent query mechanisms on a shared codebook.

### 6.2 Training Bonbon on PPI data

**Data sources for PPI pretraining**:
- STRING (>10B interactions, scored), HuMAP (human protein complexes), BioGRID (~2M curated interactions)
- Training signal: contrastive (interacting pairs are positive, random pairs are negative) + codebook alignment (same as current Bonbon training)

**Impact on virtual cell**: PPI-trained Bonbon embeddings would directly encode inter-protein relationships, making the proteome transformer's job easier — it wouldn't have to learn the wiring diagram entirely from downstream supervision.

### 6.3 Molecules in virtual cell training

**Yes, molecules should be used**. The proteome transformer needs to model drug effects, not just genetic perturbations. Training data:

| Data source | Perturbation type | Readout | Scale |
|-------------|-------------------|---------|-------|
| Replogle 2022 | CRISPR knockdown | scRNA-seq | ~5K genes, 2 cell lines |
| Norman 2019 | CRISPR (combinatorial) | scRNA-seq | ~100 gene pairs |
| PRISM | Compound treatment | Viability | 4,518 drugs × 578 cell lines |
| GDSC | Compound treatment | Viability + IC50 | ~500 drugs × ~1,000 cell lines |
| L1000/LINCS | Compound + genetic | Gene expression (978 landmarks) | ~20K perturbations |
| JUMP-CP | Compound + CRISPR | Cell Painting morphology | ~100K compounds, ~8K genes |

Compound perturbations are modeled by injecting the compound's Bonbon embedding as an additional token in the proteome transformer input.

---

## 7. Reusable BonbonToken Components

### 7.1 Inventory of existing code

| Component | File | Class/Function | Reuse status |
|-----------|------|---------------|--------------|
| **ModernBERT backbone** | `embedding/modernbert.py` | HF `ModernBertModel` (RoPE, flash attention) | **Reuse as-is** for proteome transformer backbone |
| **ModernBERT + cross-attention** | `embedding/modernbert_cross_attention.py` | `ModernBertCrossAttention` (separate Wq/Wk/Wv/Wo, flash_attn_varlen_func, cu_seqlens support) | **Reuse as-is** — production-grade flash attention cross-attention |
| **ModernBERT + token types** | `models/fusion/fusion_encoder.py` | `ModernBertWithTokenTypes` (adds token_type_embeddings to ModernBert) | **Reuse as-is** for distinguishing protein vs. compound vs. CLS tokens |
| **Bidirectional cross-attention encoder** | `models/fusion/fusion_encoder.py` | `BonbonBidirectionalCrossAttentionEncoder` (ModernBertConfig, dual CLS tokens, cls_pooler=Linear+Tanh) | **Adapt** — already handles two modality streams with CLS tokens, flash attention, ModernBERT |
| **Codebook** | `models/codebook.py` | `BonbonCodebook` (num_tokens × token_dim), `BonbonCodebookQuery` (projection→attention→pool, supports sparsemax/softmax/entmax), `BonbonCodebookEncoder` (shared codebook + query_1 + query_2 for two modalities) | **Reuse as-is** for extracting codebook tokens/PA from frozen Bonbon; for PPI, add query_3 or reuse query_1 for second protein |
| **Projection layers** | `blocks/projector.py` | `BonbonProjection` | **Reuse as-is** for projecting embeddings between dimensions |
| **Perceiver resampler** | `blocks/perceiver.py` | `PerceiverResampler`, `PerceiverAttentionLayer` | **Reuse** if bottleneck attention needed (compressing 20K tokens to latent set) |
| **RoPE transformer block** | `blocks/rope_block.py` | `RoPETransformerBonbonBlock` | **Available** but ModernBERT already has RoPE built in |
| **Transformer block** | `blocks/transformer.py` | `TransformerBonbonBlock` | **Available** as lightweight alternative |
| **CLS pooler** | `models/fusion/fusion_encoder.py` | `nn.Sequential(Linear, Tanh)` | **Reuse pattern** for cell-state CLS extraction |
| **Data module** | `data/datamodule.py`, `data/dataset.py` | `BonbonDataModule`, dataset classes | **Extend** with Perturb-seq data loader |
| **Data batch** | `data_classes/bonbon_batch.py` | `BonbonDataBatch` | **Extend** with expression profile fields |
| **Model output** | `data_classes/bonbon_output.py` | `BonbonModelOutput` | **Extend** with expression prediction fields |
| **Training loop** | `bonbon.py` | PyTorch Lightning CLI + WandB | **Reuse as-is** — handles S3, checkpoints, distributed training |
| **Loss functions** | `losses/` | Contrastive, pairwise, token losses | **Reuse** contrastive losses for PPI; **add** expression prediction losses |
| **ESM2 encoder** | `embedding/esm2.py` | ESM2 wrapper | **Reuse** for protein encoding in PPI mode |
| **Metrics** | `metrics/` | Alignment, uniformity, scoring | **Extend** with perturbation prediction metrics |

### 7.2 From bonbon-shu (for generative layer)

| Component | Reuse status |
|-----------|-------------|
| **DiT backbone** | **Reuse** for UDLM discrete diffusion |
| **ProteinCrossAttn** | **Adapt** — change from protein→molecule to cell_state→expression |
| **adaLN conditioning** | **Reuse as-is** — modulate blocks with cell-state embedding |
| **CFG/CBG guidance** | **Reuse** for guided expression generation |
| **UDLM noise schedule** | **Reuse as-is** |

### 7.3 What NOT to write from scratch

- **DO NOT** write a transformer — use `ModernBertModel` from HuggingFace (already in bonbontoken)
- **DO NOT** write cross-attention — use `ModernBertCrossAttention` (already written with flash attention)
- **DO NOT** write codebook query — use `BonbonCodebookQuery` (already handles sparse/softmax attention, pooling)
- **DO NOT** write training loop — use PyTorch Lightning CLI from `bonbon.py`
- **DO NOT** write projection layers — use `BonbonProjection`
- **DO NOT** write perceiver — use `PerceiverResampler`
- **DO NOT** write UDLM diffusion — use bonbon-shu's implementation

---

## 8. Execution Plan

### Phase 0: Codebook composition baseline (1-2 days)

**Goal**: Quick test of the linear composition hypothesis.

**Implementation**:
```python
# Cell state = weighted sum of PA vectors
cell_state = sum(expression[i] * PA_vector[i] for i in expressed_genes)
# Perturbation = modify weights
delta_cell_state = cell_state_perturbed - cell_state_baseline
# Decoder: delta_cell_state → delta_expression
model = MLP(8192, [2048, 1024], num_genes)
```

**Data**: Replogle et al. 2022 (K562, RPE1)
**Expected outcome**: If this works even modestly, it validates that codebook activations compose meaningfully

### Phase 1: Proteome transformer MVP (2-4 weeks)

**Goal**: Train a proteome transformer on Perturb-seq data.

**Architecture**:
- Input: 20,337 expression-gated Bonbon codebook tokens (1280-dim)
- Backbone: `ModernBertModel` (6-8 layers, 8-16 heads, hidden_dim=1280)
- Expression gating: learned gate per protein
- CLS token for cell state
- Expression prediction head: MLP from per-token outputs → Δexpression

**Training**:
- Data: Replogle 2022 (~5K knockdowns × 2 cell lines)
- Loss: MSE on differentially expressed genes + ranking loss (PDS-style)
- Freeze: Bonbon embeddings completely frozen; train only transformer + heads
- Hardware: 1-2 A100s, ~1-3 days

**Key implementation decisions**:
- Use `BonbonBidirectionalCrossAttentionEncoder` pattern but adapted for self-attention over proteome tokens
- Use `ModernBertWithTokenTypes` to distinguish protein tokens vs. perturbation tokens vs. CLS
- Expression gating as a new `ExpressionGate` module (small, ~100 lines)
- New `PerturbSeqDataModule` extending `BonbonDataModule` patterns

**New code needed** (~500-800 lines total):
- `ExpressionGate` module (~50 lines)
- `ProteomeTransformer` model class wrapping ModernBertModel (~200 lines)
- `ExpressionPredictionHead` (~50 lines)
- `PerturbSeqDataset` and `PerturbSeqDataModule` (~200 lines)
- Training config YAML (~50 lines)
- Evaluation script (~200 lines)

### Phase 2: Multi-task training + drug response (2-4 weeks)

**Goal**: Add compound perturbation modeling and multi-task heads.

**Additions**:
- Compound token injection (compound's Bonbon embedding as additional token)
- Viability prediction head (PRISM/GDSC)
- Genetic interaction head (DepMap, Costanzo)
- Multi-task loss balancing (already have uncertainty weighting in bonbontoken losses)

### Phase 3: Generative layer (2-4 weeks)

**Goal**: Add bonbon-shu UDLM as generative decoder for expression prediction.

**Adaptation**:
- Replace molecular token vocabulary with expression/codebook vocabulary
- Condition on proteome transformer cell state instead of single protein
- Reuse ProteinCrossAttn, adaLN, CFG from bonbon-shu

### Phase 4a: query_3 on fusion (2-4 weeks)

**Goal**: Add interaction-level codebook query to Bonbon.

**Architecture change**:
- Add `query_3` to `BonbonCodebookEncoder` — a third `BonbonCodebookQuery` that operates on fusion transformer output
- Same shared codebook as query_1 and query_2
- Training signal: existing Bonbon training objectives (contrastive + codebook alignment) — query_3 is trained end-to-end alongside the fusion transformer
- Implementation: ~50 lines in `models/codebook.py` (add query_3 init + forward), ~20 lines in the model forward pass

**Impact**: All downstream phases become strictly stronger — interaction-level codebook tokens replace protein-only tokens as input to the proteome transformer.

### Phase 4b: PPI pretraining (4-8 weeks)

**Goal**: Extend Bonbon pretraining to protein-protein interactions.

**Architecture change**:
- Use protein encoder (ESM2/ModernBERT) for both interaction partners — NOT peptides-as-SAFE (SAFE is limited to ~50 AAs and lacks structural features)
- query_3 on PPI fusion captures protein-protein interaction motifs in the same codebook
- Training signal: contrastive on PPI pairs (STRING, BioGRID)
- The codebook becomes a universal interaction vocabulary: protein-ligand AND protein-protein motifs
- Expected outcome: PPI-aware embeddings improve proteome transformer's ability to learn inter-protein context

### Phase 5: Arc Virtual Cell Challenge entry (by Nov 5, 2026)

**Goal**: Competitive entry on the 2026 challenge.

**Architecture**: Proteome transformer (Phase 1-2) + generative layer (Phase 3) + hybrid statistical baseline (inspired by 2025 winners)

---

## 9. Data Sources

### 9.1 Training data

| Dataset | Type | Scale | Access | Use |
|---------|------|-------|--------|-----|
| **Replogle et al. 2022** | Perturb-seq (CRISPRi) | ~5K genes × K562, RPE1 | Public (GEO) | Primary perturbation training |
| **Norman et al. 2019** | Perturb-seq (combinatorial) | ~100 gene pairs, K562 | Public | Combinatorial perturbation training |
| **PRISM** | Drug viability | 4,518 drugs × 578 cell lines | Public (DepMap portal) | Drug response training |
| **GDSC** | Drug IC50 | ~500 drugs × ~1,000 cell lines | Public | Drug response training |
| **L1000/LINCS** | Gene expression (978 landmarks) | ~20K perturbations | Public (clue.io) | Expression change training |
| **JUMP-CP** | Cell Painting morphology | ~100K compounds, 8K genes | Public | Morphology prediction training |
| **STRING** | Protein-protein interactions | >10B scored interactions | Public | PPI pretraining (Phase 4) |
| **BioGRID** | Curated PPI | ~2M interactions | Public | PPI pretraining (Phase 4) |
| **DepMap** | Gene essentiality | ~18K genes × ~1,000 cell lines | Public | Genetic interaction training |

### 9.2 Evaluation benchmarks

| Benchmark | What it tests | Published baselines | Priority |
|-----------|--------------|---------------------|----------|
| **Arc Virtual Cell Challenge 2025 test set** | Expression prediction post-CRISPRi | BioMap, XLearning, Team Outlier | High |
| **Zhong et al. gene embedding benchmark** | Genetic interactions (SL, NG, TF) | 38 methods (ESM2, GenePT, PPI) | High |
| **EFAAR (already done)** | Protein network recovery | MAE-G/8, CellProfiler | Done |
| **RxRx3 (already done)** | Compound → cell phenotype | Boltz-2 | Done |
| **Chem2Gen-Bench** | Chemical→genetic translation | Foundation models + baselines | Medium |
| **RxRx3-core (Kraus et al.)** | Cell Painting with Recursion baselines | CellProfiler, OpenPhenom, MAE-G/8 | High |

---

## 10. Key Precomputed Assets

All embeddings from dark-snowball-245 checkpoint, stored on disk and S3:

| Asset | Path | Dimensions |
|-------|------|------------|
| Proteome projector | `/opt/dlami/nvme/rxrx3_phenomics/results/dark-snowball-245/human_proteome/human_proteome_pairs_protein_protein_mean_dark-snowball-245/` | 20,337 × 1024 |
| Proteome codebook tokens | `.../human_proteome_pairs_protein_codebook_dark-snowball-245/tokens/` | 20,337 × 1280 |
| Proteome codebook PA | `.../human_proteome_pairs_protein_codebook_dark-snowball-245/pooled_attention/` | 20,337 × 8192 |
| Evaluation JSONs | `.../evaluation/` | 22 files |
| Compound embeddings | `.../fusion_cls/` | 1,674 × 1024 |
| S3 backup | `s3://outputbio-models/Embeddings/rxrx3-phenomics-dark-snowball-245/` | All above |
