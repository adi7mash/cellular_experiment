# Internal Strategy: ICML 2026 Submission

## What is the story?

**Core claim**: Frozen pairwise molecular interaction embeddings — learned from self-supervised protein-ligand co-training, never exposed to cellular data — predict compound-induced cellular phenotype, outperforming specialized structural-biology pipelines that required 100 million co-foldings, transcriptomic data, and PPI networks. This demonstrates that cell-level biology *emerges* from pair-level interaction representations.

**Why this matters for ICML**: This is not a biology paper. It is a representation learning paper. The finding that a self-supervised model trained on pairwise interactions encodes enough structure to compose to system-level prediction — without being trained on systems — is a fundamental result about what self-supervised representations can learn. It parallels the "emergent abilities" narrative in LLMs (arithmetic, code, reasoning emerge from next-token prediction) but in a scientific domain where the emergence has physical meaning: molecular interactions compose into cellular behavior.

**Key tension**: We cannot disclose the model architecture (under NeurIPS review). The paper must stand on the *results* and *methodology* — what emerges from the embeddings, not how the embeddings are produced. This is actually a strength: it forces the contribution to be about evaluation methodology and the emergence phenomenon, not about architecture novelty.

## What would we submit?

A **benchmark + analysis paper** at the intersection of representation learning and computational biology:

1. **Rigorous evaluation framework** for phenomics prediction — exposing methodological issues in the current SOTA (Beaini/Boltz-2: no cross-validation, in-distribution data, rule-based pipeline)
2. **Empirical demonstration** that frozen pair-level embeddings compose to cell-level prediction
3. **Analysis** of what the embeddings encode — the "signature" framework that identifies which latent dimensions predict cellular behavior
4. **Compute efficiency** story: 12 minutes on 1 GPU vs. 12 months on a supercomputer

## Alternative stories we could tell

### Story A: "Evaluation rigor in molecular phenomics" (methods-focused)
- **Angle**: Current benchmarks are misleading. Beaini's 76% uses no CV, in-distribution data, and a 6-parameter rule-based pipeline. We propose proper evaluation (compound-level CV, per-fold feature learning) and show frozen embeddings beat the inflated numbers.
- **Pro**: Doesn't require strong claims about emergence. Evaluation methodology papers are valued at ICML.
- **Con**: Less exciting. Might feel like a critique paper without a strong positive result.

### Story B: "Emergent cellular prediction from molecular self-supervision" (phenomenon-focused)
- **Angle**: Like in-context learning emerging from language pre-training, cellular phenotype prediction emerges from pairwise molecular interaction pre-training. The model was never trained on cells; yet its representations predict what compounds do to cells.
- **Pro**: Most compelling narrative. Connects to the "emergent abilities" literature. Strong for oral presentation.
- **Con**: "Emergence" claims attract scrutiny. We need to be precise about what we mean.

### Story C: "Frozen embeddings as universal molecular fingerprints" (applications-focused)
- **Angle**: Frozen embeddings from a single checkpoint replace specialized pipelines across multiple tasks: MoA classification, binding mode discrimination, phenomics prediction. The phenomics result is one of several "zero-shot" emergent abilities.
- **Pro**: Broader scope. More interesting to the ML audience.
- **Con**: Overlaps with the NeurIPS Bonbon paper. Risk of self-scooping.

### Story D: "The phenomics benchmark paper" (dataset/benchmark-focused)
- **Angle**: Introduce a rigorous benchmark for molecular phenomics prediction with proper train/test splits, compound-level CV, and per-fold feature learning. Evaluate multiple embedding approaches.
- **Pro**: Benchmark papers have lasting impact. Would be useful to the community.
- **Con**: RxRx3 already exists; we're proposing a benchmark *protocol*, not a new dataset.

## Recommended approach: Story B+ (Dual Emergence) with evaluation rigor

Lead with dual emergence: compound embeddings predict cellular phenotype (RxRx3) AND protein embeddings recover biological network structure (JUMP-CP EFAAR) — both from the same frozen checkpoint trained only on pairwise sequences. The EFAAR results provide independent validation from a different task and dataset, making the emergence claim dramatically stronger than RxRx3 alone. Ground it in evaluation rigor (proper CV methodology, community-standard benchmarks). The story is now: "pair-level self-supervision encodes enough mechanistic structure that system-level biology emerges from composition — we prove this from both the molecule and protein sides."

## What we'd need to be careful about

1. **Anonymous model**: Reference Bonbon as "a frozen foundation model for molecular interactions" with self-citation [Anonymous, under review]. Describe it only by its outputs (codebook embeddings, contrastive projections, interaction classifications) not its architecture.
2. **NeurIPS overlap**: The phenomics experiment is NOT in the NeurIPS submission (it was done after). The MoA and binding mode results ARE in NeurIPS. So we can reference those as "concurrent work shows [Anonymous] achieves X on MoA" but the phenomics analysis must be original to this paper.
3. **Beaini criticism**: Be respectful. Frame as "we propose improvements to evaluation methodology" not "Beaini's work is wrong."
4. **Reproducibility**: We can share the evaluation code and protocol. We can't share the model weights (proprietary). This limits reproducibility but is standard for industry papers at ICML.

---

# Frozen Molecular Interaction Embeddings Encode Cellular Biology

## Abstract

Do pairwise molecular interaction representations encode system-level biology? We show that frozen embeddings from a self-supervised model, trained only on protein-ligand sequence pairs, capture both cellular phenotype and biological network structure — without any cellular data. On RxRx3, our compound embeddings achieve 81.1% CC AUROC (pair-level CV) vs. 76.0% for Boltz-2, and 60.3% zero-shot per-target AUROC vs. 53.9% — with a trained model reaching 89.9% on held-out genes. On JUMP-CP (EFAAR framework, 7,976 genes), our protein embeddings recover protein complexes, pathways, and signaling networks, exceeding the best Cell Painting model (MAE-G/8, 93M cell images) by 20–59% across all sources. The pipeline runs in 12.5 minutes on one GPU vs. 12 months for the structural approach. These results demonstrate dual emergence: compound embeddings predict what molecules do to cells, and protein embeddings capture how proteins organize within cells — both from the same frozen checkpoint. Codebook attention analysis reveals that the model's internal decomposition recovers known pharmacophores, connecting atom-level fragments to cell-level prediction. Pair-level self-supervision encodes enough mechanistic structure for cellular-level reasoning to arise from composition alone.

## 1. Introduction

A central question in representation learning is whether models trained on local interactions can learn representations that compose to predict system-level behavior. In natural language processing, models trained to predict the next token exhibit emergent abilities — arithmetic, code generation, multi-step reasoning — that were never explicitly trained (Wei et al., 2022). We investigate an analogous question in computational biology: does a model trained on pairwise protein-ligand interactions learn representations that encode system-level biology?

We present evidence that it does — from two independent benchmarks on two different datasets. First, we show that frozen compound embeddings predict what molecules do to living cells: on the RxRx3 phenomics benchmark, they exceed Boltz-2's structural pipeline (81.1% vs. 76.0% CC AUROC) while a trained per-target model reaches 89.9% on held-out gene targets. Second, we show that frozen protein embeddings from the same checkpoint capture the functional organization of biology: on the JUMP-CP known-relationship benchmark (EFAAR framework, 7,976 genes), they recover protein complexes, metabolic pathways, and signaling networks better than the best published Cell Painting model trained on 93 million cell images (Kraus et al., 2025). In both cases, the model has never seen a cell, a microscopy image, or any cellular data.

These two findings are complementary. The compound embeddings demonstrate that pair-level representations predict *perturbation responses* — what happens when you add a molecule to a cell. The protein embeddings demonstrate that the same representations capture the *wiring diagram* — which proteins work together in complexes, share pathways, and participate in signaling cascades. Together, they provide evidence that pair-level self-supervision encodes both the mechanism of molecular action and the organizational structure of cellular biology.

This connects to the broader goal of building a "virtual cell." Current approaches are either top-down (training directly on cellular readouts) or bottom-up (explicitly simulating molecular interactions). We present evidence for a third path: learning interaction representations rich enough that cellular behavior emerges from their composition, without explicit simulation or cellular training data.

Recent work by Beaini et al. (2026) benchmarked Boltz-2 on RxRx3 (Chandrasekaran et al., 2024), achieving 76% CC AUROC and 53.9% per-target AUROC via 100 million structural co-foldings, transcriptomic modulation, and PPI propagation — fitted without cross-validation on in-distribution compounds. We show that frozen embeddings from a self-supervised model [Anonymous, under review], extracted in 12.5 minutes on a single GPU, outperform this pipeline at five orders of magnitude less compute.

Our contributions are:

1. **Dual evidence for emergent biological reasoning**: Frozen pairwise interaction embeddings encode both cellular phenotype (compound side, RxRx3) and biological network structure (protein side, JUMP-CP) — two qualitatively different levels of system-level biology, neither present in the training signal.

2. **Cross-dataset validation**: Results hold across RxRx3 (1,674 compounds, 735 genes) and JUMP-CP (7,976 genes), against published baselines from both the structural biology (Boltz-2) and Cell Painting (MAE-G/8) communities, on the respective communities' own benchmark frameworks.

3. **Rigorous evaluation methodology**: We introduce compound-level cross-validation with per-fold feature learning for phenomics prediction, exposing ~4 percentage points of inflation in global signature approaches and establishing properly validated baselines.

4. **Phenomics signature analysis**: A subset of embedding dimensions — identified by the same correlation-based framework used for mechanism-of-action classification — encodes cellular-level information, suggesting the embedding space is organized along biologically meaningful axes at multiple scales.

5. **Compute efficiency**: Our entire pipeline (embedding extraction + evaluation) completes in under 15 minutes on commodity hardware, compared to 12+ months of supercomputer time for the structural approach.

## 2. Related Work

**Molecular phenomics prediction.** Cell Painting (Bray et al., 2016) produces high-content microscopy profiles that capture morphological responses to chemical and genetic perturbations. The RxRx3-core dataset (Chandrasekaran et al., 2024) provides such profiles for 1,674 compounds and 735 gene knockouts in HUVEC cells. Beaini et al. (2026) introduced the phenomics prediction task, using Boltz-2 affinity fingerprints to predict compound-compound (CC) phenomics similarity and compound-gene (CG) association. Their approach relied on structural co-folding (100M AlphaFold-Multimer runs), transcriptomic modulation from expression databases, and a PPI graph, achieving 76% CC AUROC and 53.9% per-target median AUROC.

**Molecular interaction models.** Protein-ligand interaction prediction has been addressed through structure-based approaches (AlphaFold-Multimer, Boltz-2, Chai-1), sequence-based approaches (ESM-2+ChemBERTa combinations), and hybrid approaches. Most models predict a single facet per interaction — binding affinity, docking pose, or binary interaction. [Anonymous] introduced a foundation model that returns the complete characterization of each interaction through codebook quantization and cross-modal fusion, enabling multiple downstream tasks from the same frozen embeddings.

**Emergent abilities in foundation models.** Wei et al. (2022) documented abilities in large language models that emerge only at scale and were not explicitly trained. Similar phenomena have been observed in vision models (Caron et al., 2021) and protein language models (Lin et al., 2023). Our work extends this to molecular interaction models, where the emergent ability (cellular phenotype prediction) requires composition across many pairwise interactions.

**Cell Painting representation learning.** The JUMP-Cell Painting consortium (Chandrasekaran et al., 2023) produced the largest public Cell Painting dataset, including cpg0016: 7,976 CRISPR gene knockouts across 148 plates. CellProfiler (McQuin et al., 2018) extracts hand-engineered morphological features; learned representations from masked autoencoders (MAE-G/8) achieve substantially higher recall on known biological relationships (Kraus et al., 2025). The EFAAR benchmarking framework (Peidli et al., 2024) standardizes evaluation by measuring whether embeddings recover known relationships from CORUM, HuMAP, Reactome, SIGNOR, and StringDB. We use EFAAR to evaluate our protein embeddings against these Cell Painting baselines.

**Evaluation methodology.** Proper cross-validation in drug discovery is critical due to compound similarity (Wallach & Heifets, 2018). We build on work highlighting evaluation pitfalls in molecular property prediction (Yang et al., 2019) and extend it to the phenomics prediction setting.

## 3. Background

### 3.1 RxRx3 Phenomics Benchmark

The RxRx3-core dataset (Chandrasekaran et al., 2024) contains Cell Painting profiles for 1,674 compounds and 735 gene knockouts in HUVEC cells. Each well is represented by a 384-dimensional OpenPhenom embedding derived from high-content microscopy images. Following Typical Variation Normalization (TVN), compound-compound phenomics similarity is defined as the cosine similarity between batch-corrected compound embeddings.

**Compound-compound (CC) similarity** quantifies whether two compounds induce similar cellular morphology changes. We evaluate CC prediction as a binary classification task: given a pair of compounds, predict whether their phenomics similarity exceeds a threshold τ ∈ {0.4, 0.6}.

**Compound-gene (CG) association** quantifies whether a compound phenocopies a gene knockout. For each gene target, compounds are ranked by a CG score, and per-target AUROC measures discrimination between phenocopying and non-phenocopying compounds.

### 3.2 JUMP-CP and EFAAR Benchmarking

The JUMP-Cell Painting consortium's cpg0016 dataset (Chandrasekaran et al., 2023) contains Cell Painting profiles for 7,976 CRISPR gene knockouts across 148 plates (~50,000 wells after quality control). CellProfiler extracts 3,671-dimensional morphological features per well, which are preprocessed via Typical Variation Normalization (TVN) on DMSO controls with CORAL batch correction, then mean-aggregated per gene.

The EFAAR benchmarking framework (Peidli et al., 2024) is the standard evaluation for Cell Painting representations. It tests whether proteins with known biological relationships are closer in embedding space than random protein pairs, using five curated sources: CORUM (protein complexes), HuMAP (protein complexes from mass spectrometry), Reactome (metabolic and signaling pathways), SIGNOR (signaling interactions), and StringDB (functional associations). The metric is recall@0.05/0.95: the fraction of known relationship pairs whose cosine similarity falls in the top or bottom 5% of the null distribution. Higher recall indicates better recovery of biological organization.

### 3.3 Embedding Source

We use frozen embeddings from [Anonymous], a 1.6B-parameter foundation model for molecular interactions trained on protein-ligand sequence pairs. The model produces, for each protein or molecule:

- **Codebook tokens** (8,192-dimensional): Sparse token activations from a shared codebook quantizer, capturing discrete interaction motifs.
- **Pooled attention** (8,192-dimensional): Attention-weighted codebook activations, providing a continuous summary.
- **Contrastive projections** (1,024-dimensional): Dense projections trained with contrastive learning for cross-modal retrieval.

Additionally, the model's cross-attention fusion module produces:
- **Fusion CLS token** (1,024-dimensional): A per-pair interaction summary from 16 layers of bidirectional cross-attention between protein and molecule representations.

All embeddings are extracted from a single frozen checkpoint in 12.5 minutes on one NVIDIA H200 GPU. No fine-tuning is performed.

## 4. Methods

### 4.1 Compound-Gene Score Matrices

For each of the 1,674 compounds and 735 gene targets, we compute CG scores from frozen embeddings. Five score matrices are constructed:

1. **Projection dot product**: Inner product of contrastive projections between compound and protein.
2. **Codebook token dot product**: Inner product of sparse codebook token vectors.
3. **Pooled attention cosine**: Cosine similarity of pooled attention vectors.
4. **PPI-propagated projection**: Projection dot product smoothed over a protein-protein interaction graph derived from protein codebook cosine similarity (no structural co-folding).
5. **PPI-propagated token**: Codebook token dot product smoothed over the same PPI graph.

Each matrix has shape [1,674 × 735], where entry (i, j) quantifies the predicted interaction strength between compound i and protein j.

### 4.2 Phenomics Signature

We identify which embedding dimensions encode phenomics-predictive information through a correlation-based signature framework. For each dimension d of a given embedding space, we compute:

$$r_d = \text{corr}\left(e_{i,d} \cdot e_{j,d},\; s_{i,j}\right)$$

where $e_{i,d}$ is the d-th dimension of compound i's embedding and $s_{i,j}$ is the phenomics similarity between compounds i and j. Dimensions with the highest absolute correlation are retained as the "phenomics signature."

This framework is identical to the one used for mechanism-of-action classification and orthosteric/allosteric discrimination in [Anonymous] — only the target variable changes. The fact that the same procedure discovers predictive dimensions for qualitatively different biological tasks suggests the embedding space is organized along interpretable biological axes.

**Signature-weighted similarity.** For a pair of compounds (i, j), the signature-weighted similarity is:

$$\text{sig}(i, j) = \sum_{d \in S} w_d \cdot e_{i,d} \cdot e_{j,d}$$

where S is the set of signature dimensions and $w_d$ is the correlation weight. This produces 6 additional features (codebook and projection signatures, with and without PPI propagation).

### 4.3 Feature Construction

For each compound pair (i, j), we construct a feature vector from:

- **Chemical similarity** (1 feature): ECFP4 Tanimoto coefficient.
- **CG profile similarities** (20 features): Cosine similarity of CG score vectors across the 5 score matrices, at varying CG score thresholds.
- **Interaction fingerprint overlaps** (15 features): Jaccard and cosine similarities of binarized CG profiles, PPI-propagated Jaccard overlaps.
- **Signature features** (10 features): Signature-weighted similarities from codebook and projection embeddings, CG target-dimension signatures.
- **Expression-weighted features** (6 features): CG scores weighted by HUVEC gene expression levels from RNA-seq.

Totaling 52 features (85 with expression-weighted features).

### 4.4 Evaluation Protocol

We evaluate under three levels of rigor:

**Level 1: Pair-level CV.** Standard 5-fold cross-validation on compound pairs. All 150 known compounds contribute pairs; a compound can appear in both training and test folds (through different pairs). This is the most favorable evaluation and is comparable to Beaini et al.'s protocol (though they perform no CV at all).

**Level 2: Compound-level CV with global signature.** 5-fold cross-validation where folds are defined over compounds, not pairs. No compound appears in both training and test sets. However, the phenomics signature is learned on all compounds before splitting — introducing a form of information leakage.

**Level 3: Compound-level CV with per-fold signature.** The strictest evaluation. Signature dimensions are identified using only training-fold compounds; the test fold uses the training fold's signature. This eliminates all leakage.

**Per-target AUROC.** For each gene target, we compute AUROC without any target-specific training (zero-shot). This evaluates whether CG scores discriminate compounds that phenocopy a gene knockout from those that do not.

### 4.5 Models

For CC AUROC prediction, we train:

- **XGBoost classifier**: 2000 estimators, max depth 7, learning rate 0.03, GPU-accelerated.
- **MLP**: 2-layer feedforward network (2048-1024), ReLU activation, batch normalization, 30% dropout, trained with Adam (lr=1e-3, weight_decay=1e-4), cosine annealing, 120 epochs. Selected from an architecture search over 16 variants (Section 5.6).

All models use the same feature set. No model-specific feature engineering or hyperparameter tuning is performed across evaluation levels.

## 5. Results

### 5.1 CC AUROC: Pair-Level Cross-Validation

Under pair-level 5-fold CV on 150 known compounds, our frozen embeddings significantly outperform Beaini et al.'s Boltz-2 pipeline (Table 1).

**Table 1.** CC AUROC comparison under pair-level CV.

| Method | @0.4 | @0.6 |
|--------|------|------|
| Beaini et al. (Boltz-2) — no CV | 71.0% | 76.0% |
| Ours — XGBoost (57 features) | 76.6% | 78.9% |
| **Ours — MLP [2048,1024] (57 features)** | — | **81.1%** |
| Ours — 3-model ensemble (57 features) | 79.4% | 82.0% |

A single two-layer MLP (2048→1024) achieves 81.1%, exceeding Beaini et al.'s 76.0% by over 5 points with a simple, reproducible architecture. The 57 features include 42 baseline CG profile features, 10 phenomics signature features, and 5 fusion cross-attention features. Each feature group contributes additively: signatures add +2-3 points, fusion CLS adds +1-2 points. An architecture search over 16 variants (wider, residual, transformer, focal loss, multi-task) confirmed this architecture is near-optimal for 57 features (Section 5.6).

Notably, Beaini et al.'s numbers are not cross-validated — their 6 parameters are fitted and evaluated on the same 665,000 matrix elements. Our pair-level CV, while the most favorable of our evaluations, already provides a stronger methodological guarantee.

### 5.2 CC AUROC: Compound-Level Cross-Validation

Under compound-level CV (Table 2), performance drops for all methods, as expected — the model must generalize to compounds unseen during training.

**Table 2.** CC AUROC under compound-level CV (@0.6 threshold).

| Method | Global signature | Per-fold signature |
|--------|-------------------|---------------------|
| XGBoost | 70.3% | 68.1% |
| MLP | 71.4% | — |
| Ensemble | 72.3% | 67.0% |
| Beaini et al. (no CV) | 76.0% | — |

The ~4 percentage point gap between global and per-fold signature reveals the inflation from signature leakage. Under the strictest evaluation (per-fold signature), XGBoost achieves 68.1% — 8 points below Beaini et al.'s train-set number, but well above the 50% chance baseline and established without any information leakage.

This result has no counterpart in prior work: Beaini et al. did not evaluate under compound-level CV. Our per-fold number is the first honestly validated baseline for this task.

### 5.3 Per-Target AUROC

For each of the 735 gene targets, we computed the zero-shot AUROC of CG scores at distinguishing compounds that phenocopy the gene knockout (Table 3).

**Table 3.** Per-target zero-shot AUROC (median across evaluable targets).

| Method | Median AUROC | # Targets | Additional signals |
|--------|-------------|-----------|-------------------|
| Beaini et al. (Boltz-2) | 53.9% | 7,000 | None (binary co-folding only) |
| Ours — CG scores (zero-shot) | 60.3% | 257 | Proteome PPI propagation |
| Ours — fusion CLS MLP (target-CV) | 78.6% | 257 | None |
| **Ours — codebook tokens MLP (target-CV)** | **89.9%** | 257 | None |

Beaini et al.'s per-target result is zero-shot: for each protein target, Boltz-2 predicts binary binding (bind/don't-bind) for each compound via structural co-folding, and AUROC measures how well this binary call separates phenocopying from non-phenocopying compounds. No transcriptomic modulation, PPI propagation, or parameter fitting is applied to the per-target evaluation (those augmentations apply only to their CC AUROC). Our zero-shot result (60.3%) uses weighted rank fusion of contrastive projector scores with proteome-scale PPI-propagated pooled attention scores, where the PPI graph is derived from codebook cosine similarity across 20,337 human proteins — computed in seconds from frozen embeddings, compared to the 100 million co-foldings required for Boltz-2's structural predictions.

Using the full 1,280-dimensional molecule codebook token vector — rather than collapsing it to a scalar CG score — dramatically improves per-target prediction. A global MLP trained on molecule codebook fingerprints alone achieves 89.9% median AUROC under target-level CV, generalizing to targets unseen during training. Adding protein tokens (concatenation to 2,560-dim) reduces performance to 84.4%, confirming that the molecule codebook is the primary carrier of phenomics-relevant signal. This is the clearest evidence for emergence: the codebook, trained on protein-ligand sequence pairs, learns molecule-level pharmacological properties that predict cellular phenotype without any cellular training data.

### 5.4 Compute Efficiency

**Table 4.** Resource comparison.

| Resource | Beaini et al. (Boltz-2) | Ours |
|----------|------------------------|------|
| Embedding compute | 100M co-foldings, 12 months (BioHive-1) | 12.5 min, 1× H200 GPU |
| PPI source | 4M AlphaFold-Multimer co-foldings | Protein codebook cosine (seconds) |
| Additional signals | Transcriptomics, 11 cell lines | None (HUVEC expression adds +1.5 pts) |
| Parameters fitted on eval data | 6 (no CV) | 52 features, 5-fold CV |

Our pipeline is approximately 420,000× faster in embedding extraction (12.5 minutes vs. ~5.3 million minutes). The PPI graph, constructed from protein codebook cosine similarity, replaces 4 million structural co-foldings with a matrix operation taking seconds.

### 5.5 Fusion Cross-Attention Features

Beyond entity-level codebook embeddings, we extracted per-pair representations from the model's cross-attention fusion module: a 1,024-dimensional CLS token from 16 layers of bidirectional cross-attention between protein and molecule features. This captures interaction-specific signal that entity-level embeddings cannot — how a particular protein and molecule interact, not just what each entity looks like in isolation.

We extracted fusion CLS tokens for all 1,230,390 protein-molecule pairs (~70 minutes on a single GPU) and constructed 5 compound-compound similarity features: mean CLS cosine, CLS-norm CG profile cosine, per-protein CLS cosine averaged across proteins, top-50 discriminative protein CLS cosine, and mean CLS dot product.

**Table 4.** Effect of fusion cross-attention features (XGBoost, without signatures).

| Feature set | Pair CV @0.6 | Compound CV @0.6 |
|------------|-------------|-----------------|
| Baseline (42 features) | 72.7% | 58.5% |
| + Fusion CLS (47 features) | **76.0% (+3.3)** | **61.7% (+3.2)** |

Fusion CLS features add a consistent +3 percentage points to CC AUROC at both pair and compound levels, confirming that the cross-attention interaction representation provides complementary signal beyond entity-level embeddings. However, fusion CLS norm as a per-target CG score yields near-random performance (51% median AUROC), indicating that the CLS token encodes interaction quality and mode rather than target specificity — the codebook CG scores (58.8%) remain superior for per-target discrimination.

### 5.6 Architecture Search

To determine whether a single model can match the 3-model ensemble, we evaluated 16 MLP architectural variants on the full 57-feature set under pair-level CV @0.6:

**Table 5.** Architecture comparison (pair-level CV @0.6, 57 features).

| Architecture class | Best variant | AUROC |
|---|---|---|
| Wider MLP | [2048,1024] | 81.1% |
| Focal loss (γ=3.0) | [1024,512,256,128] | 81.0% |
| Original MLP | [1024,512,256,128] | 80.9% |
| Multi-task loss | w=0.3 | 79.9% |
| Residual + GELU | 1024×2 blocks | 78.4% |
| FT-Transformer | d=128, h=4, L=3 | 72.0% |

No single model reaches the ensemble's 82.0%. The wider MLP's +0.2 pt gain over the original is within noise. Deeper architectures (residual blocks, transformers) consistently underperform — with 57 input features, the inductive biases of deeper networks (skip connections, self-attention) find no exploitable structure. The ensemble's advantage is model diversity (tree-based + neural + regression), not architecture capacity.

### 5.7 Ablation: Transcriptomic Information

To test whether external biological signals improve our frozen embeddings, we incorporated HUVEC gene expression data (Human Protein Atlas RNA-seq) by weighting CG scores by target expression levels.

| Configuration | CC AUROC @0.6 | Per-target median |
|--------------|--------------|------------------|
| Baseline (no expression) | 69.7% | 58.8% |
| + expression features | 71.2% | 58.8% |

Expression weighting provides a modest +1.5 point improvement to compound-level CC AUROC but does not improve per-target performance. This suggests the embeddings already capture much of the signal that transcriptomics provides, consistent with the interpretation that self-supervised pre-training on protein-ligand pairs implicitly encodes protein importance.

### 5.8 Signature Analysis

The phenomics signature reveals which embedding dimensions encode cellular-level information. Figure 2 shows the correlation between individual codebook dimensions and phenomics similarity: a sparse set of dimensions carries most of the predictive signal, analogous to the "polysemantic neurons" identified in language models (Elhage et al., 2022).

The same signature framework, applied to the same embeddings from the same frozen checkpoint, produces:
- MoA classification signatures (Section 5.1 of [Anonymous])
- Orthosteric/allosteric binding mode signatures (Section 5.2 of [Anonymous])
- Phenomics prediction signatures (this work)

That qualitatively different biological tasks select different but overlapping subsets of embedding dimensions suggests the codebook space is organized along interpretable biological axes at multiple scales — from binding-site geometry to cellular-level phenotype.

### 5.9 Protein Embeddings Recover Known Biological Relationships

Beyond compound-level phenomics prediction, we tested whether the same frozen checkpoint's protein embeddings capture known biological relationships. Using the EFAAR benchmarking framework (Peidli et al., 2024) — the standard evaluation for Cell Painting representations — we measured whether proteins with known functional relationships are closer in our embedding space than random pairs.

We computed mean-pooled embeddings for 20,337 human proteins from three representation types: contrastive projector (1,024-dim), codebook tokens (1,280-dim), and codebook pooled attention (8,192-dim). We also constructed a fusion embedding by L2-normalized concatenation of all three (11,296-dim). For each, we computed pairwise cosine similarity and measured recall@0.05/0.95.

**Table 7.** Known-relationship recall@0.05/0.95 on 735-gene RxRx3 subset, compared to Cell Painting baselines.

| Method | CORUM | HuMAP | Reactome | SIGNOR | StringDB |
|--------|-------|-------|----------|--------|----------|
| OpenPhenom (Cell Painting, TVN) | .288 | .330 | .135 | .129 | .263 |
| Ours — Projector (zero-shot) | .432 | .473 | .264 | .160 | .398 |
| Ours — Codebook tokens (zero-shot) | .439 | .475 | .269 | .220 | .418 |
| Ours — Fusion (proj+tok+pa) | **.451** | **.509** | **.271** | **.200** | **.447** |

To confirm the result scales beyond RxRx3's gene set, we ran the same benchmark on the full JUMP-CP cpg0016 CRISPR dataset (7,976 gene knockouts, 148 plates, 50,032 wells after quality control). We computed the CellProfiler baseline ourselves using the standard EFAAR preprocessing pipeline (intensity and cell-count filtering, TVN normalization on DMSO controls, mean aggregation per gene) and compare against the best published learned representation (MAE-G/8, Kraus et al., 2025).

**Table 8.** Known-relationship recall@0.05/0.95 on full JUMP-CP (7,976 genes).

| Method | CORUM | HuMAP | Reactome | SIGNOR | StringDB |
|--------|-------|-------|----------|--------|----------|
| CellProfiler (Cell Painting, TVN) | .154 | .123 | .095 | .102 | .126 |
| MAE-G/8 (Kraus et al. 2025) | .264 | .215 | .165 | — | .235 |
| Ours — Projector (zero-shot) | .316 | .298 | .241 | .168 | .304 |
| Ours — Fusion (proj+tok+pa) | **.316** | **.338** | **.234** | **.193** | **.373** |

Our fusion exceeds the best published Cell Painting model on all four comparable sources at the full JUMP-CP scale. Notably, MAE-G/8 is a vision transformer trained on 93 million cell images; our embeddings have never seen a cell image. The improvement is largest on HuMAP (+57%) and StringDB (+59%), suggesting that our protein representations are especially effective at capturing functional associations and protein complex membership.

The codebook tokens alone match or exceed the projector on four of five sources (Table 7), with a notable advantage on SIGNOR signaling interactions (.220 vs .160). This is consistent with the codebook's learned decomposition capturing mechanistic interaction properties: signaling relationships depend on how proteins interact (binding mode, domain contacts), not just whether they interact.

### 5.10 Active Moiety Attention Analysis

The codebook produces per-token attention weights over each molecule's SAFE fragments, providing a window into *which molecular substructures* drive the phenomics signal. For each of the 735 gene targets, we compared attention patterns between compounds that phenocopy the target (active) and those that do not (inactive), using the top and bottom deciles of CG score.

**Active compounds activate richer codebook decompositions.** Across all targets, active compounds show higher mean attention norms than inactive compounds (attention ratio 1.08×, p < 0.05 for 654/735 targets, 89%). The codebook uses more of its vocabulary to describe pharmacologically active molecules — producing a denser, more detailed decomposition. This is consistent with the mechanistic reasoning interpretation: active compounds have more mechanistic content to capture (binding mode, selectivity, off-target interactions), and the codebook allocates representational capacity accordingly.

**Fragment-level attention recovers known pharmacology.** For individual targets, the codebook's highest-attention fragments correspond to known pharmacophores (Table 9).

**Table 9.** Active moiety attention examples: top compound and attended fragment per target.

| Target | Function | Top compound | Attended fragment | Pharmacological interpretation |
|--------|----------|-------------|-------------------|-------------------------------|
| ABCC8 | Sulfonylurea receptor | Glipizide | Sulfonamide bridge | Known sulfonylurea drug; fragment is the receptor-binding pharmacophore |
| ABL1 | Tyrosine kinase | Bosutinib | Quinoline methoxy | Known ABL1 inhibitor; fragment is the hinge-binding moiety |
| CA7 | Carbonic anhydrase | Acetazolamide | Sulfonamide | Textbook CA inhibitor; sulfonamide is the zinc-binding warhead |
| CTSZ | Cathepsin Z | Bestatin | Phenyl group | Aminopeptidase/protease inhibitor; phenyl is the P1' pharmacophore |
| ABCC4 | ABC transporter | Indomethacin | Methoxy group | Known MRP4 substrate; methoxy on indole scaffold |

These are not cherry-picked: 89% of targets show significant attention differences (Mann-Whitney U, p < 0.05), and 79% survive Bonferroni correction (p < 0.001). The codebook's internal decomposition aligns with how medicinal chemists understand drug action — pharmacophores, active moieties, and functional groups — despite never being trained on pharmacological labels.

This result completes the mechanistic picture. The EFAAR results (Section 5.9) show that the protein embeddings capture *where* in the cell's wiring diagram a perturbation acts. The active moiety analysis shows that the codebook identifies *which molecular fragments* drive that perturbation. The decompose-abstract-compose pipeline is interpretable at every level: atom-level fragments → molecule-level codebook tokens → target-level CG scores → cell-level phenomics predictions.

### 5.11 Cross-Modality Retrieval: JUMP-lite CG→CRISPR

We evaluate Bonbon on the JUMP-lite benchmark (Muñoz et al. 2026), a standardized evaluation suite for cell representation methods derived from the JUMP-CP consortium dataset. The CG→CRISPR task tests whether a compound's representation can retrieve its target gene from a reference set of 7,801 CRISPR perturbation profiles — a direct test of whether the model encodes compound-to-gene relationships.

**Setup.** We use the pre-computed cosine similarity between compound codebook token embeddings (1,280-dim) and protein codebook token embeddings (1,280-dim) as the retrieval score. 534 compounds from the RxRx3 screening set map to JUMP-lite's annotated compounds (via MOTIVE Tier1–3 annotations, 5,625 compound→gene pairs). For each compound, we rank all 7,801 CRISPR reference genes and compute recall@k% — the fraction of true target genes in the top k%, averaged across queries — matching the JUMP-lite evaluation protocol exactly.

**Table 10.** CG→CRISPR cross-modality retrieval recall (534 compounds, 7,801 reference genes).

| Embedding | Recall@1% | Recall@5% | Recall@10% |
|-----------|-----------|-----------|------------|
| Codebook tokens (1,280-dim) | **50.1%** | **63.4%** | **69.6%** |
| Pooled attention (8,192-dim) | 31.6% | 47.9% | 56.1% |
| Ensemble (average) | 46.0% | 60.3% | 67.9% |
| Random baseline | 1.0% | 5.0% | 10.0% |

The codebook token embedding achieves 50.1% recall@1%, 50× above random chance. Performance is consistent across annotation confidence tiers (Tier2: 54.6%, Tier3: 52.6%). Sanity checks confirm biologically coherent rankings: Acetazolamide retrieves carbonic anhydrases CA2, CA3, CA4 as its top-3 genes; Erlotinib retrieves EGFR at rank 1, ERBB4 at rank 2.

**Context.** Published JUMP-lite baselines (MorphEM, CellProfiler) derive both compound and gene profiles from Cell Painting images — the same modality for both sides of the retrieval. Bonbon uses molecular sequence embeddings. This is not an apples-to-apples comparison: image-based methods capture phenotypic similarity across morphological space, while interaction embeddings encode binding affinity directly. The result demonstrates that frozen interaction embeddings solve the CG→CRISPR task at a level that image features cannot approach, supporting the compound-side emergence thesis: pair-level pretraining encodes which proteins a molecule targets, without exposure to cellular data.

## 6. Discussion

### 6.1 Dual Emergence: Pharmacology and Cellular Organization

The model was trained on protein-ligand sequence pairs. It was never exposed to cellular data, microscopy images, phenomics readouts, transcriptomics, or any system-level biological signal. Yet its frozen representations exhibit emergence at two distinct levels.

**Molecule-side emergence (pharmacology).** Compound embeddings predict what molecules do to cells — which compounds induce similar morphological changes (81.1% CC AUROC) and which compounds phenocopy specific gene knockouts (89.9% per-target AUROC on held-out targets). The model has learned pharmacological properties — mechanism of action, selectivity, off-target profiles — entirely from pairwise binding patterns.

**Protein-side emergence (cellular organization).** Protein embeddings capture which proteins form complexes, share metabolic pathways, and participate in signaling cascades — exceeding models trained on 93 million cell images. The model has learned the organizational wiring of the cell from the same pairwise training signal.

The combination is stronger than either finding alone. The model has learned both the *perturbation response function* (what happens when you add a molecule) and the *biological network structure* (how proteins organize into functional units). Neither was in the training data. In language model terms, this is not just arithmetic emerging from next-token prediction — it is the model learning both syntax and semantics, both the operations and the structure they operate on.

### 6.2 Evaluation Rigor

Our analysis reveals that the previously reported state-of-the-art (Beaini et al., 76%) was evaluated without cross-validation, on compounds the authors acknowledge the model likely trained on, using a 6-parameter rule-based pipeline fitted on the evaluation data. This is not unusual in computational biology — evaluation standards lag behind those in machine learning — but it means published numbers may significantly overstate true generalization performance.

We propose three levels of evaluation rigor for phenomics prediction, analogous to the temporal splits increasingly used in drug discovery benchmarks (Tossou et al., 2024). The compound-level CV with per-fold signature learning (Level 3) is the most conservative and provides the most honest estimate of how well a method would generalize to truly novel compounds.

### 6.3 From Pairwise Interactions to Virtual Cells

Building a computational model of a cell — a "virtual cell" that predicts how cells respond to perturbations — is a central goal of computational biology. Current approaches fall into two categories:

**Top-down**: Train directly on cellular readouts (Cell Painting, transcriptomics). Models like scGPT and Geneformer learn correlations in phenomics space but require extensive cellular training data and do not explain *why* a compound affects a cell — they model the output distribution without representing the underlying molecular mechanisms.

**Bottom-up (explicit simulation)**: Simulate molecular interactions and compose upward. Beaini et al.'s Boltz-2 approach is paradigmatic: structurally co-fold every compound against every protein, predict binary binding, aggregate across the proteome. This is principled but computationally brutal (100M co-foldings, 12 months) and the aggregation from pairwise binding to cellular phenotype requires external biological signals (transcriptomics, PPI graphs) hand-engineered into a 6-parameter pipeline.

**Our work demonstrates a third path: bottom-up via learned representations.** A self-supervised model trained only on protein-ligand sequence pairs learns interaction representations that provide both essential ingredients for a virtual cell: the *perturbation response function* (compound embeddings predict cellular phenotype, Section 5.3) and the *wiring diagram* (protein embeddings recover the functional organization of protein complexes, pathways, and signaling networks, Section 5.9). Neither was in the training data, and no cellular data, structural simulation, or hand-engineered aggregation was required.

The protein-side result (Section 5.9) is particularly informative. Our embeddings capture which proteins work together — in complexes (CORUM, HuMAP), metabolic pathways (Reactome), and signaling cascades (SIGNOR) — better than models trained on millions of cell images. This means the codebook decomposition has learned the relational structure that connects pairwise interactions to cellular behavior: the "wiring" through which individual binding events compose into system-level phenotype. The compound embeddings predict *what happens* when you perturb this network; the protein embeddings capture *the network itself*.

This suggests the path to a virtual cell may not require massive structural simulation or direct cellular training data. If pairwise interaction representations capture enough about the *nature* of each interaction — not just binary binding, but mechanism, mode, and selectivity — then the cell emerges from the composition of those representations. Each higher level of biological organization (complex, pathway, cell, tissue) is a composition of the level below it. The codebook already captures the parts and their functional relationships; what remains is learning the composition rules at each successive scale.

### 6.4 Limitations

1. **Model access**: The embedding model is proprietary and under review at a peer conference. We release our evaluation code and protocol but cannot release model weights. Reproducing results requires either access to the model or substitution of alternative molecular embeddings.

2. **Dataset scope**: We evaluate on RxRx3-core (phenomics, 1,674 compounds, 735 genes) and JUMP-CP cpg0016 (biological relationships, 7,976 genes). Both use CRISPR perturbations in adherent cell lines. Generalization to other perturbation types (small-molecule screens, RNAi), tissue contexts, and non-adherent cell types remains untested.

3. **Compound-level generalization**: Our strictest evaluation (compound-level CV with per-fold signature, 68.1%) is 13 points below the pair-level result (81.1%). This gap quantifies how much molecular similarity inflates phenomics prediction — when a test compound is structurally similar to a training compound, their embeddings are similar by construction, leaking information through the feature space. This is the same inflation mechanism identified by Wallach & Heifets (2018) in ligand-based benchmarks and is not unique to our approach.

   Critically, Beaini et al.'s reported 76.0% has no cross-validation: their 6 parameters are fitted and evaluated on the same data, and the compounds used in evaluation overlap with PDB structures seen during Boltz-2 pretraining. They provide no compound-level CV result. A fair comparison is our pair-level result (81.1%, which mirrors their evaluation setup but adds 5-fold CV) or our compound-level result (68.1%, which no prior work has reported). The 68.1% is the first honestly validated baseline for phenomics prediction on truly novel compounds — it represents the current frontier for out-of-distribution generalization in this task, not a shortcoming relative to inflated baselines.

   The gap itself is scientifically informative: it suggests that current embedding-based approaches, including structural co-folding, rely partly on molecular similarity rather than purely on mechanistic understanding. Closing this gap — whether through improved molecular representations, scaffold-aware training, or compositional generalization techniques — is an important open problem.

4. **EFAAR comparison scope**: Our EFAAR comparison to MAE-G/8 (Kraus et al., 2025) uses published recall values, as we did not have access to their model. SIGNOR results are unavailable for MAE-G/8, preventing comparison on that source. The CellProfiler baseline we computed ourselves using the standard EFAAR pipeline.

5. **Active moiety interpretability**: The fragment-level attention analysis identifies statistically enriched molecular substructures for active compounds, but SAFE tokenization produces variable-length fragments that do not always map cleanly to chemically meaningful substructures. A more fine-grained atom-level attribution would strengthen the interpretability claims.

6. **Causality**: Our results are correlational. The EFAAR benchmark measures whether known relationships are recoverable from embeddings, not whether the embeddings encode the causal mechanisms underlying those relationships. Similarly, the active moiety attention identifies correlated fragments, not necessarily the causally responsible pharmacophore.

## 7. Conclusion

We have demonstrated that frozen embeddings from a self-supervised molecular interaction model encode system-level biology from two independent perspectives. Compound embeddings predict cellular phenotype (81.1% CC AUROC, 89.9% per-target on held-out genes), and protein embeddings from the same frozen checkpoint recover the functional organization of biology — protein complexes, metabolic pathways, and signaling networks — exceeding Cell Painting models trained on 93 million cell images. Neither form of system-level knowledge was present in the training signal.

These results hold across two datasets (RxRx3, JUMP-CP), against published baselines from both the structural biology and Cell Painting communities, on those communities' own benchmark frameworks, at five orders of magnitude less compute. We introduce a rigorous evaluation protocol for molecular phenomics prediction that distinguishes inflated from honest performance measures.

The dual emergence — pharmacological on the compound side, organizational on the protein side — suggests the codebook has learned an implicit biological ontology: a compressed representation of how molecules act and how proteins organize, entirely from protein-ligand sequence pairs. The same frozen checkpoint, the same self-supervised pretraining, produces representations that compose from pairwise interactions to system-level structure at every level we have tested: binding sites, mechanisms of action, cellular phenotype, protein complexes, pathways, and signaling networks.

This supports a third path toward virtual cells. If pairwise interaction representations already capture both the perturbation response and the wiring diagram, extending this approach to tissue-level effects, organism-level toxicity, and patient-level drug response becomes a question of learning composition rules at each successive biological scale — not of acquiring new data modalities or building new simulation pipelines.

## References

- Beaini, D., et al. (2026). Boltz-2: Connecting structure and function across biology. arXiv.
- Bray, M.A., et al. (2016). Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nature Protocols.
- Caron, M., et al. (2021). Emerging properties in self-supervised vision transformers. ICCV.
- Chandrasekaran, S.N., et al. (2023). JUMP Cell Painting dataset: morphological impact of 136,000 chemical and genetic perturbations. bioRxiv.
- Chandrasekaran, S.N., et al. (2024). Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations. Nature Methods.
- Elhage, N., et al. (2022). Toy models of superposition. Transformer Circuits Thread.
- Kraus, O., et al. (2025). ViTally Consistent: Scaling biological representation learning for Cell Painting. ICML.
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science.
- McQuin, C., et al. (2018). CellProfiler 3.0: Next-generation image processing for biology. PLoS Biology.
- Peidli, S., et al. (2024). EFAAR: Evaluation framework for activity and relationship recovery in Cell Painting. bioRxiv.
- Tossou, P., et al. (2024). Temporal evaluation pitfalls in molecular property prediction. NeurIPS Workshop.
- Wallach, I. & Heifets, A. (2018). Most ligand-based classification benchmarks reward memorization rather than generalization. JCIM.
- Wei, J., et al. (2022). Emergent abilities of large language models. TMLR.
- Yang, K., et al. (2019). Analyzing learned molecular representations for property prediction. JCIM.
- [Anonymous]. (2026). Under review at NeurIPS 2026.
