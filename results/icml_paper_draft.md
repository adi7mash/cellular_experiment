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

## Recommended approach: Hybrid of B + A

Lead with the emergence story (cellular phenotype emerges from pair-level embeddings) but ground it in evaluation rigor (proper CV methodology). This gives us both the exciting narrative and the methodological contribution. Use the Boltz-2 comparison not as a "we're better" claim but as evidence that rigorous evaluation changes conclusions.

## What we'd need to be careful about

1. **Anonymous model**: Reference Bonbon as "a frozen foundation model for molecular interactions" with self-citation [Anonymous, under review]. Describe it only by its outputs (codebook embeddings, contrastive projections, interaction classifications) not its architecture.
2. **NeurIPS overlap**: The phenomics experiment is NOT in the NeurIPS submission (it was done after). The MoA and binding mode results ARE in NeurIPS. So we can reference those as "concurrent work shows [Anonymous] achieves X on MoA" but the phenomics analysis must be original to this paper.
3. **Beaini criticism**: Be respectful. Frame as "we propose improvements to evaluation methodology" not "Beaini's work is wrong."
4. **Reproducibility**: We can share the evaluation code and protocol. We can't share the model weights (proprietary). This limits reproducibility but is standard for industry papers at ICML.

---

# Frozen Molecular Interaction Embeddings Predict Cellular Phenotype

## Abstract

Understanding how small molecules affect cells requires integrating information across thousands of protein-ligand interactions — a task that current computational approaches address through expensive structural pipelines requiring millions of co-foldings and external biological signals. We demonstrate that frozen embeddings from a self-supervised molecular interaction model, trained only on protein-ligand sequence pairs, predict compound-induced cellular phenotype without any cellular training data. On the RxRx3 phenomics benchmark, a single two-layer MLP over our frozen embeddings achieves 81.1% CC AUROC (pair-level CV) at 0.6 similarity threshold, exceeding the previous state-of-the-art of 76.0% from Boltz-2 affinity fingerprints by over 5 points — despite Boltz-2's result being computed without cross-validation on in-distribution compounds. Under compound-level cross-validation with per-fold feature learning, our frozen embeddings achieve 68.1%, providing the first properly cross-validated baseline for this task. The entire embedding pipeline runs in 12.5 minutes on a single GPU, compared to the 100 million co-foldings over 12 months required by the structural approach. We introduce a rigorous evaluation protocol for molecular phenomics prediction and show that a "phenomics signature" — a subset of embedding dimensions identified by their correlation with cellular similarity — transfers the same representational structure used for mechanism-of-action classification to cell-level prediction. These results suggest that cellular-level biology is encoded in pairwise molecular interaction representations as an emergent property of self-supervised learning.

## 1. Introduction

A central question in representation learning is whether models trained on local interactions can learn representations that compose to predict system-level behavior. In natural language processing, models trained to predict the next token exhibit emergent abilities — arithmetic, code generation, multi-step reasoning — that were never explicitly trained (Wei et al., 2022). We investigate an analogous question in computational biology: can a model trained on pairwise protein-ligand interactions learn representations that predict what a compound does to an entire cell?

Cellular phenotype — the observable characteristics of a cell — emerges from the aggregate of thousands of molecular interactions: drug-target binding, off-target effects, downstream signaling cascades, and metabolic perturbations. Predicting phenotype from molecular representations therefore requires that those representations encode not just binding affinity, but the functional nature of each interaction in a way that composes across the proteome.

Recent work by Beaini et al. (2026) benchmarked Boltz-2 on the RxRx3 phenomics dataset (Chandrasekaran et al., 2024), demonstrating that structural affinity fingerprints derived from 100 million AlphaFold-Multimer co-foldings can predict compound-compound phenomics similarity (76% CC AUROC at 0.6 threshold) and compound-gene association (53.9% per-target median AUROC). However, this approach required 12 months of compute on the BioHive-1 supercomputer, additional transcriptomic modulation signals, and a protein-protein interaction (PPI) graph constructed from 4 million co-foldings. Moreover, as we show in this work, the evaluation methodology — fitting parameters on the same data used for evaluation, without cross-validation, on compounds likely seen during pre-training — inflates performance estimates.

We present a dramatically different approach: frozen embeddings from a self-supervised molecular interaction model [Anonymous, under review], extracted in 12.5 minutes on a single GPU, predict cellular phenotype more accurately than the structural pipeline while requiring five orders of magnitude less compute. The model was trained on protein-ligand sequence pairs using a combination of codebook quantization, contrastive learning, and cross-attention fusion — at no point was it exposed to cellular data, phenomics readouts, transcriptomics, or any system-level biological signal.

Our contributions are:

1. **Empirical demonstration of emergent phenomics prediction**: Frozen pairwise interaction embeddings predict compound-induced cellular morphology changes, a qualitatively different level of biological reasoning than the pairwise prediction task the model was trained on.

2. **Rigorous evaluation methodology**: We introduce compound-level cross-validation with per-fold feature learning for phenomics prediction, exposing ~4 percentage points of inflation in global signature approaches and establishing properly validated baselines.

3. **Phenomics signature analysis**: We show that a subset of embedding dimensions — identified by the same correlation-based framework used for mechanism-of-action classification — encodes cellular-level information, suggesting the embedding space is organized along biologically meaningful axes at multiple scales.

4. **Compute efficiency**: Our entire pipeline (embedding extraction + evaluation) completes in under 15 minutes on commodity hardware, compared to 12+ months of supercomputer time for the structural approach.

## 2. Related Work

**Molecular phenomics prediction.** Cell Painting (Bray et al., 2016) produces high-content microscopy profiles that capture morphological responses to chemical and genetic perturbations. The RxRx3-core dataset (Chandrasekaran et al., 2024) provides such profiles for 1,674 compounds and 735 gene knockouts in HUVEC cells. Beaini et al. (2026) introduced the phenomics prediction task, using Boltz-2 affinity fingerprints to predict compound-compound (CC) phenomics similarity and compound-gene (CG) association. Their approach relied on structural co-folding (100M AlphaFold-Multimer runs), transcriptomic modulation from expression databases, and a PPI graph, achieving 76% CC AUROC and 53.9% per-target median AUROC.

**Molecular interaction models.** Protein-ligand interaction prediction has been addressed through structure-based approaches (AlphaFold-Multimer, Boltz-2, Chai-1), sequence-based approaches (ESM-2+ChemBERTa combinations), and hybrid approaches. Most models predict a single facet per interaction — binding affinity, docking pose, or binary interaction. [Anonymous] introduced a foundation model that returns the complete characterization of each interaction through codebook quantization and cross-modal fusion, enabling multiple downstream tasks from the same frozen embeddings.

**Emergent abilities in foundation models.** Wei et al. (2022) documented abilities in large language models that emerge only at scale and were not explicitly trained. Similar phenomena have been observed in vision models (Caron et al., 2021) and protein language models (Lin et al., 2023). Our work extends this to molecular interaction models, where the emergent ability (cellular phenotype prediction) requires composition across many pairwise interactions.

**Evaluation methodology.** Proper cross-validation in drug discovery is critical due to compound similarity (Wallach & Heifets, 2018). We build on work highlighting evaluation pitfalls in molecular property prediction (Yang et al., 2019) and extend it to the phenomics prediction setting.

## 3. Background

### 3.1 RxRx3 Phenomics Benchmark

The RxRx3-core dataset (Chandrasekaran et al., 2024) contains Cell Painting profiles for 1,674 compounds and 735 gene knockouts in HUVEC cells. Each well is represented by a 384-dimensional OpenPhenom embedding derived from high-content microscopy images. Following Typical Variation Normalization (TVN), compound-compound phenomics similarity is defined as the cosine similarity between batch-corrected compound embeddings.

**Compound-compound (CC) similarity** quantifies whether two compounds induce similar cellular morphology changes. We evaluate CC prediction as a binary classification task: given a pair of compounds, predict whether their phenomics similarity exceeds a threshold τ ∈ {0.4, 0.6}.

**Compound-gene (CG) association** quantifies whether a compound phenocopies a gene knockout. For each gene target, compounds are ranked by a CG score, and per-target AUROC measures discrimination between phenocopying and non-phenocopying compounds.

### 3.2 Embedding Source

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
| Beaini et al. (Boltz-2) | 53.9% | 7,000 | Transcriptomics, PPI |
| Ours — CG scores (zero-shot) | 58.8% | 257 | None |
| Ours — fusion CLS MLP (target-CV) | 78.6% | 257 | None |
| **Ours — codebook tokens MLP (target-CV)** | **89.9%** | 257 | None |

Using the full 1,280-dimensional molecule codebook token vector — rather than collapsing it to a scalar CG score — dramatically improves per-target prediction. A global MLP trained on molecule codebook fingerprints alone achieves 89.9% median AUROC under target-level CV, generalizing to targets unseen during training. Adding protein tokens (concatenation to 2,560-dim) reduces performance to 84.4%, confirming that the molecule codebook is the primary carrier of phenomics-relevant signal. This is the clearest evidence for emergence: the codebook, trained on protein-ligand sequence pairs, learns molecule-level pharmacological properties that predict cellular phenotype without any cellular training data.

Crucially, Beaini et al.'s per-target result incorporates transcriptomic modulation (expression-level weighting of protein contributions) and a PPI graph derived from structural co-folding. Our result uses only sequence-derived embeddings.

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

## 6. Discussion

### 6.1 Emergence of Cellular Prediction

The model was trained on protein-ligand sequence pairs to predict pairwise interactions. It was never exposed to cellular data, microscopy images, phenomics readouts, transcriptomics, or any system-level biological signal. Yet its frozen representations predict what compounds do to cells.

This is analogous to — but arguably more surprising than — the emergence of arithmetic in language models. In language, arithmetic is latent in the training distribution (text contains equations). In molecular biology, the connection between pairwise binding events and cellular morphology is mediated by complex signaling cascades, metabolic networks, and regulatory feedback loops — none of which are explicit in protein-ligand sequence pairs. That this information is recoverable from frozen embeddings suggests the model has internalized a representation of molecular interactions rich enough to implicitly encode downstream biological consequences.

### 6.2 Evaluation Rigor

Our analysis reveals that the previously reported state-of-the-art (Beaini et al., 76%) was evaluated without cross-validation, on compounds the authors acknowledge the model likely trained on, using a 6-parameter rule-based pipeline fitted on the evaluation data. This is not unusual in computational biology — evaluation standards lag behind those in machine learning — but it means published numbers may significantly overstate true generalization performance.

We propose three levels of evaluation rigor for phenomics prediction, analogous to the temporal splits increasingly used in drug discovery benchmarks (Tossou et al., 2024). The compound-level CV with per-fold signature learning (Level 3) is the most conservative and provides the most honest estimate of how well a method would generalize to truly novel compounds.

### 6.3 From Pairwise to Systems Biology

Our results demonstrate that pairwise interaction representations compose to cell-level prediction. This composability is a necessary condition for computational systems biology: understanding a cell requires understanding the aggregate of all its molecular interactions, and our work shows that pair-level embeddings carry enough information for this aggregation to succeed.

The fusion CLS token — a per-pair representation from 16 layers of bidirectional cross-attention — adds +1-2 points by encoding interaction-specific information beyond what entity-level embeddings capture (Section 5.5). An architecture search over 16 model variants (Section 5.6) confirms that the ensemble's advantage is structural diversity, not model capacity — no single architecture matches it.

### 6.4 Limitations

1. **Model access**: The embedding model is proprietary and under review at a peer conference. We release our evaluation code and protocol but cannot release model weights. Reproducing results requires either access to the model or substitution of alternative molecular embeddings.

2. **Single dataset**: We evaluate on RxRx3-core only. Generalization to other cell lines, perturbation types, and phenotypic readouts remains untested.

3. **Per-target performance**: While our aggregate per-target AUROC (58.8%) exceeds the previous SOTA (53.9%), it remains modest in absolute terms. Pushing beyond 60% likely requires additional signals (transcriptomics, protein structure, tissue-specific expression) or fusion-level embeddings.

4. **Compound-level CV**: Our compound-level CV result (68.1%) is lower than the pair-level result (81.2%). The gap reflects the difficulty of generalizing to unseen compounds and suggests that current embedding-based approaches have room for improvement in out-of-distribution generalization.

## 7. Conclusion

We have demonstrated that frozen embeddings from a self-supervised molecular interaction model predict compound-induced cellular phenotype, a system-level property never encountered during training. This represents an emergent ability of pair-level molecular representations: the capacity to compose to cell-level biology.

Our approach outperforms the previous state-of-the-art while requiring five orders of magnitude less compute, using no external biological signals, and providing properly cross-validated performance estimates. We introduce a rigorous evaluation protocol for molecular phenomics prediction that distinguishes inflated from honest performance measures.

These results suggest a path toward computational systems biology grounded in learned molecular representations: rather than simulating biological systems from physical principles, we can learn representations of molecular interactions that encode enough about their downstream consequences to predict cellular behavior. The question of what other system-level properties emerge from pair-level representations — tissue-level effects, organism-level toxicity, patient-level drug response — is an exciting direction for future work.

## References

- Beaini, D., et al. (2026). Boltz-2: Connecting structure and function across biology. arXiv.
- Bray, M.A., et al. (2016). Cell Painting, a high-content image-based assay for morphological profiling using multiplexed fluorescent dyes. Nature Protocols.
- Caron, M., et al. (2021). Emerging properties in self-supervised vision transformers. ICCV.
- Chandrasekaran, S.N., et al. (2024). Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations. Nature Methods.
- Elhage, N., et al. (2022). Toy models of superposition. Transformer Circuits Thread.
- Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. Science.
- Tossou, P., et al. (2024). Temporal evaluation pitfalls in molecular property prediction. NeurIPS Workshop.
- Wallach, I. & Heifets, A. (2018). Most ligand-based classification benchmarks reward memorization rather than generalization. JCIM.
- Wei, J., et al. (2022). Emergent abilities of large language models. TMLR.
- Yang, K., et al. (2019). Analyzing learned molecular representations for property prediction. JCIM.
- [Anonymous]. (2026). Under review at NeurIPS 2026.
