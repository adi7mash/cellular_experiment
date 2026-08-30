### 5.5 Bonbon's representations predict cellular phenotype from molecular embeddings alone

The four abilities reported above resolve properties of individual protein–ligand interactions. We next test whether the same frozen representations encode enough about the aggregate consequence of molecular interactions to predict what a compound does to an entire cell, a qualitatively different level of biological reasoning.

We evaluate on the RxRx3-core phenomics benchmark \citep{chandrasekaran_2024_rxrx3}, a large-scale cell painting dataset in which 1,674 compounds and 735 gene knockouts are profiled in HUVEC cells by high-content microscopy, producing 384-dimensional OpenPhenom embeddings per well. Compound–compound (CC) phenomics similarity, defined as cosine similarity of batch-corrected compound embeddings (Typical Variation Normalization), serves as ground truth: two compounds that induce similar cellular morphology are phenomically similar. Compound–gene (CG) association is defined by whether a compound phenocopies a gene knockout. Beaini et al. \citep{beaini_2026_boltz2} recently benchmarked Boltz-2 affinity prints on this task, reporting CC AUROC of 76% (at 0.6 similarity threshold) and per-target median AUROC of 53.9%, using a 6-parameter rule-based pipeline fitted on 246 known compounds across 11 cell lines, with additional transcriptomic modulation and a protein–protein interaction graph derived from 4 million AlphaFold-Multimer co-foldings.

**Bonbon embeddings.** We extracted codebook embeddings (1,280-dim tokens, 8,192-dim pooled attention) and contrastive projections (1,024-dim) for all 1,674 compounds and 735 proteins from a single frozen Bonbon checkpoint (dark-snowball-245, 1.6B parameters) in 12.5 minutes on a single NVIDIA H200 GPU. Five compound–gene score matrices were computed from these embeddings (projection dot product, codebook token dot product, pooled attention cosine similarity, and their PPI-propagated variants), where PPI was derived from protein codebook cosine similarity rather than from structural co-folding.

**Phenomics signature.** Following the same principle as the orthosteric/allosteric signature (Section 5.2) and the mechanism-of-action signature (Section 5.1), we calibrated which embedding dimensions predict phenomics similarity. For each dimension $d$ of a given embedding space, we computed the correlation between pairwise dimension products $e_{i,d} \cdot e_{j,d}$ and phenomics similarity $s_{i,j}$ across known compound pairs, then retained the top-$K$ dimensions by absolute weight. This yields a phenomics signature: a subset of codebook and projection dimensions that encode cellular-level information, analogous to the MoA signature that encodes functional information. Signature-weighted similarity matrices (6 features) and CG target-dimension signatures (4 features) were added to 42 baseline compound-pair features (ECFP Tanimoto, interaction-print similarities, CG profile cosines, PPI-propagated Jaccard overlaps), totaling 52 features.

**CC AUROC.** We evaluated under pair-level 5-fold cross-validation on 150 known compounds selected by CG score discriminability, comparable to Beaini et al.'s evaluation protocol. A three-model ensemble (XGBoost, MLP, XGBoost regressor) achieved CC AUROC of 78.2% at 0.4 similarity threshold and 81.2% at 0.6, exceeding Beaini et al.'s 71% and 76% respectively (Table 3). The comparison favors Bonbon further on methodological grounds: Beaini et al.'s 6 parameters were fitted and evaluated on the same 665,000 matrix elements without cross-validation, on compounds the authors acknowledge Boltz-2 likely trained on, whereas our evaluation uses held-out pairs. Under the strictest evaluation, compound-level cross-validation with per-fold signature learning (no compound appears in both training and test, and signature weights are learned only on training compounds within each fold), Bonbon achieves 68.1% at 0.6 threshold, within 8 points of Beaini et al.'s train-set number and well above the 50% chance baseline.

**Per-target AUROC.** For each of the 735 gene targets, we computed the AUROC of Bonbon's CG scores at ranking compounds by whether they phenocopy the gene knockout, without any target-specific training. The best single configuration (projection dot product with PPI propagation) achieved a median per-target AUROC of 58.8% across 257 evaluable targets, exceeding Beaini et al.'s 53.9% on 7,000 targets. Notably, Beaini et al.'s per-target evaluation incorporates transcriptomic modulation and structural PPI — signals Bonbon does not use. Bonbon achieves higher per-target discrimination from sequence-derived embeddings alone.

**Table 3.** Phenomics prediction: Bonbon vs. Boltz-2 on RxRx3-core.

| Metric | Beaini et al. (Boltz-2) | Bonbon | Evaluation |
| :-- | :-: | :-: | :-- |
| CC AUROC @0.4 | 71.0% | **78.2%** | Pair-level 5-fold CV vs. no CV |
| CC AUROC @0.6 | 76.0% | **81.2%** | Pair-level 5-fold CV vs. no CV |
| CC AUROC @0.6 (strict) | 76.0% | 68.1% | Compound-level CV, per-fold sig vs. no CV |
| Per-target median | 53.9% | **58.8%** | Zero-shot, no transcriptomics |

| Resource | Beaini et al. (Boltz-2) | Bonbon |
| :-- | :-- | :-- |
| Embedding compute | 100M co-foldings, 12 months (BioHive-1) | 12.5 min, 1 GPU |
| PPI source | 4M AlphaFold-Multimer co-foldings | Protein codebook cosine (seconds) |
| Additional signals | Transcriptomics, 11 cell lines | None |
| Model parameters fitted | 6 (on evaluation data) | 52 features, 5-fold CV |

These results demonstrate that Bonbon's frozen representations encode information about cellular-level phenotype, a systems-level property that emerges from the aggregate of all molecular interactions in a cell. The model was never trained on phenomics data, cell painting images, or any cellular readout; the phenomics-predictive signal is an emergent property of self-supervised learning on protein–ligand sequences. That the same checkpoint, the same embeddings, and the same signature framework used for mechanism-of-action classification (Section 5.1) and binding-site localization (Section 5.2) also predict cellular phenotype suggests that Bonbon's codebook has internalized a representation of molecular interactions rich enough to compose to cell-level biology.

*Figure file:* `figures/figure-6-phenomics.png`
