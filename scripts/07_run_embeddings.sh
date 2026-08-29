#!/bin/bash
set -e

CHECKPOINT=${1:-"s3://output-model-checkpoints/checkpoints/final/e_cheese_five_epoch_dark-snowball-245.ckpt"}
LABEL=${2:-"dark-snowball-245"}
PAIR_FILE="/opt/dlami/nvme/rxrx3_phenomics/data/rxrx3_pairs.tsv"
OUTPUT_BASE="/opt/dlami/nvme/rxrx3_phenomics/${LABEL}"
MILLEFEUILLE=~/millefeuille/main.py
LOGFILE="/home/adimash/cellular_experiment/results/timing_${LABEL}.log"

source ~/output/bin/activate

mkdir -p "$OUTPUT_BASE"
mkdir -p /home/adimash/cellular_experiment/results

echo "=== RxRx3 Embedding Run: ${LABEL} ===" | tee "$LOGFILE"
echo "Checkpoint: ${CHECKPOINT}" | tee -a "$LOGFILE"
echo "Start: $(date -Iseconds)" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

TOTAL_START=$SECONDS

# --- 1. Codebook embeddings — molecule side ---
echo "--- [1/4] Codebook molecule ---" | tee -a "$LOGFILE"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --molecule_encoder codebook \
    --pooling none \
    --codebook-model-path "$CHECKPOINT" \
    --codebook-query 2 \
    --output "$OUTPUT_BASE" \
    --no-protein \
    --molecule-sequence-column SAFE \
    --nan-filter-column SAFE \
    --modality_type safe \
    --codebook-output-types tokens,pooled_attention
STEP_ELAPSED=$((SECONDS - STEP_START))
echo "  Codebook molecule: ${STEP_ELAPSED}s ($(date -d@${STEP_ELAPSED} -u +%Mm%Ss))" | tee -a "$LOGFILE"

# --- 2. Codebook embeddings — protein side ---
echo "--- [2/4] Codebook protein ---" | tee -a "$LOGFILE"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --protein_encoder codebook \
    --pooling none \
    --codebook-model-path "$CHECKPOINT" \
    --codebook-query 1 \
    --output "$OUTPUT_BASE" \
    --no-molecule \
    --modality_type protein \
    --codebook-output-types tokens,pooled_attention
STEP_ELAPSED=$((SECONDS - STEP_START))
echo "  Codebook protein: ${STEP_ELAPSED}s ($(date -d@${STEP_ELAPSED} -u +%Mm%Ss))" | tee -a "$LOGFILE"

# --- 3. Contrastive projection — molecule side ---
echo "--- [3/4] Contrastive projection molecule ---" | tee -a "$LOGFILE"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --molecule_encoder molecule \
    --pooling mean \
    --molecule-model-path "$CHECKPOINT" \
    --modality_type modality_2 \
    --output "$OUTPUT_BASE" \
    --no-protein \
    --molecule-sequence-column SAFE \
    --nan-filter-column SAFE \
    --use-safe-tokenizer \
    --use-contrastive-projection
STEP_ELAPSED=$((SECONDS - STEP_START))
echo "  Projection molecule: ${STEP_ELAPSED}s ($(date -d@${STEP_ELAPSED} -u +%Mm%Ss))" | tee -a "$LOGFILE"

# --- 4. Contrastive projection — protein side ---
echo "--- [4/4] Contrastive projection protein ---" | tee -a "$LOGFILE"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --protein_encoder protein \
    --pooling mean \
    --protein-model-path "$CHECKPOINT" \
    --modality_type modality_1 \
    --output "$OUTPUT_BASE" \
    --no-molecule \
    --use-contrastive-projection
STEP_ELAPSED=$((SECONDS - STEP_START))
echo "  Projection protein: ${STEP_ELAPSED}s ($(date -d@${STEP_ELAPSED} -u +%Mm%Ss))" | tee -a "$LOGFILE"

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
echo "" | tee -a "$LOGFILE"
echo "=== All 4 runs complete ===" | tee -a "$LOGFILE"
echo "Total wall time: ${TOTAL_ELAPSED}s ($(date -d@${TOTAL_ELAPSED} -u +%Hh%Mm%Ss))" | tee -a "$LOGFILE"
echo "End: $(date -Iseconds)" | tee -a "$LOGFILE"

# --- Verify outputs ---
echo "" | tee -a "$LOGFILE"
echo "=== Output verification ===" | tee -a "$LOGFILE"
for dir in "$OUTPUT_BASE"/*/; do
    dirname=$(basename "$dir")
    count=$(find "$dir" -name "*.pt" | wc -l)
    echo "  ${dirname}: ${count} .pt files" | tee -a "$LOGFILE"
done
