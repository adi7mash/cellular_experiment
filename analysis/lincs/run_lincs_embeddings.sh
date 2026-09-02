#!/bin/bash
set -e

CHECKPOINT="s3://output-model-checkpoints/checkpoints/final/e_cheese_five_epoch_dark-snowball-245.ckpt"
PAIR_FILE="/opt/dlami/nvme/lincs_l1000/lincs_pairs.tsv"
OUTPUT_BASE="/opt/dlami/nvme/lincs_l1000"
MILLEFEUILLE=~/millefeuille/main.py

source ~/output/bin/activate

echo "=== LINCS L1000 Embedding Run ==="
echo "Pairs: $(wc -l < $PAIR_FILE) rows"
echo "Start: $(date -Iseconds)"

TOTAL_START=$SECONDS

# --- 1. Codebook molecule ---
echo "--- [1/2] Codebook molecule ---"
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
    --codebook-output-types tokens
echo "  Done: $((SECONDS - STEP_START))s"

# --- 2. Codebook protein ---
echo "--- [2/2] Codebook protein ---"
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
    --codebook-output-types tokens
echo "  Done: $((SECONDS - STEP_START))s"

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
echo ""
echo "=== Both runs complete ==="
echo "Total: ${TOTAL_ELAPSED}s"
echo "End: $(date -Iseconds)"

# Verify outputs
echo ""
echo "=== Output verification ==="
for dir in "$OUTPUT_BASE"/lincs_pairs_*/; do
    dirname=$(basename "$dir")
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.pt" | wc -l)
        echo "  ${dirname}: ${count} .pt files"
    fi
done
