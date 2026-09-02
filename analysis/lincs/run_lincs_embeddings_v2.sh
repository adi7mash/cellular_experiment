#!/bin/bash
set -e

CHECKPOINT="s3://output-model-checkpoints/checkpoints/final/e_cheese_five_epoch_dark-snowball-245.ckpt"
PAIR_FILE="/opt/dlami/nvme/lincs_l1000/lincs_pairs.tsv"
OUTPUT_BASE="/opt/dlami/nvme/lincs_l1000"
MILLEFEUILLE=~/millefeuille/main.py

source ~/output/bin/activate

echo "=== LINCS L1000 Embedding Run v2 — missing types ==="
echo "Start: $(date -Iseconds)"
TOTAL_START=$SECONDS

# --- 1. Codebook molecule with pooled_attention ---
echo "--- [1/4] Codebook molecule (tokens + pooled_attention) ---"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --molecule_encoder codebook \
    --pooling none \
    --codebook-model-path "$CHECKPOINT" \
    --codebook-query 2 \
    --output "$OUTPUT_BASE/v2" \
    --no-protein \
    --molecule-sequence-column SAFE \
    --nan-filter-column SAFE \
    --modality_type safe \
    --codebook-output-types tokens,pooled_attention
echo "  Done: $((SECONDS - STEP_START))s"

# --- 2. Codebook protein with pooled_attention ---
echo "--- [2/4] Codebook protein (tokens + pooled_attention) ---"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --protein_encoder codebook \
    --pooling none \
    --codebook-model-path "$CHECKPOINT" \
    --codebook-query 1 \
    --output "$OUTPUT_BASE/v2" \
    --no-molecule \
    --modality_type protein \
    --codebook-output-types tokens,pooled_attention
echo "  Done: $((SECONDS - STEP_START))s"

# --- 3. Contrastive projection molecule ---
echo "--- [3/4] Contrastive projection molecule ---"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --molecule_encoder molecule \
    --pooling mean \
    --molecule-model-path "$CHECKPOINT" \
    --modality_type modality_2 \
    --output "$OUTPUT_BASE/v2" \
    --no-protein \
    --molecule-sequence-column SAFE \
    --nan-filter-column SAFE \
    --use-safe-tokenizer \
    --use-contrastive-projection
echo "  Done: $((SECONDS - STEP_START))s"

# --- 4. Contrastive projection protein ---
echo "--- [4/4] Contrastive projection protein ---"
STEP_START=$SECONDS
python $MILLEFEUILLE \
    --pair-file "$PAIR_FILE" \
    --protein_encoder protein \
    --pooling mean \
    --protein-model-path "$CHECKPOINT" \
    --modality_type modality_1 \
    --output "$OUTPUT_BASE/v2" \
    --no-molecule \
    --use-contrastive-projection
echo "  Done: $((SECONDS - STEP_START))s"

TOTAL_ELAPSED=$((SECONDS - TOTAL_START))
echo ""
echo "=== All 4 runs complete: ${TOTAL_ELAPSED}s ==="
echo "End: $(date -Iseconds)"

echo ""
echo "=== Output verification ==="
for dir in "$OUTPUT_BASE"/v2/lincs_pairs_*/; do
    dirname=$(basename "$dir")
    if [ -d "$dir" ]; then
        count=$(find "$dir" -name "*.pt" | wc -l)
        echo "  ${dirname}: ${count} .pt files"
    fi
done
