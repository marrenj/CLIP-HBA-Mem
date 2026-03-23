#!/bin/bash
# ------------------------------------------------------------------------
# Extract clip_hba_mem embeddings for the combined LaMem + MemCat dataset,
# folds 2-10.  Fold 1 is assumed already complete.
#
# Usage (from the CLIP-HBA directory):
#   bash extract_embeddings_combined_hba_folds2_10.sh
#
# Override defaults via environment variables:
#   LAMEM_IMG_ROOT=/path/to/lamem/images \
#   MEMCAT_IMG_ROOT=/path/to/memcat/images \
#   CUDA_DEVICE=0 \
#   bash extract_embeddings_combined_hba_folds2_10.sh
#
# Existing .pt files are skipped (safe to re-run after interruption).
#
# Outputs:
#   ./Data/combined_lamem_memcat/embeddings/clip_hba_mem_fold{2..10}_{train,val,test}.pt
# ------------------------------------------------------------------------

set -euo pipefail

PYTHON="${PYTHON:-python}"

LAMEM_IMG_ROOT="${LAMEM_IMG_ROOT:-./Data/lamem/images/}"
MEMCAT_IMG_ROOT="${MEMCAT_IMG_ROOT:-./Data/memcat/images/}"
MEMCAT_META_CSV="${MEMCAT_META_CSV:-./Data/memcat/memcat_image_data.csv}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-./Data/lamem/epoch97_dora_params.pth}"
OUT_DIR="${OUT_DIR:-./Data/combined_lamem_memcat/embeddings/}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${OUT_DIR}" ./logs

for FOLD in 1 2 3 4 5 6 7 8 9 10; do
    "${PYTHON}" extract_embeddings.py \
        --training_data   combined_lamem_memcat \
        --model_type      clip_hba_mem \
        --fold            "${FOLD}" \
        --img_root        "${LAMEM_IMG_ROOT}" \
        --memcat_img_root "${MEMCAT_IMG_ROOT}" \
        --memcat_meta_csv "${MEMCAT_META_CSV}" \
        --backbone_checkpoint "${BACKBONE_CHECKPOINT}" \
        --out_dir         "${OUT_DIR}" \
        --batch_size      "${BATCH_SIZE}" \
        --num_workers     "${NUM_WORKERS}" \
        --cuda            "${CUDA_DEVICE}"
done

echo ""
echo "=========================================================="
echo "All folds complete."
echo "End time : $(date)"
echo "================================================================"
