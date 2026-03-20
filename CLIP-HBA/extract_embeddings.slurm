#!/bin/bash
#SBATCH --job-name=extract_embeddings
#SBATCH --output=./logs/slurm_extract_embeddings_%j.out
#SBATCH --error=./logs/slurm_extract_embeddings_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch extract_embeddings.slurm
#
# Runs extract_embeddings.py for:
#   - All 5 LaMem folds (clip_hba_mem + clip_frozen_mlp)
#   - All 10 combined LaMem+MemCat folds (clip_frozen_mlp)
#
# Images are copied to /tmp once and reused across all folds.
# Embedding files are written to:
#   ./Data/lamem/embeddings/
#   ./Data/combined_lamem_memcat/embeddings/
# Existing .pt files are skipped, so this script is safe to re-run.
#
# Output files:
#   ./Data/lamem/embeddings/clip_hba_mem_fold{1..5}_{train,val,test}.pt
#   ./Data/lamem/embeddings/clip_frozen_mlp_fold{1..5}_{train,val,test}.pt
#   ./Data/combined_lamem_memcat/embeddings/clip_frozen_mlp_fold{01..10}_{train,val,test}.pt
#
# After this job completes, add to your training config:
#   'embeddings_dir': './Data/lamem/embeddings/'            (LaMem)
#   'embeddings_dir': './Data/combined_lamem_memcat/embeddings/'  (combined)
# and training will use EmbeddingDataset + MLPOnlyHead automatically.
#
# Submit from the CLIP-HBA directory:
#   cd /path/to/CLIP-HBA-Mem/CLIP-HBA
#   sbatch extract_embeddings.slurm
#
# Logs:
#   SLURM stdout/stderr -> ./logs/slurm_extract_embeddings_<jobid>.out / .err
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Environment setup ------------------------------------------------------
module load python/3.11.5
source ~/envs/clip_hba/bin/activate

cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA

echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

mkdir -p ./logs ./Data/lamem/embeddings ./Data/combined_lamem_memcat/embeddings

# =============================================================================
# Copy images to local /tmp once; all folds read from the same copy.
# =============================================================================

LOCAL_LAMEM=/tmp/lamem_cache
if [ ! -d "${LOCAL_LAMEM}" ]; then
    echo "Copying LaMem images to local storage..."
    mkdir -p "${LOCAL_LAMEM}"
    rsync -a --info=progress2 "./Data/lamem/images/" "${LOCAL_LAMEM}/"
else
    echo "Using cached LaMem data at ${LOCAL_LAMEM}"
fi

LOCAL_MEMCAT=/tmp/memcat_cache
if [ ! -d "${LOCAL_MEMCAT}" ]; then
    echo "Copying MemCat images to local storage..."
    mkdir -p "${LOCAL_MEMCAT}"
    rsync -a --info=progress2 "./Data/memcat/images/" "${LOCAL_MEMCAT}/"
else
    echo "Using cached MemCat data at ${LOCAL_MEMCAT}"
fi

export LAMEM_IMG_ROOT="${LOCAL_LAMEM}"
export MEMCAT_IMG_ROOT="${LOCAL_MEMCAT}"
export MEMCAT_META_CSV="./Data/memcat/memcat_image_data.csv"
export CUDA_DEVICE=0

echo "=========================================="

# =============================================================================
# LaMem — clip_hba_mem (all 5 folds)
# =============================================================================
echo ""
echo "=========================================="
echo "Extracting clip_hba_mem embeddings (LaMem)"
echo "=========================================="

for FOLD in 1 2 3 4 5; do
    echo ""
    echo "--- clip_hba_mem | lamem | fold ${FOLD} ---"
    python extract_embeddings.py \
        --training_data lamem \
        --model_type clip_hba_mem \
        --fold "${FOLD}" \
        --img_root "${LAMEM_IMG_ROOT}" \
        --out_dir "./Data/lamem/embeddings/" \
        --backbone_checkpoint "./Data/lamem/epoch97_dora_params.pth" \
        --batch_size 256 \
        --num_workers 16 \
        --cuda 0
done

# =============================================================================
# LaMem — clip_frozen_mlp (all 5 folds)
# =============================================================================
echo ""
echo "=========================================="
echo "Extracting clip_frozen_mlp embeddings (LaMem)"
echo "=========================================="

for FOLD in 1 2 3 4 5; do
    echo ""
    echo "--- clip_frozen_mlp | lamem | fold ${FOLD} ---"
    python extract_embeddings.py \
        --training_data lamem \
        --model_type clip_frozen_mlp \
        --fold "${FOLD}" \
        --img_root "${LAMEM_IMG_ROOT}" \
        --out_dir "./Data/lamem/embeddings/" \
        --batch_size 256 \
        --num_workers 16 \
        --cuda 0
done

# =============================================================================
# Combined LaMem + MemCat — clip_frozen_mlp (all 10 folds)
# =============================================================================
echo ""
echo "=========================================="
echo "Extracting clip_frozen_mlp embeddings (combined LaMem + MemCat)"
echo "=========================================="

for FOLD in 1 2 3 4 5 6 7 8 9 10; do
    echo ""
    echo "--- clip_frozen_mlp | combined_lamem_memcat | fold ${FOLD} ---"
    python extract_embeddings.py \
        --training_data combined_lamem_memcat \
        --model_type clip_frozen_mlp \
        --fold "${FOLD}" \
        --img_root "${LAMEM_IMG_ROOT}" \
        --memcat_img_root "${MEMCAT_IMG_ROOT}" \
        --memcat_meta_csv "${MEMCAT_META_CSV}" \
        --out_dir "./Data/combined_lamem_memcat/embeddings/" \
        --batch_size 256 \
        --num_workers 16 \
        --cuda 0
done

echo ""
echo "=========================================="
echo "All embeddings extracted."
echo "End time : $(date)"
echo "=========================================="
