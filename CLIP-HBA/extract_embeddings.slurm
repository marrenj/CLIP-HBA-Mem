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
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch extract_embeddings.slurm
#
# Runs extract_embeddings.py for all 5 LaMem folds for both model types:
#   - clip_hba_mem   (CLIP-HBA frozen backbone, DoRA-patched ViT-L/14)
#   - clip_frozen_mlp (vanilla frozen CLIP ViT-L/14 from HuggingFace)
#
# Images are copied to /tmp once and reused across all folds.
# Embedding files are written to ./Data/lamem/embeddings/.
# Existing .pt files are skipped, so this script is safe to re-run.
#
# Output files (30 total):
#   ./Data/lamem/embeddings/clip_hba_mem_fold{1..5}_{train,val,test}.pt
#   ./Data/lamem/embeddings/clip_frozen_mlp_fold{1..5}_{train,val,test}.pt
#
# After this job completes, add to your training config:
#   'embeddings_dir': './Data/lamem/embeddings/'
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

mkdir -p ./logs ./Data/lamem/embeddings

# =============================================================================
# Copy LaMem images to local /tmp once; all folds read from the same copy.
# =============================================================================

LOCAL_DATA=/tmp/lamem_cache
if [ ! -d "${LOCAL_DATA}" ]; then
    echo "Copying LaMem images to local storage..."
    mkdir -p "${LOCAL_DATA}"
    rsync -a --info=progress2 "./Data/lamem/images/" "${LOCAL_DATA}/"
else
    echo "Using cached LaMem data at ${LOCAL_DATA}"
fi

export LAMEM_IMG_ROOT="${LOCAL_DATA}"
export CUDA_DEVICE=0

echo "=========================================="

# =============================================================================
# Extract embeddings — clip_hba_mem (CLIP-HBA backbone, all 5 folds)
# =============================================================================
echo ""
echo "=========================================="
echo "Extracting clip_hba_mem embeddings"
echo "=========================================="

for FOLD in 1 2 3 4 5; do
    echo ""
    echo "--- clip_hba_mem | fold ${FOLD} ---"
    python extract_embeddings.py \
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
# Extract embeddings — clip_frozen_mlp (vanilla CLIP backbone, all 5 folds)
# =============================================================================
echo ""
echo "=========================================="
echo "Extracting clip_frozen_mlp embeddings"
echo "=========================================="

for FOLD in 1 2 3 4 5; do
    echo ""
    echo "--- clip_frozen_mlp | fold ${FOLD} ---"
    python extract_embeddings.py \
        --model_type clip_frozen_mlp \
        --fold "${FOLD}" \
        --img_root "${LAMEM_IMG_ROOT}" \
        --out_dir "./Data/lamem/embeddings/" \
        --batch_size 256 \
        --num_workers 16 \
        --cuda 0
done

echo ""
echo "=========================================="
echo "All embeddings extracted."
echo "End time : $(date)"
echo "=========================================="
