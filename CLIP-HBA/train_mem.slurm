#!/bin/bash
#SBATCH --job-name=clip_hba_mem
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=30:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu
 
# ---------------------------------------------------------------------------
# Usage:
#   sbatch train_mem.slurm
#
# Trains all 5 LaMem folds sequentially on a single node.
# Data is copied to /tmp once and reused across all folds.
#
# Submit from the CLIP-HBA directory (the one containing train_mem.py):
#   cd /path/to/CLIP-HBA-Mem/CLIP-HBA
#   sbatch train_mem.slurm
#
# Logs:
#   SLURM stdout/stderr  ->  ./logs/slurm_<jobid>.out / .err
#   Training log         ->  ./logs/mem_fold<N>_<timestamp>.log  (written by train_mem.py)
# ---------------------------------------------------------------------------
 
set -euo pipefail
 
# --- Environment setup ------------------------------------------------------
# conda activate clip-hba
module load python/3.11.5
source ~/envs/clip_hba/bin/activate
 
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
 
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPUs         : $CUDA_VISIBLE_DEVICES"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# =============================================================================
# Configuration
# =============================================================================

# Dataset to train on. Options:
#   - lamem
#   - combined_lamem_memcat
export TRAINING_DATA="lamem"

# Model type for memorability training. Options:
#   - clip_hba_mem
#   - perceptclip
export MODEL_TYPE="clip_hba_mem"

# =============================================================================
# Copy image data to local /tmp once; all folds read from the same copy.
# =============================================================================

LOCAL_LAMEM=/tmp/lamem_cache
if [ ! -d "${LOCAL_LAMEM}" ]; then
    echo "Copying LaMem images to local storage..."
    mkdir -p "${LOCAL_LAMEM}"
    rsync -a --info=progress2 "./Data/lamem/images/" "${LOCAL_LAMEM}/"
else
    echo "Using cached LaMem data at ${LOCAL_LAMEM}"
fi
export LAMEM_IMG_ROOT="${LOCAL_LAMEM}"

if [ "${TRAINING_DATA}" = "combined_lamem_memcat" ]; then
    LOCAL_MEMCAT=/tmp/memcat_cache
    if [ ! -d "${LOCAL_MEMCAT}" ]; then
        echo "Copying MemCat images to local storage..."
        mkdir -p "${LOCAL_MEMCAT}"
        rsync -a --info=progress2 "./Data/memcat/images/" "${LOCAL_MEMCAT}/"
    else
        echo "Using cached MemCat data at ${LOCAL_MEMCAT}"
    fi
    export MEMCAT_IMG_ROOT="${LOCAL_MEMCAT}"
fi

echo "=========================================="

# --- Train all folds sequentially --------------------------------------------
for FOLD in 2 3; do
    echo ""
    echo "=========================================="
    echo "Starting fold ${FOLD} at $(date)"
    echo "=========================================="
    export FOLD="${FOLD}"
    N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    if [ "$N_GPUS" -eq 1 ]; then
        export CUDA_DEVICE=0
    else
        export CUDA_DEVICE=-1  # DataParallel for multi-GPU
    fi
    python train_mem.py
done
 
echo "=========================================="
echo "All folds complete."
echo "End time : $(date)"
echo "=========================================="