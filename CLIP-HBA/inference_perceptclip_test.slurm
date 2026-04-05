#!/bin/bash
#SBATCH --job-name=perceptclip_test
#SBATCH --output=./logs/perceptclip_test_%j.out
#SBATCH --error=./logs/perceptclip_test_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

# ---------------------------------------------------------------------------
# Usage:
#   sbatch inference_perceptclip_test.slurm
#
# Downloads the pre-trained PerceptCLIP checkpoint from HuggingFace and runs
# inference on all 10 test splits of the combined LaMem+MemCat dataset.
#
# Submit from the CLIP-HBA directory:
#   cd /path/to/CLIP-HBA-Mem/CLIP-HBA
#   sbatch inference_perceptclip_test.slurm
#
# Logs:
#   SLURM stdout/stderr  ->  ./logs/perceptclip_test_<jobid>.out / .err
#   Per-fold predictions ->  ./preds/inference/combined_lamem_memcat_<timestamp>/
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Environment setup -------------------------------------------------------
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

mkdir -p ./logs

# --- Stage images to local /tmp for faster I/O -------------------------------
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

echo "=========================================="
echo "Running PerceptCLIP inference on all 10 combined LaMem+MemCat test folds..."
echo "=========================================="

python inference_mem.py \
    --dataset combined_lamem_memcat \
    --fold all \
    --model_type perceptclip \
    --checkpoint huggingface \
    --data_dir ./Data \
    --batch_size 64 \
    --device cuda:0

echo "=========================================="
echo "Inference complete."
echo "End time : $(date)"
echo "=========================================="
