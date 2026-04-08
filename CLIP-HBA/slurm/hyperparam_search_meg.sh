#!/bin/bash
# =============================================================================
# SLURM job — Bayesian hyperparameter search for CLIP-HBA-MEG MLP heads
#
# Runs all 36 new timepoint searches in PARALLEL on a single GPU.
# Each search is launched as a background process; they share GPU time via
# CUDA's built-in time-multiplexing.  Since each individual search uses only
# a small fraction of the A100's compute and memory, this saturates the GPU
# far more efficiently than the job-array approach (one task per GPU).
#
# New timepoints (36 total, already-completed timepoints skipped):
#   0, 25, 75, 125, 175, 225, 275, 325, 375, 425, 450, 475,
#   525, 550, 575, 625, 650, 675, 725, 750, 775, 825, 850, 875,
#   925, 950, 975, 1025, 1050, 1075, 1125, 1150, 1175, 1225, 1250, 1275
#
# Already covered (skipped):
#   50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800,
#   900, 1000, 1100, 1200, 1300
#
# Resource notes
# --------------
# GPU:    1 A100 (40 GB) — shared across all N_PARALLEL processes via CUDA
#         time-multiplexing.  Each process uses ~200–500 MB VRAM (tiny MLP +
#         66-dim in-memory embeddings), so 36 parallel processes ≈ 7–18 GB
#         total VRAM, well within the 40 GB limit.
# CPUs:   16 — shared across all parallel Python/Optuna processes; each
#         individual search is single-threaded so 16 cores handles ~36
#         concurrent processes comfortably.
# Memory: 32 G — 36 processes × ~200–400 MB each ≈ 7–14 GB; 32 G is safe.
# Time:   36:00:00 — all 36 searches run simultaneously, each taking ~17–25 h;
#         36 h provides headroom for variance.
#
# Tuning N_PARALLEL
# -----------------
# N_PARALLEL controls how many searches run simultaneously.  Start conservatively
# and increase if GPU memory allows.  Recommended values:
#   N_PARALLEL=9   →  4 batches × ~24 h ≈ 4 days   (very safe)
#   N_PARALLEL=12  →  3 batches × ~24 h ≈ 3 days   (safe)
#   N_PARALLEL=18  →  2 batches × ~24 h ≈ 2 days   (comfortable for A100)
#   N_PARALLEL=36  →  1 batch   × ~24 h ≈ 1 day    (ideal; fits A100 memory)
# If you hit GPU OOM errors, reduce N_PARALLEL.
#
# Usage
# -----
#   cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA
#   sbatch slurm/hyperparam_search_meg.sh
#
# To override N_PARALLEL without editing:
#   N_PARALLEL=12 sbatch slurm/hyperparam_search_meg.sh
#
# Monitoring
# ----------
#   squeue -u $USER
#   tail -f ./logs/hyperparam_search_meg/slurm_<jobid>.out     # top-level log
#   tail -f ./logs/hyperparam_search_meg/tp<TP>_<jobid>.log    # per-timepoint log
# =============================================================================

#SBATCH --job-name=meg_hpsearch
#SBATCH --output=./logs/hyperparam_search_meg/slurm_%j.out
#SBATCH --error=./logs/hyperparam_search_meg/slurm_%j.err
#SBATCH --account=dsi_dgx_iacc
#SBATCH --partition=interactive_gpu
#SBATCH --qos=dgx_iacc
#SBATCH --gres=gpu:nvidia_a100-sxm4-40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=marren.jenkins.1@vanderbilt.edu

set -euo pipefail

# ---------------------------------------------------------------------------
# How many searches to run simultaneously on this one GPU.
# Override at submission time: N_PARALLEL=12 sbatch slurm/hyperparam_search_meg.sh
# See "Tuning N_PARALLEL" notes above for guidance.
# ---------------------------------------------------------------------------
N_PARALLEL=${N_PARALLEL:-36}

# ---------------------------------------------------------------------------
# Timepoints — 25 ms resolution, skipping already-completed timepoints
# (50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300)
# ---------------------------------------------------------------------------
TIMEPOINTS=(
      0    25    75   125   175   225   275   325   375
    425   450   475   525   550   575   625   650   675
    725   750   775   825   850   875   925   950   975
   1025  1050  1075  1125  1150  1175  1225  1250  1275
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module load python/3.11.5
source ~/envs/clip_hba/bin/activate

cd /panfs/accrepfs.vampire/home/jenkm22/CLIP-HBA-Mem/CLIP-HBA
mkdir -p ./logs/hyperparam_search_meg

# Each task gets exclusive GPU 0 within this job.
export CUDA_DEVICE=0

# ---------------------------------------------------------------------------
# Job info
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "GPU(s)       : $CUDA_VISIBLE_DEVICES"
echo "N_PARALLEL   : ${N_PARALLEL}"
echo "Timepoints   : ${#TIMEPOINTS[@]} total"
echo "Start time   : $(date)"
echo "Working dir  : $(pwd)"
echo "=========================================="

# ---------------------------------------------------------------------------
# Launch all searches — N_PARALLEL at a time
#
# Each timepoint's search runs as a background process writing its own log.
# When a batch of N_PARALLEL processes is full, we wait for all of them to
# finish before launching the next batch.
# ---------------------------------------------------------------------------
batch_pids=()

for i in "${!TIMEPOINTS[@]}"; do
    TP=${TIMEPOINTS[$i]}

    echo "[$(date '+%H:%M:%S')] Launching tp=${TP} ms (index ${i})"

    python train_mem_meg_hyperparam_search.py \
        --timepoint_ms "${TP}" \
        --n-trials 50 \
        > "./logs/hyperparam_search_meg/tp${TP}_${SLURM_JOB_ID}.log" 2>&1 &

    batch_pids+=($!)

    # When the batch is full, wait for all processes in it to finish.
    if [ "${#batch_pids[@]}" -ge "${N_PARALLEL}" ]; then
        echo "[$(date '+%H:%M:%S')] Batch of ${N_PARALLEL} launched — waiting..."
        for pid in "${batch_pids[@]}"; do
            wait "$pid"
        done
        echo "[$(date '+%H:%M:%S')] Batch complete."
        batch_pids=()
    fi
done

# Wait for any remaining processes in an incomplete final batch.
if [ "${#batch_pids[@]}" -gt 0 ]; then
    echo "[$(date '+%H:%M:%S')] Waiting for final batch (${#batch_pids[@]} processes)..."
    for pid in "${batch_pids[@]}"; do
        wait "$pid"
    done
fi

# ---------------------------------------------------------------------------
echo "=========================================="
echo "All ${#TIMEPOINTS[@]} hyperparameter searches complete."
echo "End time : $(date)"
echo "=========================================="
