"""Hyperparameter sweep for the memorability MLP head.

Usage
-----
    cd CLIP-HBA
    python train_mem_sweep.py

Edit BASE_CONFIG (data paths, backbone, device) and SWEEP_GRID (search space)
below before running.  All 48 combinations are evaluated sequentially on fold 1
using 20 % of the training data and early-stopping patience of 5, so each run
is much faster than a full training run.  After the sweep, re-train the winning
config on the full training set (train_fraction=1.0, patience=10) across all 5
folds to get final performance numbers.

Outputs
-------
  sweep_out/<timestamp>/run_NNN/log.txt  -- per-run training log
  sweep_out/<timestamp>/sweep_results.csv -- all results sorted by val Spearman rho
"""

import csv
import datetime
import itertools
import os
import sys

import torch.nn as nn

from functions.train_mem_pipeline import run_mem_training

# ---------------------------------------------------------------------------
# Fixed configuration — edit paths and device to match your environment
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    # Fold used for the sweep; run the winning config on all 5 folds afterward
    'fold':      1,
    'train_csv': './Data/lamem/lamem_train_1.csv',
    'val_csv':   './Data/lamem/lamem_val_1.csv',
    'test_csv':  './Data/lamem/lamem_test_1.csv',
    'img_root':  './Data/lamem/images/',

    # Backbone (frozen — these must match the checkpoint's DoRA config)
    'backbone_checkpoint': './Data/lamem/epoch97_dora_params.pth',
    'backbone':            'ViT-L/14',
    'vision_layers':       2,
    'transformer_layers':  1,
    'rank':                32,

    # Device: 0=cuda:0, 1=cuda:1, -1=DataParallel, 2=cpu
    'cuda': 1,

    # Fixed training settings for the sweep
    'epochs':                  300,
    'early_stopping_patience': 5,    # shorter than final training (10) to speed up sweep
    'train_fraction':          0.2,  # subsample 20 % of training data per run
    'criterion':               nn.MSELoss(),
    'random_seed':             42,

    # Checkpoints written here during sweep (overwritten each run — not for production use)
    'checkpoint_path': './models/sweep/clip_hba_mem_sweep',
}

# ---------------------------------------------------------------------------
# Search grid   (4 × 2 × 3 × 2 = 48 total combinations)
# ---------------------------------------------------------------------------
SWEEP_GRID = {
    'hidden_dims':  [(512, 256), (256, 128), (512, 256, 128), (256,)],
    'dropout_rate': [0.3, 0.5],
    'lr':           [1e-4, 5e-5, 1e-5],
    'batch_size':   [32, 64],
}

# ---------------------------------------------------------------------------
# Sweep output directory
# ---------------------------------------------------------------------------
SWEEP_DIR = f'./sweep_out/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'


def run_sweep():
    os.makedirs(SWEEP_DIR, exist_ok=True)
    os.makedirs(BASE_CONFIG['checkpoint_path'], exist_ok=True)

    # Keep a reference to the real stdout so sweep progress is always visible
    # even though run_mem_training redirects stdout to a per-run log file.
    sweep_stdout = sys.stdout

    keys = list(SWEEP_GRID.keys())
    combos = list(itertools.product(*[SWEEP_GRID[k] for k in keys]))
    n_total = len(combos)

    results_csv = os.path.join(SWEEP_DIR, 'sweep_results.csv')
    with open(results_csv, 'w', newline='') as f:
        csv.writer(f).writerow(['run_id', *keys, 'best_val_rho'])

    results = []

    for run_id, values in enumerate(combos):
        params = dict(zip(keys, values))

        sweep_stdout.write(
            f'\n[Sweep {run_id + 1}/{n_total}] '
            + '  '.join(f'{k}={v}' for k, v in params.items())
            + '\n'
        )
        sweep_stdout.flush()

        run_dir = os.path.join(SWEEP_DIR, f'run_{run_id:03d}')
        os.makedirs(run_dir, exist_ok=True)

        config = {
            **BASE_CONFIG,
            **params,
            'log_path': os.path.join(run_dir, 'log.txt'),
        }

        try:
            best_rho = run_mem_training(config)
        except Exception as exc:
            sweep_stdout.write(f'  [ERROR] run {run_id} failed: {exc}\n')
            sweep_stdout.flush()
            best_rho = float('nan')

        # sys.stdout was restored inside run_mem_training; sweep output is safe.
        results.append({'run_id': run_id, **params, 'best_val_rho': best_rho})

        with open(results_csv, 'a', newline='') as f:
            csv.writer(f).writerow(
                [run_id, *[params[k] for k in keys], f'{best_rho:.6f}']
            )

        sweep_stdout.write(f'  -> best_val_rho = {best_rho:.4f}\n')
        sweep_stdout.flush()

    # --- Summary ---
    results.sort(key=lambda r: r['best_val_rho'], reverse=True)

    sweep_stdout.write('\n' + '=' * 65 + '\n')
    sweep_stdout.write(f'Sweep complete. Full results: {results_csv}\n')
    sweep_stdout.write('Top 10 configurations by validation Spearman rho:\n\n')
    header = (
        f"{'Run':>4}  {'hidden_dims':>18}  {'drop':>5}"
        f"  {'lr':>8}  {'bs':>4}  {'val_rho':>8}"
    )
    sweep_stdout.write(header + '\n')
    sweep_stdout.write('-' * len(header) + '\n')
    for r in results[:10]:
        sweep_stdout.write(
            f"{r['run_id']:>4}  {str(r['hidden_dims']):>18}  {r['dropout_rate']:>5.1f}"
            f"  {r['lr']:>8.0e}  {r['batch_size']:>4}  {r['best_val_rho']:>8.4f}\n"
        )
    sweep_stdout.write('=' * 65 + '\n')
    sweep_stdout.write(
        '\nNext step: copy the winning config into train_mem.py and run with\n'
        '  train_fraction=1.0  and  early_stopping_patience=10  across all 5 folds.\n'
    )


if __name__ == '__main__':
    run_sweep()
