"""Bayesian hyperparameter search for the memorability MLP head using Optuna.

Replaces the exhaustive grid search with Tree-structured Parzen Estimator (TPE),
a Bayesian optimisation algorithm that builds a probabilistic model of the
objective and focuses subsequent trials in promising regions of the search space.
This is substantially more sample-efficient than grid search: good configurations
are typically found in 30–60 trials rather than exhaustively enumerating all
combinations.

Each trial trains on all 5 LaMem folds and uses the mean validation MSE across
folds as the Optuna objective.  This makes the search robust to fold-specific
variance and eliminates the need for a separate final cross-validation run to
validate the winning config.

Usage
-----
    cd CLIP-HBA
    python train_mem_sweep.py [--n-trials N] [--resume DB]

Arguments
---------
    --n-trials N   Number of Optuna trials to run (default: 50).
    --resume DB    Path to an existing optuna_study.db to resume a study.
                   Completed trials are automatically skipped by Optuna;
                   no manual bookkeeping required.

Outputs
-------
    sweep_out/<timestamp>/run_NNN/fold_K/log.txt     -- per-fold training log
    sweep_out/<timestamp>/run_NNN/config.json        -- sampled hyperparameters
    sweep_out/<timestamp>/sweep_results.csv          -- all trials, mean ± std across folds
    sweep_out/<timestamp>/optuna_study.db            -- SQLite study for resuming / analysis
"""

import argparse
import csv
import datetime
import ast
import json
import math
import os
import sys

import torch.nn as nn

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError as _e:
    raise ImportError(
        "Optuna is required for Bayesian hyperparameter search.\n"
        "Install it with:  pip install optuna"
    ) from _e

from functions.train_mem_pipeline import run_mem_training

# ---------------------------------------------------------------------------
# Fixed configuration — edit paths and device to match your environment
# ---------------------------------------------------------------------------
BASE_CONFIG = {
    'model_type': 'clip_hba_mem',  # sweep only covers the CLIPHBAMem head

    # fold/train_csv/val_csv/test_csv are injected per fold inside _objective
    'img_root':  './Data/lamem/images/',
    'embeddings_dir': './Data/lamem/embeddings/',

    # Backbone (frozen — these must match the checkpoint's DoRA config)
    #'backbone_checkpoint': './Data/lamem/epoch97_dora_params.pth',
    #'backbone':            'ViT-L/14',
    #'vision_layers':       2,
    #'transformer_layers':  1,
    #'rank':                32,

    # Device: 0=cuda:0, 1=cuda:1, -1=DataParallel, 2=cpu
    'cuda': 1,

    # Fixed training settings for the sweep
    'epochs':                  300,
    'early_stopping_patience': 20,    # shorter than final training to speed up sweep
    'train_fraction':          1.0,   # full training set — epochs are ~3 s with precomputed embeddings
    'criterion':               nn.MSELoss(),
    'random_seed':             42,
}

N_FOLDS: int = 5
LAMEM_CSV_PATTERN: str = './Data/lamem/lamem_{split}_{fold}.csv'

# ---------------------------------------------------------------------------
# Search space
#
# Continuous parameters (lr, dropout_rate) are sampled from distributions,
# giving the TPE sampler far more flexibility than a fixed grid.
# Categorical parameters keep the same options as the original grid search.
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict = {
    # Backbone — HBA-tuned CLIP vs. vanilla openai/clip-vit-large-patch14
    'backbone':     ['hba', 'vanilla_clip'],

    # Architecture — categorical; same options as the original grid
    'hidden_dims':  ['(512, 256)', '(256, 128)', '(512, 256, 128)', '(256,)'],

    # Regularisation — continuous uniform; wider than the original [0.3, 0.5]
    'dropout_rate': (0.1, 0.7),

    # Optimisation — log-uniform; covers the original [1e-5, 1e-4] range and beyond
    'lr':           (1e-5, 5e-4),

    # Regularisation — AdamW weight decay, log-uniform over a wide range
    'weight_decay': (1e-5, 1e-1),

    # Batch size — categorical; extended to include 16 and 128
    'batch_size':   [32, 64, 128, 256],
}

N_TRIALS_DEFAULT: int = 50

# New sweep directory (only used when not resuming)
SWEEP_DIR: str = f'./sweep_out/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'

_CSV_FIELDS = [
    'trial_id', 'backbone', 'hidden_dims', 'dropout_rate', 'lr', 'weight_decay', 'batch_size',
    'mean_val_mse', 'std_val_mse', 'mean_val_rho', 'std_val_rho',
]

# HBA backbone config keys not applicable to vanilla CLIP — removed from fold_config
# when backbone='vanilla_clip' to avoid passing irrelevant args to CLIPFrozenMLP.
_HBA_ONLY_KEYS: frozenset = frozenset({
    'backbone_checkpoint', 'backbone', 'vision_layers', 'transformer_layers', 'rank',
})


def _sample_std(values: list) -> float:
    """Return the sample standard deviation of *values*, or 0 if len < 2."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _objective(
    trial: 'optuna.Trial',
    sweep_dir: str,
    results_csv: str,
    sweep_stdout,
) -> float:
    """Sample hyperparameters, train on all 5 folds, return mean validation MSE.

    Running all folds per trial makes the objective robust to fold-specific
    variance and removes the need for a post-hoc cross-validation step to
    validate the winning config.  MSE is used because it directly drives
    checkpointing and early stopping inside train_mem_model.  Spearman rho
    is logged in the CSV as a secondary diagnostic metric.

    Args:
        trial:        Optuna trial object used for hyperparameter suggestion.
        sweep_dir:    Root directory for this sweep's per-trial subdirectories.
        results_csv:  Path to the incremental CSV results file.
        sweep_stdout: Reference to the real stdout so progress is always visible
                      even though run_mem_training redirects stdout to a log file.

    Returns:
        Mean best validation MSE across all folds (Optuna minimises this).

    Raises:
        Exception: Re-raised after logging so Optuna marks the trial as FAILED.
    """
    # --- Sample hyperparameters ---
    backbone:     str   = trial.suggest_categorical('backbone',     SEARCH_SPACE['backbone'])
    hidden_dims:  tuple = ast.literal_eval(
        trial.suggest_categorical('hidden_dims', SEARCH_SPACE['hidden_dims'])
    )
    dropout_rate: float = trial.suggest_float('dropout_rate', *SEARCH_SPACE['dropout_rate'])
    lr:           float = trial.suggest_float('lr', *SEARCH_SPACE['lr'], log=True)
    weight_decay: float = trial.suggest_float('weight_decay', *SEARCH_SPACE['weight_decay'], log=True)
    batch_size:   int   = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size'])

    trial_id = trial.number
    run_dir  = os.path.join(sweep_dir, f'run_{trial_id:03d}')
    os.makedirs(run_dir, exist_ok=True)

    sweep_stdout.write(
        f'\n[Trial {trial_id}] '
        f'backbone={backbone}  hidden_dims={hidden_dims}  dropout={dropout_rate:.3f}'
        f'  lr={lr:.2e}  wd={weight_decay:.2e}  batch_size={batch_size}\n'
    )
    sweep_stdout.flush()

    # Save shared hyperparameters once (fold-specific paths are not included)
    with open(os.path.join(run_dir, 'config.json'), 'w') as _f:
        json.dump(
            {
                'backbone':     backbone,
                'hidden_dims':  str(hidden_dims),
                'dropout_rate': dropout_rate,
                'lr':           lr,
                'weight_decay': weight_decay,
                'batch_size':   batch_size,
            },
            _f, indent=2,
        )

    # --- Train on all folds and collect per-fold metrics ---
    fold_mses: list[float] = []
    fold_rhos: list[float] = []

    try:
        for fold in range(1, N_FOLDS + 1):
            fold_dir = os.path.join(run_dir, f'fold_{fold}')
            os.makedirs(fold_dir, exist_ok=True)

            fold_config = {
                **BASE_CONFIG,
                'fold':            fold,
                'train_csv':       LAMEM_CSV_PATTERN.format(split='train', fold=fold),
                'val_csv':         LAMEM_CSV_PATTERN.format(split='val',   fold=fold),
                'test_csv':        LAMEM_CSV_PATTERN.format(split='test',  fold=fold),
                'hidden_dims':     hidden_dims,
                'dropout_rate':    dropout_rate,
                'lr':              lr,
                'weight_decay':    weight_decay,
                'batch_size':      batch_size,
                'log_path':        os.path.join(fold_dir, 'log.txt'),
                'checkpoint_path': os.path.join(fold_dir, 'checkpoint'),
            }
            if backbone == 'vanilla_clip':
                # Switch to the frozen vanilla CLIP backbone; HBA-specific config
                # keys are not consumed by CLIPFrozenMLP and must be removed to
                # avoid confusing the pipeline.
                fold_config['model_type'] = 'clip_frozen_mlp'
                for k in _HBA_ONLY_KEYS:
                    fold_config.pop(k, None)

            fold_mse, fold_rho = run_mem_training(fold_config)
            fold_mses.append(fold_mse)
            fold_rhos.append(fold_rho)

            sweep_stdout.write(
                f'  fold {fold}/{N_FOLDS}: mse={fold_mse:.6f}  rho={fold_rho:.4f}\n'
            )
            sweep_stdout.flush()

    except Exception as exc:
        sweep_stdout.write(f'  [ERROR] trial {trial_id} failed on fold {fold}: {exc}\n')
        sweep_stdout.flush()
        raise  # Let Optuna mark this trial as FAILED

    mean_mse = sum(fold_mses) / N_FOLDS
    std_mse  = _sample_std(fold_mses)
    mean_rho = sum(fold_rhos) / N_FOLDS
    std_rho  = _sample_std(fold_rhos)

    # Append aggregated result to incremental CSV
    with open(results_csv, 'a', newline='') as f:
        csv.writer(f).writerow([
            trial_id, backbone, str(hidden_dims), f'{dropout_rate:.6f}',
            f'{lr:.2e}', f'{weight_decay:.2e}', batch_size,
            f'{mean_mse:.6f}', f'{std_mse:.6f}',
            f'{mean_rho:.6f}', f'{std_rho:.6f}',
        ])

    sweep_stdout.write(
        f'  -> mean_val_mse = {mean_mse:.6f} ± {std_mse:.6f}'
        f'  |  mean_val_rho = {mean_rho:.4f} ± {std_rho:.4f}\n'
    )
    sweep_stdout.flush()

    return mean_mse  # minimise mean MSE across folds


# ---------------------------------------------------------------------------
# Main sweep entry point
# ---------------------------------------------------------------------------

def _write_sweep_config(sweep_dir: str, n_trials: int, resumed: bool) -> None:
    """Write (or update) sweep_config.json with metadata for this sweep run.

    On a fresh sweep the full configuration is written.  On resume, a compact
    event record is appended to the existing file so the full history of runs
    against this study is preserved.

    Args:
        sweep_dir: Root directory of the sweep (contains the Optuna DB).
        n_trials:  Number of trials requested for this invocation.
        resumed:   True when continuing an existing study, False for a fresh one.
    """
    import subprocess

    config_path = os.path.join(sweep_dir, 'sweep_config.json')

    # Serialisable subset of BASE_CONFIG — exclude nn.Module and other non-JSON types.
    base_config_serialisable = {
        k: v for k, v in BASE_CONFIG.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    }

    try:
        git_hash: str | None = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        git_hash = None

    timestamp = datetime.datetime.now().isoformat(timespec='seconds')

    if not resumed:
        config: dict = {
            'started_at':        timestamp,
            'n_trials_requested': n_trials,
            'model_type':        BASE_CONFIG.get('model_type', 'unknown'),
            'n_folds':           N_FOLDS,
            'lamem_csv_pattern': LAMEM_CSV_PATTERN,
            'base_config':       base_config_serialisable,
            'search_space': {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in SEARCH_SPACE.items()
            },
            'optuna_sampler':    'TPESampler',
            'python_version':    sys.version,
            'git_commit':        git_hash,
            'resume_events':     [],
        }
    else:
        with open(config_path, encoding='utf-8') as _f:
            config = json.load(_f)
        config.setdefault('resume_events', []).append({
            'resumed_at':         timestamp,
            'n_trials_requested': n_trials,
            'git_commit':         git_hash,
        })

    with open(config_path, 'w', encoding='utf-8') as _f:
        json.dump(config, _f, indent=2)



def run_sweep(n_trials: int = N_TRIALS_DEFAULT, resume_db: str | None = None) -> None:
    """Run (or resume) a Bayesian hyperparameter search over the MLP head.

    Args:
        n_trials:  Number of Optuna trials to execute.
        resume_db: Path to an existing ``optuna_study.db``.  When provided,
                   the sweep directory is inferred from the DB's parent folder
                   and all previously completed trials are loaded automatically.
    """
    sweep_dir = os.path.dirname(resume_db) if resume_db else SWEEP_DIR
    os.makedirs(sweep_dir, exist_ok=True)

    _write_sweep_config(sweep_dir, n_trials, resumed=resume_db is not None)

    sweep_stdout = sys.stdout

    results_csv = os.path.join(sweep_dir, 'sweep_results.csv')
    db_path     = resume_db if resume_db else os.path.join(sweep_dir, 'optuna_study.db')
    storage_url = f'sqlite:///{db_path}'

    # Write CSV header only for brand-new studies
    if not os.path.exists(results_csv):
        with open(results_csv, 'w', newline='') as f:
            csv.writer(f).writerow(_CSV_FIELDS)

    # Create (or resume) the Optuna study.
    # TPESampler is Optuna's Bayesian (TPE) sampler — the default, but made
    # explicit here with a fixed seed for reproducibility.
    study = optuna.create_study(
        study_name='clip_hba_mem_bayesian_sweep',
        direction='minimize',
        sampler=TPESampler(seed=BASE_CONFIG['random_seed']),
        storage=storage_url,
        load_if_exists=True,  # seamless resume when DB already exists
    )

    n_done = sum(
        1 for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    )
    if n_done > 0:
        sweep_stdout.write(
            f'[Resume] {n_done} completed trial(s) loaded from {db_path}\n'
        )

    sweep_stdout.write(
        f'Bayesian search: {n_trials} new trial(s) | TPE sampler | '
        f'{N_FOLDS}-fold CV per trial | DB: {db_path}\n'
        f'Search space:\n'
        f'  backbone     (categorical): {SEARCH_SPACE["backbone"]}\n'
        f'  hidden_dims  (categorical): {SEARCH_SPACE["hidden_dims"]}\n'
        f'  dropout_rate (uniform):     [{SEARCH_SPACE["dropout_rate"][0]}, '
        f'{SEARCH_SPACE["dropout_rate"][1]}]\n'
        f'  lr           (log-uniform): [{SEARCH_SPACE["lr"][0]:.0e}, '
        f'{SEARCH_SPACE["lr"][1]:.0e}]\n'
        f'  weight_decay (log-uniform): [{SEARCH_SPACE["weight_decay"][0]:.0e}, '
        f'{SEARCH_SPACE["weight_decay"][1]:.0e}]\n'
        f'  batch_size   (categorical): {SEARCH_SPACE["batch_size"]}\n\n'
    )
    sweep_stdout.flush()

    study.optimize(
        lambda trial: _objective(trial, sweep_dir, results_csv, sweep_stdout),
        n_trials=n_trials,
        catch=(Exception,),
    )

    # --- Final leaderboard ---
    all_results = []
    with open(results_csv, newline='') as f:
        for row in csv.DictReader(f):
            try:
                mean_mse = float(row['mean_val_mse'])
                std_mse  = float(row['std_val_mse'])
                mean_rho = float(row['mean_val_rho'])
                std_rho  = float(row['std_val_rho'])
            except ValueError:
                mean_mse = std_mse = mean_rho = std_rho = float('nan')
            all_results.append({
                'trial_id':     int(row['trial_id']),
                'backbone':     row['backbone'],
                'hidden_dims':  row['hidden_dims'],
                'dropout_rate': float(row['dropout_rate']),
                'lr':           float(row['lr']),
                'weight_decay': float(row['weight_decay']),
                'batch_size':   int(row['batch_size']),
                'mean_val_mse': mean_mse,
                'std_val_mse':  std_mse,
                'mean_val_rho': mean_rho,
                'std_val_rho':  std_rho,
            })

    valid = [r for r in all_results if not math.isnan(r['mean_val_mse'])]
    valid.sort(key=lambda r: r['mean_val_mse'])  # ascending: lower MSE is better

    sweep_stdout.write('\n' + '=' * 80 + '\n')
    sweep_stdout.write(f'Search complete. Full results: {results_csv}\n')
    sweep_stdout.write(f'Optuna study DB (for resuming / analysis): {db_path}\n')
    sweep_stdout.write(
        f'Top 10 configurations by mean validation MSE across {N_FOLDS} folds'
        f' (lower is better):\n\n'
    )
    header = (
        f"{'Trial':>5}  {'backbone':>12}  {'hidden_dims':>18}  {'drop':>5}"
        f"  {'lr':>8}  {'wd':>8}  {'bs':>4}"
        f"  {'mean_mse':>10}  {'std_mse':>9}  {'mean_rho':>9}  {'std_rho':>8}"
    )
    sweep_stdout.write(header + '\n')
    sweep_stdout.write('-' * len(header) + '\n')
    for r in valid[:10]:
        sweep_stdout.write(
            f"{r['trial_id']:>5}  {r['backbone']:>12}  {str(r['hidden_dims']):>18}"
            f"  {r['dropout_rate']:>5.2f}"
            f"  {r['lr']:>8.0e}  {r['weight_decay']:>8.0e}  {r['batch_size']:>4}"
            f"  {r['mean_val_mse']:>10.6f}  {r['std_val_mse']:>9.6f}"
            f"  {r['mean_val_rho']:>9.4f}  {r['std_val_rho']:>8.4f}\n"
        )

    if valid:
        best = valid[0]
        sweep_stdout.write('\n' + '=' * 80 + '\n')
        sweep_stdout.write(f'Best configuration (trial {best["trial_id"]}):\n')
        sweep_stdout.write(f'  backbone:     {best["backbone"]}\n')
        sweep_stdout.write(f'  hidden_dims:  {best["hidden_dims"]}\n')
        sweep_stdout.write(f'  dropout_rate: {best["dropout_rate"]:.4f}\n')
        sweep_stdout.write(f'  lr:           {best["lr"]:.2e}\n')
        sweep_stdout.write(f'  weight_decay: {best["weight_decay"]:.2e}\n')
        sweep_stdout.write(f'  batch_size:   {best["batch_size"]}\n')
        sweep_stdout.write(
            f'  mean_val_mse: {best["mean_val_mse"]:.6f} ± {best["std_val_mse"]:.6f}\n'
        )
        sweep_stdout.write(
            f'  mean_val_rho: {best["mean_val_rho"]:.4f} ± {best["std_val_rho"]:.4f}\n'
        )

    sweep_stdout.write('=' * 80 + '\n')
    sweep_stdout.write(
        '\nNext step: copy the winning config into train_mem.py.\n'
        'The sweep already used all 5 folds — no separate cross-validation run needed.\n'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bayesian hyperparameter search for CLIPHBAMem using Optuna TPE.'
    )
    parser.add_argument(
        '--n-trials', type=int, default=N_TRIALS_DEFAULT, metavar='N',
        help=f'Number of Optuna trials to run (default: {N_TRIALS_DEFAULT}).',
    )
    parser.add_argument(
        '--resume', metavar='DB', default=None,
        help='Path to an existing optuna_study.db to resume a previous search. '
             'All completed trials are loaded automatically; no re-running.',
    )
    args = parser.parse_args()
    run_sweep(n_trials=args.n_trials, resume_db=args.resume)
