"""Train final MLP memorability heads for every MEG timepoint across all 10 folds.

Reads winning hyperparameters from meg_hyperparam_search_results.csv and for each
timepoint trains one MLPOnlyHead per fold (10 folds) on 100% of the
combined_lamem_memcat training split, with val held out for early stopping and
test held out for final evaluation.

After each fold's training completes, the best checkpoint (saved at peak val
performance) is loaded and evaluated on the held-out test set.  Per-timepoint
mean ± std of test MSE and Spearman ρ across all 10 folds are written to a
summary CSV in ./results/.

The fold loop runs inside this script (consistent with train_mem_meg_hyperparam_search.py).
To parallelize across timepoints, pass --timepoints to restrict a given invocation to a
subset and run multiple instances concurrently (e.g., one SLURM job per timepoint).

Usage
-----
    cd CLIP-HBA

    # All timepoints, all 10 folds:
    python train_mem_meg_final.py

    # Subset of timepoints:
    python train_mem_meg_final.py --timepoints 100 200 300

    # Subset of folds:
    python train_mem_meg_final.py --folds 1 2 3

    # Full options:
    python train_mem_meg_final.py --epochs 300 --cuda 0 --seed 1 --data_dir ./Data

Prerequisites
-------------
    Pre-extract MEG embeddings for all folds once:
        python extract_embeddings_meg.py --meg_checkpoint ./models/cliphba_meg_group.pth
"""

import argparse
import ast
import datetime
import glob as _glob
import os
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from functions.train_mem_pipeline import (
    EmbeddingDataset,
    MLPOnlyHead,
    evaluate_mem_model,
    run_mem_training,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_hidden_dims(s: str) -> tuple:
    """Convert a string like '(256, 64)' to the tuple (256, 64)."""
    result = ast.literal_eval(s)
    if isinstance(result, int):
        result = (result,)
    return tuple(result)


def load_hyperparams(csv_path: pathlib.Path) -> dict:
    """Load and parse the hyperparameter CSV.

    Returns a dict mapping timepoint (int, ms) to a hyperparameter dict with keys:
        hidden_dims (tuple), dropout (float), lr (float),
        weight_decay (float), batch_size (int).
    """
    df = pd.read_csv(csv_path)
    # Strip leading/trailing whitespace from column names (the CSV has
    # 'weight_decay ' with a trailing space).
    df.columns = df.columns.str.strip()

    hparams: dict = {}
    for _, row in df.iterrows():
        tp = int(row['timepoint'])
        hparams[tp] = {
            'hidden_dims': parse_hidden_dims(row['hidden_dims']),
            'dropout':     float(row['dropout']),
            'lr':          float(row['lr']),
            'weight_decay': float(row['weight_decay']),
            'batch_size':  int(row['batch_size']),
        }
    return hparams


def get_device(cuda: int) -> torch.device:
    """Resolve the torch device from the --cuda argument."""
    if cuda == -1:
        return torch.device('cuda')
    if cuda == 0:
        return torch.device('cuda:0')
    if cuda == 1:
        return torch.device('cuda:1')
    return torch.device('cpu')


def build_config(
    tp: int,
    fold: int,
    hparams: dict,
    args: argparse.Namespace,
) -> dict:
    """Construct the training config dict for one (timepoint, fold) run.

    Mirrors the config structure used in train_mem_meg.py so that the same
    run_mem_training() pipeline function is reused without modification.
    """
    data_dir = args.data_dir

    train_csv = f'{data_dir}/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv'
    val_csv   = f'{data_dir}/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv'
    test_csv  = f'{data_dir}/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv'

    img_root = {
        'lamem':  os.environ.get('LAMEM_IMG_ROOT',  f'{data_dir}/lamem/images/'),
        'memcat': os.environ.get('MEMCAT_IMG_ROOT', f'{data_dir}/memcat/images/'),
    }
    memcat_meta_csv = os.environ.get(
        'MEMCAT_META_CSV', f'{data_dir}/memcat/memcat_image_data.csv'
    )

    embeddings_dir = (
        args.embeddings_dir
        or f'{data_dir}/combined_lamem_memcat/meg_embeddings/'
    )

    # model_type encodes the timepoint so _run_mem_training_impl resolves the
    # correct embedding file: {model_type}_fold{fold}_{split}.pt
    model_type = f'clip_hba_meg_mem_tp{tp}'

    checkpoint_path = (
        f'./models/clip_hba_meg_mem_final/tp{tp}/'
        f'clip_hba_meg_mem_tp{tp}_final'
    )
    log_path = f'./logs/meg_mem_final/meg_mem_tp{tp}_fold{fold}.log'

    return {
        'model_type':    model_type,
        'input_dim':     66,
        'training_data': 'combined_lamem_memcat',

        # Data splits
        'fold':      fold,
        'train_csv': train_csv,
        'val_csv':   val_csv,
        'test_csv':  test_csv,
        'img_root':        img_root,
        'memcat_meta_csv': memcat_meta_csv,
        'preds_dir': './preds/meg_mem_final/',

        # Precomputed MEG embeddings
        'embeddings_dir': embeddings_dir,

        # MLP head architecture (from sweep CSV)
        'hidden_dims':  hparams['hidden_dims'],
        'dropout_rate': hparams['dropout'],   # CSV key 'dropout' → config key 'dropout_rate'

        # Optimisation (from sweep CSV)
        'batch_size':  hparams['batch_size'],
        'lr':          hparams['lr'],
        'weight_decay': hparams['weight_decay'],

        # Training schedule
        'epochs':                   args.epochs,
        'early_stopping_patience':  args.early_stopping_patience,

        # train_fraction is intentionally omitted so the pipeline defaults to
        # 1.0 — i.e., 100% of the training split is used.

        # Checkpointing
        'save_checkpoint': True,
        'checkpoint_path': checkpoint_path,

        # Logging
        'log_path': log_path,

        # Device & reproducibility
        'cuda':        args.cuda,
        'random_seed': args.seed,
        'criterion':   nn.MSELoss(),
    }


def evaluate_on_test_from_checkpoint(
    tp: int,
    fold: int,
    hparams: dict,
    config: dict,
    device: torch.device,
) -> tuple:
    """Load the best checkpoint and evaluate on the held-out test set.

    The best checkpoint is identified by globbing for files matching the
    naming pattern written by train_mem_model:
        {checkpoint_path}_fold{fold}_{timestamp}.pth

    Args:
        tp:       Timepoint in ms.
        fold:     Fold index (1–10).
        hparams:  Hyperparameter dict for this timepoint.
        config:   Training config dict (provides checkpoint_path, embeddings_dir, etc.).
        device:   Torch device to run evaluation on.

    Returns:
        (test_mse, test_rho) — float pair, or (nan, nan) if no checkpoint found.
    """
    # Locate the checkpoint saved during this fold's training run.
    checkpoint_path = config['checkpoint_path']
    ckpt_dir = pathlib.Path(checkpoint_path).parent
    ckpt_stem = pathlib.Path(checkpoint_path).name
    pattern = str(ckpt_dir / f'{ckpt_stem}_fold{fold}_*.pth')
    matches = sorted(_glob.glob(pattern))

    if not matches:
        print(
            f'  [WARNING] No checkpoint found matching {pattern} — '
            f'skipping test eval for tp={tp} fold={fold}'
        )
        return float('nan'), float('nan')

    # Use the lexicographically last match (most recent timestamp if multiple runs).
    ckpt_path = matches[-1]
    print(f'  [Test eval] Loading checkpoint: {ckpt_path}')

    # Rebuild model with the same architecture as training.
    model = MLPOnlyHead(
        hidden_dims=hparams['hidden_dims'],
        dropout_rate=hparams['dropout'],
        input_dim=66,
    )
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    # Load the test split embedding file.
    model_type = config['model_type']
    emb_dir = pathlib.Path(config['embeddings_dir'])
    test_pt = emb_dir / f'{model_type}_fold{fold}_test.pt'

    if not test_pt.exists():
        print(
            f'  [WARNING] Test embedding file not found: {test_pt} — '
            f'skipping test eval for tp={tp} fold={fold}'
        )
        return float('nan'), float('nan')

    test_dataset = EmbeddingDataset(test_pt)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_mse, test_rho, _ = evaluate_mem_model(
        model, test_loader, device, nn.MSELoss()
    )
    print(
        f'  [Test eval] tp={tp:+d} ms  fold={fold}  '
        f'test_mse={test_mse:.4f}  test_rho={test_rho:.4f}'
    )
    return float(test_mse), float(test_rho)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Train final MLP memorability heads for every MEG timepoint '
            'across all 10 combined_lamem_memcat folds.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--csv',
        default='meg_hyperparam_search_results.csv',
        help='Path to the hyperparameter search results CSV.',
    )
    parser.add_argument(
        '--timepoints',
        nargs='+',
        type=int,
        default=None,
        metavar='MS',
        help='Restrict training to these timepoints (ms). '
             'Defaults to all timepoints in the CSV.',
    )
    parser.add_argument(
        '--folds',
        nargs='+',
        type=int,
        default=list(range(1, 11)),
        metavar='FOLD',
        help='Folds to train (1–10). Defaults to all 10 folds.',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Maximum number of training epochs per (timepoint, fold).',
    )
    parser.add_argument(
        '--early_stopping_patience',
        type=int,
        default=20,
        help='Early stopping patience (epochs without val improvement).',
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=int(os.environ.get('CUDA_DEVICE', 0)),
        help='GPU index (0, 1, …).  Use -1 for DataParallel, 2 for CPU.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=int(os.environ.get('RANDOM_SEED', 1)),
        help='Random seed (applied once per (timepoint, fold) run).',
    )
    parser.add_argument(
        '--data_dir',
        default=os.environ.get('DATA_DIR', './Data'),
        help='Root data directory. Also settable via DATA_DIR env var.',
    )
    parser.add_argument(
        '--embeddings_dir',
        default=None,
        help=(
            'Directory containing precomputed MEG .pt embedding files. '
            'Defaults to <data_dir>/combined_lamem_memcat/meg_embeddings/.'
        ),
    )
    parser.add_argument(
        '--summary_dir',
        default='./results',
        help='Directory to write the per-timepoint summary CSV.',
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load hyperparameters
    # ------------------------------------------------------------------
    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f'Hyperparameter CSV not found: {csv_path.resolve()}\n'
            'Run train_mem_meg_hyperparam_search.py first to generate it.'
        )

    all_hparams = load_hyperparams(csv_path)

    # Filter to requested timepoints (preserve CSV row order)
    timepoints = list(all_hparams.keys())
    if args.timepoints is not None:
        unknown = set(args.timepoints) - set(timepoints)
        if unknown:
            raise ValueError(
                f'Requested timepoints not found in CSV: {sorted(unknown)}'
            )
        timepoints = [tp for tp in timepoints if tp in args.timepoints]

    folds = args.folds
    device = get_device(args.cuda)

    total_runs = len(timepoints) * len(folds)
    print(f'\n{"="*60}')
    print(f'CLIP-HBA-MEG Final MLP Training')
    print(f'{"="*60}')
    print(f'  Timepoints : {timepoints}')
    print(f'  Folds      : {folds}')
    print(f'  Total runs : {total_runs}  ({len(timepoints)} tp × {len(folds)} folds)')
    print(f'  Epochs     : {args.epochs}  (patience={args.early_stopping_patience})')
    print(f'  CSV        : {csv_path}')
    print(f'  Data dir   : {args.data_dir}')
    print(f'  CUDA       : {args.cuda}  (device: {device})')
    print(f'  Seed       : {args.seed}')
    print(f'{"="*60}\n')

    # ------------------------------------------------------------------
    # Training loop: outer = timepoints, inner = folds
    # ------------------------------------------------------------------
    # fold_results: {tp: [{val_loss, val_rho, test_mse, test_rho}, ...]}
    fold_results: dict = {tp: [] for tp in timepoints}
    run_idx = 0

    for tp in timepoints:
        hparams = all_hparams[tp]
        print(f'\n{"─"*60}')
        print(f'Timepoint: {tp:+d} ms')
        print(
            f'  hidden_dims={hparams["hidden_dims"]}  '
            f'dropout={hparams["dropout"]:.4f}  '
            f'lr={hparams["lr"]:.2e}  '
            f'weight_decay={hparams["weight_decay"]:.4f}  '
            f'batch_size={hparams["batch_size"]}'
        )
        print(f'{"─"*60}')

        for fold in folds:
            run_idx += 1
            print(f'\n[Run {run_idx}/{total_runs}]  tp={tp:+d} ms  fold={fold}')

            config = build_config(tp, fold, hparams, args)

            # Train — uses 100% of the training split (no train_fraction set).
            # Val split is held out for early stopping; test split is internally
            # evaluated at early stopping time (current model state, logged to
            # the fold log file).  We re-evaluate below using the best checkpoint.
            best_val_loss, best_val_rho = run_mem_training(config)

            # Evaluate on test set using the best checkpoint.
            test_mse, test_rho = evaluate_on_test_from_checkpoint(
                tp, fold, hparams, config, device
            )

            fold_results[tp].append({
                'fold':         fold,
                'val_loss':     best_val_loss,
                'val_rho':      best_val_rho,
                'test_mse':     test_mse,
                'test_rho':     test_rho,
            })
            print(
                f'  → tp={tp:+d} ms  fold={fold}  '
                f'best_val_loss={best_val_loss:.4f}  best_val_rho={best_val_rho:.4f}  '
                f'test_mse={test_mse:.4f}  test_rho={test_rho:.4f}'
            )

    # ------------------------------------------------------------------
    # Aggregate and write summary
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_dir = pathlib.Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f'meg_mem_final_summary_{timestamp}.csv'

    summary_rows = []
    for tp in timepoints:
        rows = fold_results[tp]
        if not rows:
            continue

        val_losses  = [r['val_loss']  for r in rows if not np.isnan(r['val_loss'])]
        val_rhos    = [r['val_rho']   for r in rows if not np.isnan(r['val_rho'])]
        test_mses   = [r['test_mse']  for r in rows if not np.isnan(r['test_mse'])]
        test_rhos   = [r['test_rho']  for r in rows if not np.isnan(r['test_rho'])]

        summary_rows.append({
            'timepoint':       tp,
            'n_folds':         len(rows),
            'mean_val_mse':    float(np.mean(val_losses))  if val_losses  else float('nan'),
            'std_val_mse':     float(np.std(val_losses))   if val_losses  else float('nan'),
            'mean_val_rho':    float(np.mean(val_rhos))    if val_rhos    else float('nan'),
            'std_val_rho':     float(np.std(val_rhos))     if val_rhos    else float('nan'),
            'mean_test_mse':   float(np.mean(test_mses))   if test_mses   else float('nan'),
            'std_test_mse':    float(np.std(test_mses))    if test_mses   else float('nan'),
            'mean_test_rho':   float(np.mean(test_rhos))   if test_rhos   else float('nan'),
            'std_test_rho':    float(np.std(test_rhos))    if test_rhos   else float('nan'),
            # Per-fold detail columns (fold1_test_rho, fold2_test_rho, …)
            **{f'fold{r["fold"]}_val_mse':  r['val_loss']  for r in rows},
            **{f'fold{r["fold"]}_val_rho':  r['val_rho']   for r in rows},
            **{f'fold{r["fold"]}_test_mse': r['test_mse']  for r in rows},
            **{f'fold{r["fold"]}_test_rho': r['test_rho']  for r in rows},
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_path, index=False)
    print(f'\nSummary written to: {summary_path}')

    # ------------------------------------------------------------------
    # Console summary table
    # ------------------------------------------------------------------
    print(f'\n{"="*80}')
    print('Training complete — per-timepoint summary (mean ± std across folds)')
    print(f'{"="*80}')
    print(
        f'{"Timepoint":>10}  {"Folds":>5}  '
        f'{"Val MSE":>18}  {"Val ρ":>18}  '
        f'{"Test MSE":>18}  {"Test ρ":>18}'
    )
    print(
        f'{"":>10}  {"":>5}  '
        f'{"mean ± std":>18}  {"mean ± std":>18}  '
        f'{"mean ± std":>18}  {"mean ± std":>18}'
    )
    print(f'{"-"*10}  {"-"*5}  {"-"*18}  {"-"*18}  {"-"*18}  {"-"*18}')
    for r in summary_rows:
        print(
            f'{r["timepoint"]:>+10d}  {r["n_folds"]:>5d}  '
            f'{r["mean_val_mse"]:>8.4f} ± {r["std_val_mse"]:<7.4f}  '
            f'{r["mean_val_rho"]:>8.4f} ± {r["std_val_rho"]:<7.4f}  '
            f'{r["mean_test_mse"]:>8.4f} ± {r["std_test_mse"]:<7.4f}  '
            f'{r["mean_test_rho"]:>8.4f} ± {r["std_test_rho"]:<7.4f}'
        )
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
