"""Per-timepoint memorability MLP training using CLIP-HBA-MEG embeddings.

This script answers: **at which MEG timepoint is memorability best encoded?**

It loads pre-extracted CLIP-HBA-MEG embeddings (produced by
``extract_embeddings.py --model_type clip_hba_meg``), which are stored as
``Tensor[N, T, 768]`` float16.  For each of the T training timepoints it
trains a small MLP head (identical architecture to the CLIP-HBA-Behavior
memorability MLP) and records the validation Spearman SRCC.  The result is
a temporal profile CSV showing memorability correlation as a function of
MEG timepoint.

Usage
-----
    # Single LaMem fold:
    python train_meg_mem.py --fold 1

    # All 5 folds via SLURM:
    sbatch train_meg_mem.slurm

Outputs
-------
    <results_dir>/meg_mem_fold<N>_temporal_profile.csv
        Columns: timepoint_ms, val_srcc, test_srcc

    <checkpoint_dir>/meg_mem_fold<N>_t<ms>.pth   (best MLP per timepoint)

Notes on memory
---------------
The full [N, T, 768] float16 tensor for LaMem train (~45 k images, T=281)
is ~19 GB.  It is kept in CPU RAM and sliced per timepoint for training.
Ensure the SLURM node has ≥64 GB RAM.  Reduce ``--batch_size`` if GPU OOM
occurs.
"""

import argparse
import csv
import datetime
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.train_mem_pipeline import (
    MEGEmbeddingDataset,
    MLPOnlyHead,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Per-timepoint training
# ---------------------------------------------------------------------------

def _train_one_timepoint(
    t_idx: int,
    t_ms: int,
    train_pt: str,
    val_pt: str,
    test_pt: str,
    hidden_dims: tuple,
    dropout_rate: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    batch_size: int,
    early_stopping_patience: int,
    device: torch.device,
    checkpoint_dir: 'pathlib.Path | None',
    fold: int,
    random_seed: int,
) -> 'tuple[float, float]':
    """Train an MLP on the ``t_idx``-th timepoint's embeddings.

    Args:
        t_idx: Index into the T dimension of the embedding tensor.
        t_ms: Timepoint in milliseconds (for logging / filenames).
        train_pt / val_pt / test_pt: Paths to the .pt embedding files.
        hidden_dims: MLP hidden layer sizes.
        dropout_rate: MLP dropout probability.
        lr: AdamW learning rate.
        weight_decay: AdamW weight decay.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        early_stopping_patience: Stop after this many epochs without val improvement.
        device: Compute device.
        checkpoint_dir: Directory to save best-checkpoint .pth files.  ``None``
            disables checkpoint saving.
        fold: Fold number (used in checkpoint filenames).
        random_seed: Seed for reproducibility within this timepoint run.

    Returns:
        (val_srcc, test_srcc) at the best-val-loss epoch.
    """
    seed_everything(random_seed)

    train_ds = MEGEmbeddingDataset(train_pt, t_idx)
    val_ds   = MEGEmbeddingDataset(val_pt,   t_idx)
    test_ds  = MEGEmbeddingDataset(test_pt,  t_idx)

    _g = torch.Generator()
    _g.manual_seed(random_seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, generator=_g)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    model = MLPOnlyHead(hidden_dims=hidden_dims, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_val_srcc = float('-inf')
    best_test_srcc = float('nan')
    epochs_no_improve = 0
    chk_path = (checkpoint_dir / f'meg_mem_fold{fold}_t{t_ms:+d}ms.pth'
                if checkpoint_dir is not None else None)

    for epoch in range(epochs):
        model.train()
        for _, emb, targets in train_loader:
            emb, targets = emb.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(emb).squeeze(1), targets)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        val_loss_sum = 0.0
        with torch.no_grad():
            for _, emb, targets in val_loader:
                emb, targets = emb.to(device), targets.to(device)
                preds = model(emb).squeeze(1)
                val_loss_sum += criterion(preds, targets).item() * emb.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        avg_val_loss = val_loss_sum / len(val_ds)
        val_srcc, _ = spearmanr(val_preds, val_targets)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_srcc = val_srcc
            epochs_no_improve = 0

            # Evaluate test split at the best checkpoint
            test_preds, test_targets = [], []
            with torch.no_grad():
                for _, emb, targets in test_loader:
                    emb, targets = emb.to(device), targets.to(device)
                    test_preds.extend(model(emb).squeeze(1).cpu().numpy())
                    test_targets.extend(targets.cpu().numpy())
            best_test_srcc, _ = spearmanr(test_preds, test_targets)

            if chk_path is not None:
                torch.save(model.state_dict(), chk_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            break

    return float(best_val_srcc), float(best_test_srcc)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_meg_mem_training(config: dict) -> None:
    """Train per-timepoint memorability MLPs and save the temporal profile.

    Args:
        config: Training configuration dictionary.  See ``main()`` for the
            expected keys.
    """
    seed_everything(config['random_seed'])

    fold        = config['fold']
    emb_dir     = pathlib.Path(config['embeddings_dir'])
    results_dir = pathlib.Path(config.get('results_dir', './results/meg_mem/'))
    chk_dir_cfg = config.get('checkpoint_dir', None)
    checkpoint_dir = pathlib.Path(chk_dir_cfg) if chk_dir_cfg else None

    results_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_pt = str(emb_dir / f'clip_hba_meg_fold{fold}_train.pt')
    val_pt   = str(emb_dir / f'clip_hba_meg_fold{fold}_val.pt')
    test_pt  = str(emb_dir / f'clip_hba_meg_fold{fold}_test.pt')

    # Read the timepoints array from the embedding file (no need to load embeddings yet)
    _meta = torch.load(train_pt, map_location='cpu', weights_only=True)
    timepoints_ms: torch.Tensor = _meta['timepoints_ms']  # [T]
    T = len(timepoints_ms)
    del _meta

    device_cfg = config.get('cuda', 0)
    if device_cfg == -1:
        device = torch.device('cuda')
    elif device_cfg == 2:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{device_cfg}')

    run_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    profile_path = results_dir / f'meg_mem_fold{fold}_{run_ts}_temporal_profile.csv'

    print(f'\n[MEGMem] Fold {fold} | T={T} timepoints | device={device}')
    print(f'[MEGMem] Embeddings: {emb_dir}')
    print(f'[MEGMem] Results   : {profile_path}\n')

    fieldnames = ['timepoint_ms', 'val_srcc', 'test_srcc']
    profile_file = open(profile_path, 'w', newline='')
    writer = csv.DictWriter(profile_file, fieldnames=fieldnames)
    writer.writeheader()
    profile_file.flush()

    val_srccs  = []
    test_srccs = []

    for t_idx in tqdm(range(T), desc='Timepoints'):
        t_ms = int(timepoints_ms[t_idx].item())

        val_srcc, test_srcc = _train_one_timepoint(
            t_idx=t_idx,
            t_ms=t_ms,
            train_pt=train_pt,
            val_pt=val_pt,
            test_pt=test_pt,
            hidden_dims=config.get('hidden_dims', (256, 128)),
            dropout_rate=config.get('dropout_rate', 0.5),
            lr=config.get('lr', 1e-5),
            weight_decay=config.get('weight_decay', 1e-2),
            epochs=config.get('epochs', 300),
            batch_size=config.get('batch_size', 32),
            early_stopping_patience=config.get('early_stopping_patience', 12),
            device=device,
            checkpoint_dir=checkpoint_dir,
            fold=fold,
            random_seed=config['random_seed'],
        )

        val_srccs.append(val_srcc)
        test_srccs.append(test_srcc)

        writer.writerow({
            'timepoint_ms': t_ms,
            'val_srcc':     round(val_srcc,  4),
            'test_srcc':    round(test_srcc, 4),
        })
        profile_file.flush()

        tqdm.write(f'  t={t_ms:+5d} ms  |  val SRCC={val_srcc:.4f}  |  test SRCC={test_srcc:.4f}')

    profile_file.close()

    best_idx = int(np.argmax(val_srccs))
    print(f'\n[MEGMem] Done.  Best val SRCC={val_srccs[best_idx]:.4f} '
          f'at t={int(timepoints_ms[best_idx].item()):+d} ms '
          f'(test SRCC={test_srccs[best_idx]:.4f})')
    print(f'[MEGMem] Temporal profile saved -> {profile_path}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train per-timepoint memorability MLPs on CLIP-HBA-MEG embeddings.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--fold', type=int,
                        default=int(os.environ.get('FOLD', 1)),
                        help='LaMem fold number (1–5).')
    parser.add_argument('--embeddings_dir',
                        default=os.environ.get('MEG_EMB_DIR', './Data/lamem/meg_embeddings/'),
                        help='Directory containing clip_hba_meg_fold<N>_*.pt files.')
    parser.add_argument('--results_dir',
                        default='./results/meg_mem/',
                        help='Directory for temporal profile CSVs.')
    parser.add_argument('--checkpoint_dir',
                        default=None,
                        help='Directory to save best-epoch MLP checkpoints per timepoint. '
                             'Omit to skip checkpoint saving.')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[256, 128],
                        help='MLP hidden layer sizes (e.g. --hidden_dims 256 128).')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--lr',           type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--epochs',       type=int,   default=300)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--early_stopping_patience', type=int, default=12)
    parser.add_argument('--cuda', type=int,
                        default=int(os.environ.get('CUDA_DEVICE', 0)),
                        help='GPU index.  Use 2 for CPU.')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    config = {
        'fold':                    args.fold,
        'embeddings_dir':          args.embeddings_dir,
        'results_dir':             args.results_dir,
        'checkpoint_dir':          args.checkpoint_dir,
        'hidden_dims':             tuple(args.hidden_dims),
        'dropout_rate':            args.dropout_rate,
        'lr':                      args.lr,
        'weight_decay':            args.weight_decay,
        'epochs':                  args.epochs,
        'batch_size':              args.batch_size,
        'early_stopping_patience': args.early_stopping_patience,
        'cuda':                    args.cuda,
        'random_seed':             args.seed,
    }

    run_meg_mem_training(config)


if __name__ == '__main__':
    main()
