"""Train a memorability MLP head for a single CLIP-HBA-MEG timepoint.

This script trains one MLP head per sampled MEG timepoint.  The head operates
on precomputed 66-dim CLIP-HBA-MEG embeddings (extracted by
``extract_embeddings_meg.py``) and is trained with the same MSE + early-stopping
procedure as the CLIP-HBA-Behavior memorability head.

A separate invocation is required per timepoint; run them in parallel via SLURM
(one job per timepoint).

Usage
-----
    cd CLIP-HBA

    # Single fold, single timepoint:
    FOLD=1 TIMEPOINT_MS=100 python train_mem_meg.py

    # Or with explicit arguments:
    python train_mem_meg.py --timepoint_ms 100 --fold 1 --cuda 0

    # All timepoints sequentially (consider SLURM parallelism instead):
    for TP in -100 0 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300; do
        for FOLD in $(seq 1 10); do
            python train_mem_meg.py --timepoint_ms $TP --fold $FOLD --cuda 0
        done
    done

Prerequisites
-------------
    Pre-extract MEG embeddings once:
        python extract_embeddings_meg.py --meg_checkpoint ./models/cliphba_meg_group.pth

    Copy winning hyperparameters from the sweep into the config block below
    (or pass them as environment variables / command-line arguments).
"""

import argparse
import os

import torch.nn as nn

from functions.train_mem_pipeline import run_mem_training

# ---------------------------------------------------------------------------
# Timepoint parameters
# ---------------------------------------------------------------------------
_ALL_TIMEPOINTS: list[int] = list(range(-100, 1301, 100))  # 15 timepoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Train a memorability MLP head for one CLIP-HBA-MEG timepoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--timepoint_ms',
        type=int,
        default=int(os.environ.get('TIMEPOINT_MS', 0)),
        help='Target timepoint in ms (e.g. 100).  Must match an available '
             'embedding file: clip_hba_meg_mem_tp{timepoint_ms}_fold*_*.pt.',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=int(os.environ.get('FOLD', 1)),
        help='Cross-validation fold (1–5 for lamem; 1–10 for combined_lamem_memcat).',
    )
    parser.add_argument(
        '--training_data',
        default=os.environ.get('TRAINING_DATA', 'combined_lamem_memcat'),
        choices=['lamem', 'combined_lamem_memcat'],
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=int(os.environ.get('CUDA_DEVICE', 0)),
        help='GPU index (0, 1, …).  Use -1 for DataParallel, 2 for CPU.',
    )
    # --- MLP head architecture (winning config from sweep) ---
    parser.add_argument(
        '--hidden_dims', nargs='+', type=int, default=None,
        help='MLP hidden layer sizes (e.g. --hidden_dims 128 32).  '
             'Defaults to (32, 16) — a reasonable compression-only baseline.',
    )
    parser.add_argument('--dropout_rate', type=float,
                        default=float(os.environ.get('DROPOUT_RATE', 0.3)))
    parser.add_argument('--lr', type=float,
                        default=float(os.environ.get('LR', 1e-4)))
    parser.add_argument('--weight_decay', type=float,
                        default=float(os.environ.get('WEIGHT_DECAY', 1e-3)))
    parser.add_argument('--batch_size', type=int,
                        default=int(os.environ.get('BATCH_SIZE', 64)))
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--early_stopping_patience', type=int, default=20)
    parser.add_argument('--seed', type=int,
                        default=int(os.environ.get('RANDOM_SEED', 1)))
    # --- Paths ---
    parser.add_argument(
        '--embeddings_dir', default=None,
        help='Directory containing precomputed MEG .pt embedding files.  '
             'Defaults to ./Data/{training_data}/meg_embeddings/.',
    )
    parser.add_argument(
        '--checkpoint_path', default=None,
        help='Checkpoint save path prefix.  '
             'Defaults to ./models/clip_hba_meg_mem/tp{timepoint_ms}.',
    )
    parser.add_argument('--log_path', default=None,
                        help='Log file path.  Defaults to ./logs/meg_mem_tp{tp}_fold{fold}.log.')
    parser.add_argument('--preds_dir', default='./preds/meg_mem/',
                        help='Directory for per-epoch prediction CSVs.')
    args = parser.parse_args()

    tp = args.timepoint_ms
    if tp not in _ALL_TIMEPOINTS:
        raise ValueError(
            f'--timepoint_ms {tp} is not in the standard set {_ALL_TIMEPOINTS}. '
            f'Ensure matching embedding files exist.'
        )

    fold = args.fold
    training_data = args.training_data

    # --- Resolve paths ---
    if training_data == 'lamem':
        train_csv = f'./Data/lamem/lamem_train_{fold}.csv'
        val_csv   = f'./Data/lamem/lamem_val_{fold}.csv'
        test_csv  = f'./Data/lamem/lamem_test_{fold}.csv'
        img_root  = os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/')
        memcat_meta_csv = None
        default_emb_dir = './Data/lamem/meg_embeddings/'
    else:  # combined_lamem_memcat
        train_csv = f'./Data/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv'
        val_csv   = f'./Data/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv'
        test_csv  = f'./Data/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv'
        img_root  = {
            'lamem':  os.environ.get('LAMEM_IMG_ROOT',  './Data/lamem/images/'),
            'memcat': os.environ.get('MEMCAT_IMG_ROOT', './Data/memcat/images/'),
        }
        memcat_meta_csv = os.environ.get(
            'MEMCAT_META_CSV', './Data/memcat/memcat_image_data.csv'
        )
        default_emb_dir = './Data/combined_lamem_memcat/meg_embeddings/'

    embeddings_dir   = args.embeddings_dir or default_emb_dir
    checkpoint_path  = args.checkpoint_path or f'./models/clip_hba_meg_mem/tp{tp}'
    log_path         = args.log_path or f'./logs/meg_mem_tp{tp}_fold{fold}.log'
    hidden_dims      = tuple(args.hidden_dims) if args.hidden_dims else (32, 16)

    # model_type encodes the timepoint so _run_mem_training_impl resolves the
    # correct embedding file: {model_type}_fold{fold}_{split}.pt
    # → clip_hba_meg_mem_tp{tp}_fold{fold}_{split}.pt
    model_type = f'clip_hba_meg_mem_tp{tp}'

    config = {
        'model_type':    model_type,
        'input_dim':     66,
        'training_data': training_data,

        # Data
        'fold':      fold,
        'train_csv': train_csv,
        'val_csv':   val_csv,
        'test_csv':  test_csv,
        'img_root':        img_root,
        'memcat_meta_csv': memcat_meta_csv,
        'preds_dir': args.preds_dir,

        # Precomputed MEG embeddings (required for this model type)
        'embeddings_dir': embeddings_dir,

        # MLP head
        'hidden_dims':  hidden_dims,
        'dropout_rate': args.dropout_rate,

        # Optimisation
        'epochs':                   args.epochs,
        'batch_size':               args.batch_size,
        'lr':                       args.lr,
        'weight_decay':             args.weight_decay,
        'early_stopping_patience':  args.early_stopping_patience,

        # Device & I/O
        'cuda':            args.cuda,
        'checkpoint_path': checkpoint_path,
        'log_path':        log_path,
        'random_seed':     args.seed,
        'criterion':       nn.MSELoss(),
    }

    print(f'\n=== CLIP-HBA-MEG Memorability Training ===')
    print(f'  Timepoint:     {tp:+d} ms')
    print(f'  Fold:          {fold}')
    print(f'  Model type:    {model_type}')
    print(f'  Hidden dims:   {hidden_dims}')
    print(f'  Input dim:     66')
    print(f'  Embeddings:    {embeddings_dir}')
    print(f'  Checkpoint:    {checkpoint_path}')
    print()

    run_mem_training(config)


if __name__ == '__main__':
    main()
