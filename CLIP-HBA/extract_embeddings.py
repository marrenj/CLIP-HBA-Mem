"""Pre-extract frozen backbone embeddings for fast MLP head training.

For clip_hba_mem and clip_frozen_mlp, the backbone is fully frozen and its
output embeddings are identical every epoch.  Running 48 k images through
ViT-L/14 on every training epoch (1 500 forward passes at batch_size=32) is
the primary throughput bottleneck.  This script runs all images through the
backbone *once*, saves embeddings alongside memorability scores and image
paths, and then training uses EmbeddingDataset or MEGEmbeddingDataset (defined
in functions/train_mem_pipeline.py) to serve in-memory tensors.

For clip_hba_meg_mem the backbone also produces a **timepoint-specific** 768-dim
representation: a learned weighted sum of all 24 ViT layer CLS-token
projections, where per-layer weights come from the MEG model's trained
``weighting_matrix[t, :]``.  Because this differs across the T MEG timepoints,
the output is ``Tensor[N, T, 768]`` (float16) rather than ``Tensor[N, 768]``.

Expected speedup: ~100x per epoch (seconds rather than ~7.5 minutes).

Usage
-----
    # LaMem — single fold, clip_frozen_mlp:
    python extract_embeddings.py --fold 1

    # LaMem — single fold, clip_hba_mem:
    python extract_embeddings.py --model_type clip_hba_mem --fold 1

    # Combined LaMem + MemCat — single fold:
    python extract_embeddings.py \\
        --training_data combined_lamem_memcat \\
        --fold 1 \\
        --memcat_img_root ./Data/memcat/images/ \\
        --memcat_meta_csv ./Data/memcat/memcat_image_data.csv \\
        --out_dir ./Data/combined_lamem_memcat/embeddings/

    # All folds for all model types (via the accompanying SLURM script):
    sbatch extract_embeddings.slurm

Outputs
-------
    <out_dir>/<model_type>_fold<N>_train.pt
    <out_dir>/<model_type>_fold<N>_val.pt
    <out_dir>/<model_type>_fold<N>_test.pt

    Each .pt file is a dict:
        embeddings  - Tensor[N, 768]   float32 CLS token embeddings
        scores      - Tensor[N]        float32 memorability scores
        image_paths - list[N]          relative image path strings (from CSV)

Enabling fast training
----------------------
    Add ``'embeddings_dir': './Data/lamem/embeddings/'`` (or the combined
    equivalent) to the config dict in train_mem.py or train_mem_sweep.py.
    When this key is present, run_mem_training() uses EmbeddingDataset +
    MLPOnlyHead instead of loading raw images and running the full backbone
    on every batch.
"""

import argparse
import os
import pathlib
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.train_mem_pipeline import (
    MemDataset,
    PerceptCLIPDataset,
    CLIPHBAMem,
    CLIPFrozenMLP,
)
from functions.train_behavior_things_pipeline import seed_everything

# Allow imports from the project root (needed for clip_hba_meg_mem).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _build_backbone_and_extractor(
    model_type: str,
    backbone_checkpoint: str,
    backbone_name: str,
    vision_layers: int,
    transformer_layers: int,
    rank: int,
):
    """Instantiate the frozen backbone and return (model, DatasetClass, embed_fn).

    embed_fn(model, images) -> Tensor[B, 768] extracts the CLS-token
    embedding without running through the MLP head.
    """
    if model_type == 'clip_hba_mem':
        model = CLIPHBAMem(
            backbone_checkpoint=backbone_checkpoint,
            backbone_name=backbone_name,
            vision_layers=vision_layers,
            transformer_layers=transformer_layers,
            rank=rank,
            hidden_dims=(256, 128),  # dummy head — never called during extraction
            dropout_rate=0.0,
        )
        DatasetClass = MemDataset

        def embed_fn(m: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
            raw = m.module if isinstance(m, torch.nn.DataParallel) else m
            return raw.backbone.clip_model.encode_image(
                images, raw.backbone.pos_embedding)   # [B, 768]

    elif model_type == 'clip_frozen_mlp':
        model = CLIPFrozenMLP(
            hidden_dims=(256, 128),  # dummy head — never called during extraction
            dropout_rate=0.0,
        )
        DatasetClass = PerceptCLIPDataset

        def embed_fn(m: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
            raw = m.module if isinstance(m, torch.nn.DataParallel) else m
            vis_out = raw.vision_model(images)
            return raw.visual_projection(vis_out[1])   # [B, 768]

    else:
        raise ValueError(
            f'Unknown model_type: {model_type!r}. '
            f'Choose "clip_hba_mem", "clip_frozen_mlp", or "clip_hba_meg_mem".'
        )

    return model, DatasetClass, embed_fn


def _normalize_weights_minmax(weights: torch.Tensor) -> torch.Tensor:
    """Min-max normalise a weight vector so values lie in [0,1] and sum to 1.

    Replicates ``normalize_weights(weights, method='minmax')`` from
    ``src/models/CLIPs/clip_hba_meg/model.py`` without importing that module
    here (avoids a heavyweight import at the top of this script).
    """
    min_val = weights.min()
    max_val = weights.max()
    w = (weights - min_val) / (max_val - min_val + 1e-10)
    return w / (w.sum() + 1e-10)


def _build_meg_model(
    backbone_checkpoint: str,
    backbone_name: str,
    ms_start: int,
    ms_step: int,
    ms_end: int,
    train_start: int,
    train_step: int,
    train_end: int,
    train_window_size: int,
) -> torch.nn.Module:
    """Load and return a frozen CLIP-HBA-MEG model from checkpoint.

    Args:
        backbone_checkpoint: Path to the MEG model .pth checkpoint.
        backbone_name: CLIP backbone variant (e.g. ``'ViT-L/14'``).
        ms_start: First timepoint of the full MEG recording (ms).
        ms_step: Timepoint step (ms).
        ms_end: Last timepoint of the full MEG recording (ms).
        train_start: First *training* timepoint (ms).
        train_step: Step between training timepoints (ms).
        train_end: Last *training* timepoint (ms).
        train_window_size: Half-window used during MEG training (ms).

    Returns:
        ``CLIPHBA`` instance with all parameters frozen, checkpoint loaded.
    """
    from functions.inference_meg_group_pipeline import CLIPHBA as CLIPHBAMeg
    from functions.spose_dimensions import classnames66

    model = CLIPHBAMeg(
        classnames=classnames66,
        backbone_name=backbone_name,
        pos_embedding=True,
        ms_start=ms_start,
        ms_step=ms_step,
        ms_end=ms_end,
        train_start=train_start,
        train_step=train_step,
        train_end=train_end,
        train_window_size=train_window_size,
    )

    state_dict = torch.load(backbone_checkpoint, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f'[MEG Backbone] Loaded {len(state_dict)} keys from {backbone_checkpoint}')

    for p in model.parameters():
        p.requires_grad = False

    return model


def _compute_meg_features_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    ms_start: int,
    ms_step: int,
    ms_end: int,
    train_start: int,
    train_step: int,
    train_end: int,
    train_window_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute per-timepoint visual features for a batch of images.

    Runs a single ViT forward pass (via ``encode_image``) to collect all 24
    layer outputs, then applies the MEG model's learned ``weighting_matrix``
    to produce a 768-dim feature vector for each of the T training timepoints.

    This exactly replicates the image-feature computation in
    ``ModifiedCLIP.forward()`` (``src/models/CLIPs/clip_hba_meg/model.py``),
    minus the text encoder and noise injection.

    Args:
        model: Frozen ``CLIPHBA`` MEG model.
        images: Tensor[B, 3, 224, 224] on ``device``.
        ms_start, ms_step, ms_end: Full MEG recording range.
        train_start, train_step, train_end: Training timepoint range.
        train_window_size: Half-window (ms) used during MEG training.
        device: Compute device.

    Returns:
        Tensor[B, T, 768] float16 — one 768-dim feature per training timepoint.
    """
    clip_model = model.clip_model
    pos_embedding = model.pos_embedding

    with torch.no_grad():
        _, layer_outputs = clip_model.encode_image(images, pos_embedding)
        # layer_outputs: list of 24 tensors, each [B, n_patches+1, width]

    n_train = (train_end - train_start) // train_step + 1
    n_total = (ms_end - ms_start) // ms_step + 1
    B = images.size(0)

    features_list = []
    for i in range(n_train):
        t_ms = train_start + i * train_step
        time_index = float(t_ms - ms_start) / ms_step
        start_idx = max(0, int(time_index - train_window_size // ms_step))
        end_idx   = min(int(time_index + train_window_size // ms_step + 1), n_total)

        weights = clip_model.weighting_matrix[start_idx:end_idx].mean(dim=0)
        weights = _normalize_weights_minmax(weights).to(device)         # [24]
        visual_scaler = clip_model.visual_scaler[start_idx:end_idx].mean()  # scalar

        image_features = torch.zeros(B, 768, device=device, dtype=torch.float32)
        for j, layer_out in enumerate(layer_outputs):
            # layer_out: [B, patches+1, width]; [:, 0, :] is the CLS token
            cls_proj = layer_out[:, 0, :].float() @ clip_model.visual.proj.float()
            image_features += cls_proj * (weights[j] * visual_scaler.float())

        features_list.append(image_features.half().cpu())  # save RAM as float16

    return torch.stack(features_list, dim=1)  # [B, T, 768] float16


def extract_embeddings_for_fold(
    model_type: str,
    fold: int,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    img_root: 'str | dict',
    backbone_checkpoint: str,
    out_dir: str,
    device: torch.device,
    batch_size: int = 256,
    num_workers: int = 8,
    backbone_name: str = 'ViT-L/14',
    vision_layers: int = 2,
    transformer_layers: int = 1,
    rank: int = 32,
    memcat_meta_csv: 'str | None' = None,
    # MEG-specific parameters (only used when model_type == 'clip_hba_meg_mem')
    meg_ms_start: int = -100,
    meg_ms_step: int = 5,
    meg_ms_end: int = 1300,
    meg_train_start: int = -100,
    meg_train_step: int = 100,
    meg_train_end: int = 1300,
    meg_train_window_size: int = 300,
) -> None:
    """Extract and save embeddings for all three splits of one fold.

    Skips any split whose output file already exists so the script is safe
    to re-run after an interruption.

    For ``clip_hba_mem`` / ``clip_frozen_mlp``, saves::

        {
            'embeddings':  Tensor[N, 768],   # float32
            'scores':      Tensor[N],
            'image_paths': list[N],
        }

    For ``clip_hba_meg_mem``, saves::

        {
            'embeddings':    Tensor[N, T, 768],  # float16 to keep file size manageable
            'scores':        Tensor[N],
            'image_paths':   list[N],
            'timepoints_ms': Tensor[T],          # MEG timepoints in milliseconds
        }

    Args:
        model_type: ``'clip_hba_mem'``, ``'clip_frozen_mlp'``, or ``'clip_hba_meg_mem'``.
        fold: Fold number used in output filenames.
        train_csv: Path to the training split CSV.
        val_csv: Path to the validation split CSV.
        test_csv: Path to the test split CSV.
        img_root: Root directory for images.  Pass a ``dict`` for combined splits.
        backbone_checkpoint: Path to the checkpoint (.pth).
        out_dir: Directory where ``.pt`` embedding files are written.
        device: Torch device to run backbone inference on.
        batch_size: Batch size for backbone inference.
        num_workers: DataLoader worker count.
        backbone_name: CLIP backbone variant string.
        vision_layers: DoRA-patched vision layers (``clip_hba_mem`` only).
        transformer_layers: DoRA-patched text layers (``clip_hba_mem`` only).
        rank: DoRA rank (``clip_hba_mem`` only).
        memcat_meta_csv: Path to ``memcat_image_data.csv`` for combined splits.
        meg_ms_start: First timepoint of the full MEG recording (ms).
        meg_ms_step: Timepoint step (ms).
        meg_ms_end: Last timepoint of the full MEG recording (ms).
        meg_train_start: First training timepoint (ms).
        meg_train_step: Step between training timepoints (ms).
        meg_train_end: Last training timepoint (ms).
        meg_train_window_size: Half-window used during MEG model training (ms).
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {'train': train_csv, 'val': val_csv, 'test': test_csv}

    # ------------------------------------------------------------------
    # MEG path: produces [N, T, 768] float16 output per split
    # ------------------------------------------------------------------
    if model_type == 'clip_hba_meg_mem':
        timepoints_ms = torch.arange(
            meg_train_start, meg_train_end + 1, meg_train_step, dtype=torch.int32
        )
        T = len(timepoints_ms)
        print(f'[MEG] T={T} training timepoints '
              f'({meg_train_start}ms to {meg_train_end}ms, step={meg_train_step}ms)')

        model = _build_meg_model(
            backbone_checkpoint=backbone_checkpoint,
            backbone_name=backbone_name,
            ms_start=meg_ms_start,
            ms_step=meg_ms_step,
            ms_end=meg_ms_end,
            train_start=meg_train_start,
            train_step=meg_train_step,
            train_end=meg_train_end,
            train_window_size=meg_train_window_size,
        )
        model.eval()
        model.to(device)

        for split_name, csv_path in splits.items():
            out_path = out_dir / f'{model_type}_fold{fold}_{split_name}.pt'
            if out_path.exists():
                print(f'[Skip] {out_path} already exists — delete it to re-extract.')
                continue

            dataset = MemDataset(
                csv_file=csv_path,
                img_root=img_root,
                memcat_meta_csv=memcat_meta_csv,
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            all_embeddings:  list = []
            all_scores:      list = []
            all_image_paths: list = []

            print(f'\n[{model_type} | fold {fold} | {split_name}]  '
                  f'{len(dataset)} images  ->  {out_path}')
            print(f'  Output shape per image: [T={T}, 768] float16')

            with tqdm(loader, desc=f'  {split_name}') as pbar:
                for image_paths, images, scores in pbar:
                    images = images.to(device)
                    # [B, T, 768] float16 (computed on GPU, moved to CPU)
                    emb = _compute_meg_features_batch(
                        model=model,
                        images=images,
                        ms_start=meg_ms_start,
                        ms_step=meg_ms_step,
                        ms_end=meg_ms_end,
                        train_start=meg_train_start,
                        train_step=meg_train_step,
                        train_end=meg_train_end,
                        train_window_size=meg_train_window_size,
                        device=device,
                    )
                    all_embeddings.append(emb)
                    all_scores.append(scores.cpu())
                    all_image_paths.extend(image_paths)

            emb_tensor = torch.cat(all_embeddings, dim=0)   # [N, T, 768] float16
            payload = {
                'embeddings':    emb_tensor,
                'scores':        torch.cat(all_scores, dim=0),
                'image_paths':   all_image_paths,
                'timepoints_ms': timepoints_ms,
            }
            torch.save(payload, out_path)
            size_gb = emb_tensor.numel() * 2 / 1e9  # float16 = 2 bytes
            print(f'  Saved {emb_tensor.shape[0]:,} images × {T} timepoints × 768 dims'
                  f'  ({size_gb:.1f} GB)  ->  {out_path}')

        model.cpu()
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return

    # ------------------------------------------------------------------
    # Standard path (clip_hba_mem / clip_frozen_mlp): [N, 768] float32
    # ------------------------------------------------------------------
    model, DatasetClass, embed_fn = _build_backbone_and_extractor(
        model_type=model_type,
        backbone_checkpoint=backbone_checkpoint,
        backbone_name=backbone_name,
        vision_layers=vision_layers,
        transformer_layers=transformer_layers,
        rank=rank,
    )
    model.eval()
    model.to(device)

    for split_name, csv_path in splits.items():
        out_path = out_dir / f'{model_type}_fold{fold}_{split_name}.pt'

        if out_path.exists():
            print(f'[Skip] {out_path} already exists — delete it to re-extract.')
            continue

        dataset = DatasetClass(
            csv_file=csv_path,
            img_root=img_root,
            memcat_meta_csv=memcat_meta_csv,
        )
        loader  = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        all_embeddings:  list = []
        all_scores:      list = []
        all_image_paths: list = []

        print(f'\n[{model_type} | fold {fold} | {split_name}]  '
              f'{len(dataset)} images  ->  {out_path}')

        with torch.no_grad(), tqdm(loader, desc=f'  {split_name}') as pbar:
            for image_paths, images, scores in pbar:
                images = images.to(device)
                emb    = embed_fn(model, images)   # [B, 768]
                all_embeddings.append(emb.cpu())
                all_scores.append(scores.cpu())
                all_image_paths.extend(image_paths)

        payload = {
            'embeddings':  torch.cat(all_embeddings, dim=0),   # [N, 768]
            'scores':      torch.cat(all_scores,      dim=0),   # [N]
            'image_paths': all_image_paths,                      # list[N]
        }
        torch.save(payload, out_path)
        print(f'  Saved {payload["embeddings"].shape[0]:,} embeddings  '
              f'({payload["embeddings"].shape}) -> {out_path}')

    # Free GPU memory before the caller potentially builds another backbone.
    model.cpu()
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pre-extract frozen backbone embeddings for memorability MLP training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--training_data',
        default=os.environ.get('TRAINING_DATA', 'lamem'),
        choices=['lamem', 'combined_lamem_memcat'],
        help='Dataset to extract embeddings for.',
    )
    parser.add_argument(
        '--model_type',
        default=os.environ.get('MODEL_TYPE', 'clip_frozen_mlp'),
        choices=['clip_hba_mem', 'clip_frozen_mlp', 'clip_hba_meg_mem'],
        help='Which frozen backbone to extract from.',
    )
    parser.add_argument(
        '--fold',
        type=int,
        default=int(os.environ.get('FOLD', os.environ.get('LAMEM_FOLD', 1))),
        help='Fold number (1–5 for lamem; 1–10 for combined_lamem_memcat).',
    )
    parser.add_argument(
        '--img_root',
        default=os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/'),
        help='LaMem images root directory.',
    )
    parser.add_argument(
        '--memcat_img_root',
        default=os.environ.get('MEMCAT_IMG_ROOT', './Data/memcat/images/'),
        help='MemCat images root directory (only used for combined_lamem_memcat).',
    )
    parser.add_argument(
        '--memcat_meta_csv',
        default=os.environ.get('MEMCAT_META_CSV', './Data/memcat/memcat_image_data.csv'),
        help='Path to memcat_image_data.csv with category/subcategory columns '
             '(only used for combined_lamem_memcat).',
    )
    parser.add_argument(
        '--out_dir',
        default=None,
        help='Directory where .pt embedding files are written.  '
             'Defaults to ./Data/lamem/embeddings/ for lamem and '
             './Data/combined_lamem_memcat/embeddings/ for combined_lamem_memcat.',
    )
    parser.add_argument(
        '--backbone_checkpoint',
        default=os.environ.get('BACKBONE_CHECKPOINT', './Data/lamem/epoch97_dora_params.pth'),
        help='CLIP-HBA / CLIP-HBA-MEG checkpoint path.',
    )
    # MEG-specific args (ignored for clip_hba_mem / clip_frozen_mlp)
    parser.add_argument('--meg_ms_start',          type=int, default=-100,
                        help='[MEG] First timepoint in the full recording (ms).')
    parser.add_argument('--meg_ms_step',           type=int, default=5,
                        help='[MEG] Recording timepoint step (ms).')
    parser.add_argument('--meg_ms_end',            type=int, default=1300,
                        help='[MEG] Last timepoint in the full recording (ms).')
    parser.add_argument('--meg_train_start',       type=int, default=-100,
                        help='[MEG] First training timepoint (ms).')
    parser.add_argument('--meg_train_step',        type=int, default=100,
                        help='[MEG] Step between training timepoints (ms).')
    parser.add_argument('--meg_train_end',         type=int, default=1300,
                        help='[MEG] Last training timepoint (ms).')
    parser.add_argument('--meg_train_window_size', type=int, default=300,
                        help='[MEG] Half-window size used during MEG training (ms).')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for backbone inference during extraction.',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='DataLoader worker count for image loading.',
    )
    parser.add_argument(
        '--cuda',
        type=int,
        default=int(os.environ.get('CUDA_DEVICE', 0)),
        help='GPU index (0, 1, …).  Use 2 for CPU.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed (affects reproducibility of any stochastic ops).',
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device('cpu') if args.cuda == 2 else torch.device(f'cuda:{args.cuda}')

    fold = args.fold

    if args.training_data == 'lamem':
        train_csv = f'./Data/lamem/lamem_train_{fold}.csv'
        val_csv   = f'./Data/lamem/lamem_val_{fold}.csv'
        test_csv  = f'./Data/lamem/lamem_test_{fold}.csv'
        img_root      = args.img_root
        memcat_meta_csv = None
        out_dir = args.out_dir or './Data/lamem/embeddings/'
    else:  # combined_lamem_memcat
        train_csv = f'./Data/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv'
        val_csv   = f'./Data/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv'
        test_csv  = f'./Data/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv'
        img_root = {
            'lamem':  args.img_root,
            'memcat': args.memcat_img_root,
        }
        memcat_meta_csv = args.memcat_meta_csv
        out_dir = args.out_dir or './Data/combined_lamem_memcat/embeddings/'

    extract_embeddings_for_fold(
        model_type=args.model_type,
        fold=fold,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        img_root=img_root,
        backbone_checkpoint=args.backbone_checkpoint,
        out_dir=out_dir,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        memcat_meta_csv=memcat_meta_csv,
        meg_ms_start=args.meg_ms_start,
        meg_ms_step=args.meg_ms_step,
        meg_ms_end=args.meg_ms_end,
        meg_train_start=args.meg_train_start,
        meg_train_step=args.meg_train_step,
        meg_train_end=args.meg_train_end,
        meg_train_window_size=args.meg_train_window_size,
    )


if __name__ == '__main__':
    main()
