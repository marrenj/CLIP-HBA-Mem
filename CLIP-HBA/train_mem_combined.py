"""Training entrypoint for the combined LaMem + MemCat memorability dataset.

The combined split CSVs live in Data/combined_lamem_memcat/ and have columns:
    set   - 'lamem' or 'memcat'
    file  - image filename (relative to the respective image root)
    mem   - memorability score in [0, 1]

Images are loaded from Data/lamem/images/ or Data/memcat/images/ depending on
the 'set' column.  Everything else (backbone, MLP head, training loop) is
identical to train_mem.py.

Environment variables
---------------------
    FOLD        : fold number 1-10  (default 1)
    CUDA_DEVICE : GPU index         (default 0; use 2 for CPU)
"""

from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn
import os


def main():
    fold = int(os.environ.get('FOLD', 1))
    config = {
        'model_type':   'clip_hba_mem',
        'dataset_type': 'combined',   # triggers CombinedMemDataset

        # --- Data ---
        'fold':     fold,
        'train_csv': f'./Data/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv',
        'val_csv':   f'./Data/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv',
        'test_csv':  f'./Data/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv',
        'lamem_img_root':  './Data/lamem/images/',
        'memcat_img_root': './Data/memcat/images/',
        'preds_dir': './preds/',
        'log_path':  './logs/mem_combined.log',

        # --- Backbone (frozen CLIP-HBA) ---
        'backbone_checkpoint': './Data/lamem/epoch97_dora_params.pth',
        'backbone':            'ViT-L/14',
        'vision_layers':       2,   # must match the checkpoint's DoRA config
        'transformer_layers':  1,
        'rank':                32,

        # --- Device ---
        'cuda': int(os.environ.get('CUDA_DEVICE', 0)),   # 0=cuda:0, -1=all GPUs, 2=cpu

        # --- MLP head ---
        'hidden_dims':   (512, 256),
        'dropout_rate':  0.585585,

        # --- Training ---
        'epochs':                   300,
        'batch_size':               128,
        'lr':                       3.8e-5,
        'weight_decay':             5.36e-4,
        'early_stopping_patience':  20,
        'checkpoint_path':          './models/clip_hba_mem_combined',
        'random_seed':              1,
        'criterion':                nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
