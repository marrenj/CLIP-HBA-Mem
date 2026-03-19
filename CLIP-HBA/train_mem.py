from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn
import os


def main():
    training_data = os.environ.get('TRAINING_DATA', 'lamem')  # 'lamem' | 'combined_lamem_memcat'
    fold = int(os.environ.get('FOLD', 1))

    if training_data == 'lamem':
        train_csv = f'./Data/lamem/lamem_train_{fold}.csv'
        val_csv   = f'./Data/lamem/lamem_val_{fold}.csv'
        test_csv  = f'./Data/lamem/lamem_test_{fold}.csv'
        img_root  = './Data/lamem/images/'
    elif training_data == 'combined_lamem_memcat':
        train_csv = f'./Data/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv'
        val_csv   = f'./Data/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv'
        test_csv  = f'./Data/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv'
        img_root  = {'lamem':  './Data/lamem/images/',
                     'memcat': './Data/memcat/images/'}
    else:
        raise ValueError(f"Unknown TRAINING_DATA: {training_data!r}. "
                         f"Choose 'lamem' or 'combined_lamem_memcat'.")

    config = {
        'model_type':    'clip_hba_mem',  # 'clip_hba_mem' | 'perceptclip'
        'training_data': training_data,

        # --- Data ---
        'fold':      fold,
        'train_csv': train_csv,
        'val_csv':   val_csv,
        'test_csv':  test_csv,
        'img_root':  img_root,
        'preds_dir': './preds/',
        'log_path':  './logs/mem.log',

        # --- Precomputed embeddings (optional, ~100x epoch speedup) ---
        # Run extract_embeddings.slurm once to populate this directory, then
        # uncomment the line below to skip backbone inference during training.
        # 'embeddings_dir': './Data/lamem/embeddings/',

        # --- Backbone (frozen CLIP-HBA) ---
        'backbone_checkpoint': './Data/lamem/epoch97_dora_params.pth',
        'backbone':            'ViT-L/14',
        'vision_layers':       2,   # must match the checkpoint's DoRA config
        'transformer_layers':  1,
        'rank':                32,

        # --- Device ---
        'cuda': int(os.environ.get('CUDA_DEVICE', 0)),   # 0=cuda:0, 1=cuda:1, -1=all GPUs (DataParallel), 2=cpu

        # --- MLP head ---
        'hidden_dims':   (512, 256),
        'dropout_rate':  0.585585,

        # --- Training ---
        'epochs':                   300,
        'batch_size':               128,
        'lr':                       3.8e-5,
        'weight_decay':             5.36e-4,
        'early_stopping_patience':  20,
        'checkpoint_path':          './models/clip_hba_mem',
        'random_seed':              1,
        'criterion':                nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
