from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn
import os


def main():
    fold = int(os.environ.get('LAMEM_FOLD', 1))
    model_type = os.environ.get('MODEL_TYPE', 'clip_frozen_mlp')
    img_root = os.environ.get('LAMEM_IMG_ROOT', './Data/lamem/images/')
    cuda_device = int(os.environ.get('CUDA_DEVICE', 1))

    config = {
        'model_type': model_type,

        # --- Data ---
        'fold':      fold,
        'train_csv': f'./Data/lamem/lamem_train_{fold}.csv',
        'val_csv':   f'./Data/lamem/lamem_val_{fold}.csv',
        'test_csv':  f'./Data/lamem/lamem_test_{fold}.csv',
        'img_root':  img_root,
        'preds_dir': './preds/',
        'log_path':  './logs/clip_frozen_mlp.log',

        # --- Precomputed embeddings (optional, ~100x epoch speedup) ---
        # Run extract_embeddings.slurm once to populate this directory, then
        # uncomment the line below to skip backbone inference during training.
        # 'embeddings_dir': './Data/lamem/embeddings/',

        # --- Device ---
        'cuda': cuda_device,

        # --- MLP head ---
        'hidden_dims':  (512, 256),
        'dropout_rate': 0.5,

        # --- Training ---
        'epochs':                  300,
        'batch_size':              32,
        'lr':                      5e-5,
        'early_stopping_patience': 10,
        'checkpoint_path':         './models/clip_frozen_mlp',
        'random_seed':             1,
        'criterion':               nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
