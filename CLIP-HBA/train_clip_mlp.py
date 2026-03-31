from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn
import os


def main():
    data_dir    = os.environ.get('DATA_DIR', './Data')
    fold        = int(os.environ.get('LAMEM_FOLD', 5))
    model_type  = os.environ.get('MODEL_TYPE', 'clip_frozen_mlp')
    img_root    = os.environ.get('LAMEM_IMG_ROOT', f'{data_dir}/lamem/images/')
    cuda_device = int(os.environ.get('CUDA_DEVICE', 1))

    config = {
        'model_type': model_type,

        # --- Data ---
        'fold':      fold,
        'train_csv': f'{data_dir}/lamem/lamem_train_{fold}.csv',
        'val_csv':   f'{data_dir}/lamem/lamem_val_{fold}.csv',
        'test_csv':  f'{data_dir}/lamem/lamem_test_{fold}.csv',
        'img_root':  img_root,
        'preds_dir': './preds/',
        'log_path':  './logs/clip_frozen_mlp.log',

        # --- Precomputed embeddings (optional, ~100x epoch speedup) ---
        # Run extract_embeddings.slurm once to populate this directory, then
        # uncomment the line below to skip backbone inference during training.
        'embeddings_dir': f'{data_dir}/lamem/embeddings/',

        # --- Device ---
        'cuda': cuda_device,

        # --- MLP head ---
        'hidden_dims':  (512, 256),
        'dropout_rate': 0.625732,

        # --- Training ---
        'epochs':                  300,
        'batch_size':              256,
        'lr':                      8.1e-5,
        'weight_decay':            0.0205,
        'early_stopping_patience': 20,
        'train_fraction':          float(os.environ.get('TRAIN_FRACTION', 1.0)),
        'checkpoint_path':         './models/clip_frozen_mlp',
        'random_seed':             1,
        'criterion':               nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
