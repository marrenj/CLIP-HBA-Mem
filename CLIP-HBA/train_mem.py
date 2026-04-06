from functions.train_mem_pipeline import run_mem_training
import torch.nn as nn
import os


def main():
    data_dir      = os.environ.get('DATA_DIR', './Data')
    training_data = os.environ.get('TRAINING_DATA', 'combined_lamem_memcat')  # 'lamem' | 'combined_lamem_memcat'
    fold = int(os.environ.get('FOLD', 1))

    if training_data == 'lamem':
        train_csv = f'{data_dir}/lamem/lamem_train_{fold}.csv'
        val_csv   = f'{data_dir}/lamem/lamem_val_{fold}.csv'
        test_csv  = f'{data_dir}/lamem/lamem_test_{fold}.csv'
        img_root  = os.environ.get('LAMEM_IMG_ROOT', f'{data_dir}/lamem/images/')
        memcat_meta_csv = None
    elif training_data == 'combined_lamem_memcat':
        train_csv = f'{data_dir}/combined_lamem_memcat/lamem_memcat_train_split_{fold:02d}.csv'
        val_csv   = f'{data_dir}/combined_lamem_memcat/lamem_memcat_val_split_{fold:02d}.csv'
        test_csv  = f'{data_dir}/combined_lamem_memcat/lamem_memcat_test_split_{fold:02d}.csv'
        img_root  = {'lamem':  os.environ.get('LAMEM_IMG_ROOT',  f'{data_dir}/lamem/images/'),
                     'memcat': os.environ.get('MEMCAT_IMG_ROOT', f'{data_dir}/memcat/images/')}
        memcat_meta_csv = os.environ.get('MEMCAT_META_CSV', f'{data_dir}/memcat/memcat_image_data.csv')
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
        'img_root':        img_root,
        'memcat_meta_csv': memcat_meta_csv,
        'preds_dir': './preds/',
        'log_path':  './logs/mem.log',

        # --- Precomputed embeddings (optional, ~100x epoch speedup) ---
        # Run extract_embeddings.slurm once to populate this directory, then
        # uncomment the line below to skip backbone inference during training.
        # 'embeddings_dir': f'{data_dir}/lamem/embeddings/',

        # --- Backbone (frozen CLIP-HBA) ---
        'backbone_checkpoint': f'{data_dir}/lamem/epoch97_dora_params.pth',
        'backbone':            'ViT-L/14',
        'vision_layers':       2,   # must match the checkpoint's DoRA config
        'transformer_layers':  1,
        'rank':                32,

        # --- Device ---
        'cuda': int(os.environ.get('CUDA_DEVICE', 0)),   # 0=cuda:0, 1=cuda:1, -1=all GPUs (DataParallel), 2=cpu

        # --- MLP head ---
        'hidden_dims':   (256, 128),
        'dropout_rate':  0.4280261676059678,

        # --- Training ---
        'epochs':                   300,
        'batch_size':               64,
        'lr':                       2.060924941320234e-05,
        'weight_decay':             0.07556810141274425,
        'early_stopping_patience':  20,
        'checkpoint_path':          './models/clip_hba_mem',
        'random_seed':              1,
        'criterion':                nn.MSELoss(),
    }

    run_mem_training(config)


if __name__ == '__main__':
    main()
