from functions.train_meg_things_pipeline import run_meg_training_group

def main():
    # Define configuration
    config = {
        # Data paths
        'csv_file': "./Data/spose_embedding66d_rescaled_1806train.csv", # csv data annotations of the training stimuli, embedding is not required for MEG training, but need image name and indices
        'img_dir': "./Data/Things1854", # path to the image directory
        'rdm_dir': "./Data/ThingsMEG_RDMs/ThingsMEG_RDM_4P.npy", # path to the RDM file
        'n_dim': 66,
        
        # Model parameters
        'backbone': 'ViT-L/14',
        'epochs': 50,
        'fw_tuning_epochs': None,
        'batch_size': 50,
        'train_portion': 1456/1806,
        
        # Time parameters
        'ms_start': -100,
        'ms_end': 1300,
        'ms_step': 5,
        'train_start': -100,
        'train_end': 1300,
        'train_step': 5,
        'train_window_size': 15,
        
        # Learning parameters
        'lr_1': 0,
        'lr_2': 3e-5,
        'fw_lr_1': 3e-3,
        'fw_lr_2': 3e-3,
        'early_stopping_patience': 5,
        
        # Model paths and settings
        'checkpoint_path': './models/cliphba_dynamic_official_v10.pth',
        'random_seed': 1,
        'vision_layers': 24,
        'transformer_layers': 1,
        'cuda': 0,
        'pretrained_text_encoder': True,
        'text_encoder_path': "./models/cliphba_behavior_text_encoder/cliphba_behavior_text_encoder.pth",
        'freeze_text': True,
        
        # Loss weights
        'p_weight': 1,
        'm_weight': 0.1,
        'g_weight': 0.15,
        
        # Architecture parameters
        'rank': 32,
        'dora_d_state': True,
        'dora_m_state': False,
        'noise_scale': 1
    }

    # Run training
    run_meg_training_group(config)

if __name__ == '__main__':
    main()