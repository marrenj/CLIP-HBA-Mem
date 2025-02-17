from functions.train_meg_individual_pipeline import run_individual_training

def main():
    # Define configuration
    config = {
        # Data paths
        'csv_file': './Data/Cichy/cichy_100_images_list_train.csv',
        'img_dir': './Data/Cichy/stimuli',
        'rdm_dir': "./Data/Cichy/Cichy_MEG_RDM_rescaled.npy",
        'model_save_folder': './models/cliphba_meg_individual_test',
        
        # Participant settings
        'n_participants': 15, # total number of participants
        'skip_id': [], # skip the training of specific participants
        
        # Model parameters
        'backbone': 'ViT-L/14',
        'epochs': 40, # number of epochs for training
        'fw_tuning_epochs': 10, # number of epochs for fine-tuning of the feature reweighting matrix before ViT joins the training
        'batch_size': 40,
        'train_portion': 80/100,
        'smoothen_window': 5,
        
        # Time parameters
        'ms_start': -100,
        'ms_end': 1000,
        'ms_step': 5,
        'train_start': -100,
        'train_end': 1000,
        'train_step': 5,
        'train_window_size': 15,
        
        # Learning parameters
        'lr_1': 0,
        'lr_2': 3e-5,
        'fw_lr_1': 3e-3,
        'fw_lr_2': 3e-3,
        'early_stopping_patience_0': 3,
        'early_stopping_patience_1': 10,
        
        # Model architecture
        'vision_layers': 24,
        'transformer_layers': 1,
        'rank': 6,
        'dora_d_state': True,
        'dora_m_state': False,
        
        # Loss weights
        'p_weight': 1,
        'm_weight': 0.15,
        'g_weight': 0.1,
        
        # Other settings
        'cuda': 1,
        'random_seed': 1,
        'noise_scale': 1,
        'text_encoder_path': "./models/cliphba_behavior_text_encoder/cliphba_behavior_text_encoder.pth",
        'freeze_text': True
    }

    # Run training
    run_individual_training(config)

if __name__ == '__main__':
    main()