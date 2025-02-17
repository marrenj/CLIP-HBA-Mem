from functions.inference_meg_group_pipeline import run_meg_group_inference

def main():
    # Define configuration
    config = {
        # Data paths
        'img_dir': './Data/test_images/',
        'n_dim': 66,  # options: 66
        
        # Model parameters
        'backbone': 'ViT-L/14',
        'model_path': './models/cliphba_meg_group.pth',
        'save_folder': '../output/cliphba_meg_things_demo/things/',
        'batch_size': 128,
        'vision_layers': 24,
        'transformer_layers': 1,
        'rank': 32,
        
        # Time parameters
        'ms_start': -100,
        'ms_end': 1300,
        'ms_step': 5,
        'inference_start': -100,
        'inference_end': 1300,
        'inference_step': 5,
        'train_window_size': 15,
        
        # Device settings
        'cuda': 'cuda:1',
        'load_pretrained': True,
        'random_seed': 1
    }

    # Run inference
    run_meg_group_inference(config)

if __name__ == '__main__':
    main()