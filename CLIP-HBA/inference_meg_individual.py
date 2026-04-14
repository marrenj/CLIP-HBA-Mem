from functions.inference_meg_individual_pipeline import run_meg_individual_inference

def main():
    # Define configuration
    config = {
        # Data paths
        'img_dir': './Data/test_images', # input images directory
        'model_save_folder': './models/cliphba_meg_individual', # path to the pretrained model weights
        'save_folder': '../output/cliphba_meg_individual_demo/test_images/', # output path
        
        # Model parameters
        'backbone': 'ViT-L/14',
        'batch_size': 118,
        'vision_layers': 24,
        'transformer_layers': 1,
        'rank': 6,
        
        # Time parameters
        'smoothen_window': 5,
        'ms_start': -100,
        'ms_end': 1000,
        'ms_step': 5,
        'inference_start': -100,
        'inference_end': 1000,
        'inference_step': 5,
        'train_window_size': 15,
        
        # Other settings
        'cuda': 'cuda:0',
        'load_pretrained': True,
        'random_seed': 1
    }

    # Run inference
    run_meg_individual_inference(config)

if __name__ == '__main__':
    main()