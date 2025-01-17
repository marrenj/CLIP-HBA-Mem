from train_dynamic_step import  *
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import shutil
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
from torch.nn import DataParallel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import h5py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class ImageDataset(Dataset):
    def __init__(self, img_path):
        """
        img_path can be either a directory containing images or a path to a single image.
        """
        if os.path.isdir(img_path):
            self.img_dir = img_path
            self.image_names = [img for img in sorted(os.listdir(img_path))
                                if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        elif os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.img_dir, single_image_name = os.path.split(img_path)
            self.image_names = [single_image_name]  # Single image in a list
        else:
            raise ValueError(f"Provided path '{img_path}' is neither a directory nor a file, or file type is not supported.")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                 std=[0.27608301, 0.26593025, 0.28238822])
        ])


    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert('RGB')  # Ensure image is RGB
        image = self.transform(image)
        
        return image_name, image

def compute_temporal_object_rdm(emb):
    # Get the number of timepoints, objects, and dimensions
    num_timepoints, num_objects, _ = emb.shape
    
    # Initialize an array to store the RDMs for each timepoint
    rdm_array = np.zeros((num_timepoints, num_objects, num_objects))
    
    # Loop over each timepoint
    for t in range(num_timepoints):
        # Compute the Pearson correlation matrix for objects at timepoint t
        corr_matrix = np.corrcoef(emb[t])
        
        # Convert correlation to Pearson distance (1 - correlation)
        rdm_array[t] = 1 - corr_matrix
    

    return rdm_array
    
def run_image(model, dataset, data_loader, save_folder, embedding_save_folder, inference_start, inference_step, inference_end ,device=torch.device("cuda"), n_dim=66):
    model.eval()
    model.to(device)

    n_images =  len(dataset)
    
    
    with torch.no_grad():
        n_inference_timepoints = len(range(inference_start, inference_end+1, inference_step))
        image_names = []
        output_embs = np.zeros((n_inference_timepoints, n_images, n_dim))
        
        # To keep track of the current position in the dataset
        current_idx = 0

        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Processing images")
        for batch_idx, (batch_image_names, batch_images) in progress_bar:
            batch_images = batch_images.to(device)
            image_names.extend(batch_image_names)
            
            # Perform the forward pass
            pred_embs, _, _ = model(batch_images)

            # Move to CPU and convert to numpy
            pred_embs = pred_embs.cpu().numpy()  # shape: (n_inference_timepoints, batch_size, n_dim)

            # # zero out negative values in predictions
            # pred_embs = np.maximum(pred_embs, 0)


            # Get the size of the current batch
            batch_size = batch_images.size(0)

            # Store embeddings and RDMs in the output arrays
            output_embs[:, current_idx:current_idx + batch_size, :] = pred_embs  # Store embeddings

            # Update current index for the next batch
            current_idx += batch_size

        print(f"Finished processing {n_images} images")
        print(f"output_embs shape: {output_embs.shape}")

        # min max output_embs overall to 0-100
        output_embs = (output_embs - np.min(output_embs)) / (np.max(output_embs) - np.min(output_embs)) * 100

        for i in range(n_inference_timepoints):
            timepoint = inference_start + i*inference_step
            hba_embedding = pd.DataFrame(output_embs[i])
            hba_embedding['image'] = image_names
            hba_embedding = hba_embedding[['image'] + [col for col in hba_embedding if col != 'image']]
            emb_save_path = f"{embedding_save_folder}/dynamic_embedding_{timepoint}ms.csv"
            hba_embedding.to_csv(emb_save_path, index=False)
            print(f"Embedding saved to {emb_save_path}")

        output_rdms = compute_temporal_object_rdm(output_embs)
        # for i in range(n_inference_timepoints):
        #     timepoint = inference_start + i*inference_step
        #     rdm_save_path = f"{rdm_save_folder}/dynamic_rdm_{timepoint}ms.npy"
        #     np.save(rdm_save_path, output_rdms[i])
        #     print(f"RDM saved to {rdm_save_path}")
        

        

        np.save(f"{save_folder}/embeddings_{inference_start}ms-{inference_end}ms-{inference_step}step.npy", output_embs)
        np.save(f"{save_folder}/rdms_{inference_start}ms-{inference_end}ms-{inference_step}step.npy", output_rdms)




if __name__ == '__main__':

    seed_everything(1)
    
    ########################################
    img_dir = './Data/test_images/'
    n_dim = 66 # options: 49, 66
    backbone = 'ViT-L/14'
    model_path = './models/cliphba_dynamic_official_v9_step.pth'
    save_folder = '../output/cliphba_dynamic_66d_official_v2/test_images/'
    batch_size = 128
    vision_layers = 24
    transormer_layers = 1
    rank = 32
    ms_start, ms_end, ms_step = -100, 1300, 5 # Shouodn't change this, this is consistent with how the model is trained
    inference_start, inference_end, inference_step, train_window_size = -100, 1300, 5, 15 # Do not change train_window_size, this should be consistent with how the model is trained
    cuda = 'cuda:1'
    load_pretrained = True
    text_encoder_path = "./models/partial/cliphba_text_encoder.pth"
    weighting_matrix_path = "./Encoder_Correspondence/weighting_matrix/weighting_matrix_things_cv_updated.npy"
    ########################################

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
        print(f"save folder reinitialized: {save_folder}")
    os.makedirs(save_folder)

    # Create the directory if it doesn't exist
    embedding_save_folder = f"{save_folder}/emb"
    print(f"\nEmbedding will be saved to folder: {embedding_save_folder}\n")
    if os.path.exists(embedding_save_folder):
        shutil.rmtree(embedding_save_folder)
    os.makedirs(embedding_save_folder)


    if n_dim == 66:
        classnames = classnames66
    else: 
        raise ValueError("n_dim must be 66")
    
    classnames = [x[0] for x in classnames]
    
    if backbone == 'RN50':
        pos_embedding = False
    if backbone == 'ViT-B/16' or backbone == 'ViT-B/32' or backbone == 'ViT-L/14': 
        pos_embedding = True
    # Initialize model
    # weighting_matrix = torch.tensor(np.load(weighting_matrix_path), dtype=torch.float32)
    weighting_matrix = None
    model = CLIPHBA(classnames=classnames, weighting_matrix=weighting_matrix, backbone_name=backbone, pos_embedding=pos_embedding, ms_start=ms_start, ms_step=ms_step, ms_end=ms_end, train_start=inference_start, train_step=inference_step, train_end=inference_end, train_window_size=train_window_size)
    


    if load_pretrained:

        # apply_lora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=16, lora_dropout=0.1)
        # apply_lora_to_ViT(model, n_vision_layers=0, n_transformer_layers=transormer_layers, r=16, lora_dropout=0.1)

        apply_dora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=rank, dora_dropout=0.1)


        print(f'Loading Pretrained Model: "{model_path}"')
        model_state_dict = torch.load(model_path)
        adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        model.load_state_dict(adjusted_state_dict)
        print('Model Loaded Successfully\n')
    else:
        print(f"Using Original CLIP {backbone}")
        apply_lora_to_ViT(model, n_vision_layers=0, n_transformer_layers=1, r=16, lora_dropout=0.1)
        print(f"Loading pretrained model: {text_encoder_path}")
        model_state_dict = torch.load(text_encoder_path)
        adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
        # Load the state dict into the model with strict=False
        missing_keys, unexpected_keys = model.load_state_dict(adjusted_state_dict, strict=False)
        # Print out any keys in the state dict that were not loaded by the model
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(adjusted_state_dict.keys())
        not_loaded_keys = loaded_keys - model_keys
        
        if not_loaded_keys != set():
            print(f"Text Encoder State Dict Keys not loaded: {not_loaded_keys}")
        else:
            print("Pretrained Text Encoder loaded successfully\n")



    device = torch.device(cuda)
    
    # Load the dataset
    dataset = ImageDataset(img_dir)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sample_timepoints = list(range(inference_start, inference_end+1, inference_step))
        
    np.save(f"{save_folder}/weighting_matrix.npy", np.array(model.clip_model.weighting_matrix))
    # np.save(f"{save_folder}/dynamic_dim_matrix.npy", np.array(model.clip_model.dynamic_dim_matrix))
    # np.save(f"{save_folder}/richness_matrix.npy", np.array(model.clip_model.richness))

    
    # Run the model and save output embeddings
    hba_output = run_image(model, dataset, data_loader, save_folder, embedding_save_folder, inference_start, inference_step, inference_end, device=device, n_dim=n_dim)


