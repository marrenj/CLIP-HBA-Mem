import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np

from torch.nn import functional as F
from tqdm import tqdm

import random
import math

from functions.spose_dimensions import *

import sys
sys.path.append('../')
from src.models.CLIPs.clip_hba_meg import clip

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
from tqdm import tqdm
import os
import shutil
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def seed_everything(seed):
    # Set the seed for PyTorch's random number generators
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set the seed for Python's random number generator
    random.seed(seed)

    # Set the seed for NumPy's random number generator
    np.random.seed(seed)

    # Ensure that the CuDNN backend is deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def load_clip_to_cpu(backbone_name, weighting_matrix, ms_start=-100, ms_step=5, ms_end=1300, train_start=100, train_step=25, train_end=800, train_window_size=30, beta=None, noise_level=None, visual_scaler = None):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), weighting_matrix=weighting_matrix, ms_start=ms_start, ms_step=ms_step, ms_end=ms_end, train_start=train_start, train_step=train_step, train_end=train_end, train_window_size=train_window_size, beta=beta, noise_level=noise_level, visual_scaler=visual_scaler)

    return model

class CLIPHBA(nn.Module):
    def __init__(self, classnames, backbone_name='RN50', pos_embedding=False, ms_start=-100, ms_step=5, ms_end=1300, train_start=100, train_step=25, train_end=800, train_window_size=30, beta=None, weighting_matrix=None, noise_level=None, visual_scaler=None):
        super().__init__()

        self.num_clip = len(classnames)
        self.clip_model = load_clip_to_cpu(backbone_name, weighting_matrix, ms_start=ms_start, ms_step=ms_step, ms_end=ms_end, train_start=train_start, train_step=train_step, train_end=train_end, train_window_size=train_window_size, beta=beta, noise_level=noise_level, visual_scaler=visual_scaler)
        self.pos_embedding = pos_embedding

        # Disable gradients for all parameters first
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # Tokenize all prompts at once and store them as a tensor
        self.tokenized_prompts = torch.stack([clip.tokenize(classname) for classname in classnames])



    def forward(self, image):
        if self.clip_model.training:
            self.clip_model.eval()

        # Move tokenized prompts to the same device as the input image
        tokenized_prompts = self.tokenized_prompts.to(image.device)

        # Process all tokenized prompts in a single forward pass
        pred_emb_3d, pred_rdm_3d, pred_feature_3d = self.clip_model(image, tokenized_prompts, self.pos_embedding)

        # # if pred_emb_3d has nan, raise error:
        # if torch.isnan(pred_emb_3d).any():
        #     raise ValueError("pred_emb_3d has NaN values")

        pred_emb_3d = pred_emb_3d.float()

        return pred_emb_3d, pred_rdm_3d, pred_feature_3d
    


class DoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, dora_alpha=16, dora_dropout=0.1):
        super(DoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r  # Low-rank factor
        self.dora_alpha = dora_alpha  # Scaling parameter
        self.dora_dropout = nn.Dropout(p=dora_dropout)

        # Decompose original weights into magnitude and direction
        with torch.no_grad():
            W = original_layer.weight.data.clone()  # [out_features, in_features]
            W = W.T  # Transpose to [in_features, out_features]
            S = torch.norm(W, dim=0)  # Magnitudes (norms of columns), shape [out_features]
            D = W / S  # Direction matrix with unit-norm columns, shape [in_features, out_features]

        # Store S as a trainable parameter
        self.m = nn.Parameter(S)  # [out_features]
        # Store D as a buffer (since we don't want to update it directly)
        self.register_buffer('D', D)  # [in_features, out_features]

        # LoRA adaptation of D
        self.delta_D_A = nn.Parameter(torch.zeros(self.r, original_layer.out_features))
        self.delta_D_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r))

        # Scaling
        self.scaling = self.dora_alpha / self.r

        # Initialize delta_D_A and delta_D_B
        self.reset_parameters()

        # Copy the bias from the original layer
        if self.original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.bias = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.delta_D_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.delta_D_B, a=math.sqrt(5))

    @property
    def weight(self):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features], add epsilon to avoid division by zero
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features], m is [out_features]

        W = W.T  # Transpose back to [out_features, in_features]

        return W

    def forward(self, x):
        # Compute adapted D
        delta_D = (self.delta_D_B @ self.delta_D_A) * self.scaling  # [in_features, out_features]
        delta_D = self.dora_dropout(delta_D)

        D_new = self.D + delta_D  # [in_features, out_features]

        # Normalize columns of D_new
        D_norms = torch.norm(D_new, dim=0, keepdim=True) + 1e-8  # [1, out_features]
        D_normalized = D_new / D_norms  # [in_features, out_features]

        # Reconstruct the adapted weight
        W = D_normalized * self.m  # [in_features, out_features]
        W = W.T  # [out_features, in_features]

        # Compute output
        return F.linear(x, W, self.bias)


    

def apply_dora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, dora_dropout=0.1, seed=123):

    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Specific blocks to modify in the visual transformer
    block_indices = range(-n_vision_layers, 0)  # Adjusted for proper indexing

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.visual.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer

    # Specific blocks to modify in the main transformer
    block_indices = range(-n_transformer_layers, 0)

    for idx in block_indices:
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a DoRALayer
        dora_layer = DoRALayer(target_layer, r=r, dora_dropout=dora_dropout)
        target_block.attn.out_proj = dora_layer



def switch_dora_layers(model, freeze_all=True, d_state=True, m_state=False):
    """
    Freeze or unfreeze the model's parameters based on the presence of DoRA layers.
    If a DoRALayer is encountered, only its specific DoRA parameters are unfrozen.
    """
    for name, param in model.named_parameters():
        # Initially set requires_grad based on the freeze_all flag
        param.requires_grad = not freeze_all

    if freeze_all:
        # If freezing all parameters, selectively unfreeze DoRA parameters
        def recursive_unfreeze_dora(module):
            for child_name, child in module.named_children():
                if isinstance(child, DoRALayer):
                    # Unfreeze DoRA-specific parameters within DoRALayer
                    child.m.requires_grad = m_state
                    child.delta_D_A.requires_grad = d_state
                    child.delta_D_B.requires_grad = d_state
                    # Keep the original layer's parameters frozen
                    if child.bias is not None:
                        child.bias.requires_grad = False
                else:
                    recursive_unfreeze_dora(child)

        # Apply selective unfreezing to the entire model
        if isinstance(model, torch.nn.DataParallel):
            recursive_unfreeze_dora(model.module)
        else:
            recursive_unfreeze_dora(model)




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
        

        np.save(f"{save_folder}/embeddings_{inference_start}ms-{inference_end}ms-{inference_step}step.npy", output_embs)
        np.save(f"{save_folder}/rdms_{inference_start}ms-{inference_end}ms-{inference_step}step.npy", output_rdms)



def run_meg_group_inference(config):
    """
    Run MEG inference with the provided configuration.
    
    Args:
        config (dict): Configuration dictionary containing inference parameters
    """
    seed_everything(config['random_seed'])
    
    # Create output directories
    if os.path.exists(config['save_folder']):
        shutil.rmtree(config['save_folder'])
        print(f"save folder reinitialized: {config['save_folder']}")
    os.makedirs(config['save_folder'])

    embedding_save_folder = f"{config['save_folder']}/emb"
    print(f"\nEmbedding will be saved to folder: {embedding_save_folder}\n")
    if os.path.exists(embedding_save_folder):
        shutil.rmtree(embedding_save_folder)
    os.makedirs(embedding_save_folder)

    # Initialize classnames
    if config['n_dim'] == 66:
        classnames = [x[0] for x in classnames66]
    else:
        raise ValueError("n_dim must be 66")
    
    # Set position embedding based on backbone
    pos_embedding = False if config['backbone'] == 'RN50' else True

    # Initialize model
    model = CLIPHBA(classnames=classnames,
                    weighting_matrix=None,
                    backbone_name=config['backbone'],
                    pos_embedding=pos_embedding,
                    ms_start=config['ms_start'],
                    ms_step=config['ms_step'],
                    ms_end=config['ms_end'],
                    train_start=config['inference_start'],
                    train_step=config['inference_step'],
                    train_end=config['inference_end'],
                    train_window_size=config['train_window_size'])

    # Load pretrained model
    if config['load_pretrained']:
        apply_dora_to_ViT(model,
                         n_vision_layers=0,
                         n_transformer_layers=config['transformer_layers'],
                         r=32,
                         dora_dropout=0.1)
        apply_dora_to_ViT(model,
                         n_vision_layers=config['vision_layers'],
                         n_transformer_layers=0,
                         r=config['rank'],
                         dora_dropout=0.1)

        print(f'Loading Pretrained Model: "{config["model_path"]}"')
        model_state_dict = torch.load(config['model_path'])
        adjusted_state_dict = {key.replace("module.", ""): value 
                             for key, value in model_state_dict.items()}
        model.load_state_dict(adjusted_state_dict)
        print('Model Loaded Successfully\n')
    else:
        raise ValueError("Please set load_pretrained to True")

    device = torch.device(config['cuda'])
    
    # Load dataset
    dataset = ImageDataset(config['img_dir'])
    data_loader = DataLoader(dataset,
                           batch_size=config['batch_size'],
                           shuffle=False)

    sample_timepoints = list(range(config['inference_start'],
                                 config['inference_end']+1,
                                 config['inference_step']))
    
    # Save model matrices
    np.save(f"{config['save_folder']}/weighting_matrix.npy",
            np.array(model.clip_model.weighting_matrix))
    
    # Run inference
    run_image(model, dataset, data_loader,
             config['save_folder'],
             embedding_save_folder,
             config['inference_start'],
             config['inference_step'],
             config['inference_end'],
             device=device,
             n_dim=config['n_dim'])


# if __name__ == '__main__':

#     seed_everything(1)
    
#     ########################################
#     img_dir = './Data/test_images/'
#     n_dim = 66 # options: 49, 66
#     backbone = 'ViT-L/14'
#     model_path = './models/cliphba_meg_group.pth'
#     save_folder = '../output/cliphba_meg_things_demo/things/'
#     batch_size = 128
#     vision_layers = 24
#     transormer_layers = 1
#     rank = 32
#     ms_start, ms_end, ms_step = -100, 1300, 5 # Shouodn't change this, this is consistent with how the model is trained
#     inference_start, inference_end, inference_step, train_window_size = -100, 1300, 5, 15 # Do not change train_window_size, this should be consistent with how the model is trained
#     cuda = 'cuda:1'
#     load_pretrained = True
#     ########################################

#     if os.path.exists(save_folder):
#         shutil.rmtree(save_folder)
#         print(f"save folder reinitialized: {save_folder}")
#     os.makedirs(save_folder)

#     # Create the directory if it doesn't exist
#     embedding_save_folder = f"{save_folder}/emb"
#     print(f"\nEmbedding will be saved to folder: {embedding_save_folder}\n")
#     if os.path.exists(embedding_save_folder):
#         shutil.rmtree(embedding_save_folder)
#     os.makedirs(embedding_save_folder)


#     if n_dim == 66:
#         classnames = classnames66
#     else: 
#         raise ValueError("n_dim must be 66")
    
#     classnames = [x[0] for x in classnames]
    
#     if backbone == 'RN50':
#         pos_embedding = False
#     if backbone == 'ViT-B/16' or backbone == 'ViT-B/32' or backbone == 'ViT-L/14': 
#         pos_embedding = True
#     # Initialize model
#     weighting_matrix = None
#     model = CLIPHBA(classnames=classnames, weighting_matrix=weighting_matrix, backbone_name=backbone, pos_embedding=pos_embedding, ms_start=ms_start, ms_step=ms_step, ms_end=ms_end, train_start=inference_start, train_step=inference_step, train_end=inference_end, train_window_size=train_window_size)
    


#     if load_pretrained:

#         apply_dora_to_ViT(model, n_vision_layers=0, n_transformer_layers=transormer_layers, r=32, dora_dropout=0.1)
#         apply_dora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=0, r=rank, dora_dropout=0.1)


#         print(f'Loading Pretrained Model: "{model_path}"')
#         model_state_dict = torch.load(model_path)
#         adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
#         model.load_state_dict(adjusted_state_dict)
#         print('Model Loaded Successfully\n')
    
#     else:
#         raise ValueError("Please set load_pretrained to True")


#     device = torch.device(cuda)
    
#     # Load the dataset
#     dataset = ImageDataset(img_dir)
#     data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     sample_timepoints = list(range(inference_start, inference_end+1, inference_step))
        
#     np.save(f"{save_folder}/weighting_matrix.npy", np.array(model.clip_model.weighting_matrix))
#     # np.save(f"{save_folder}/dynamic_dim_matrix.npy", np.array(model.clip_model.dynamic_dim_matrix))
#     # np.save(f"{save_folder}/richness_matrix.npy", np.array(model.clip_model.richness))

    
#     # Run the model and save output embeddings
#     hba_output = run_image(model, dataset, data_loader, save_folder, embedding_save_folder, inference_start, inference_step, inference_end, device=device, n_dim=n_dim)


