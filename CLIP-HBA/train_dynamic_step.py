import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import torchvision.models as models
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmcv.runner.checkpoint import _load_checkpoint_with_prefix

from torch.nn import functional as F
import copy
from tqdm import tqdm

from torch.optim import Adam, SGD, AdamW
from torch.nn import DataParallel
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from torch.utils.data import Subset
import random
import math
import torch.optim as optim

import sys
sys.path.append('../')
from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.components.clip_hba_dynamic_3d import clip
from mmedit.models.components.clip_hba_dynamic_3d.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.nn.utils as utils
from scipy.ndimage import gaussian_filter1d

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Prompt pairs for the model
classnames66 = [
    ['metallic; artificial', 'nonmetallic; natural'],
    ['food-related', 'non-food-related'],
    ['animal-related', 'non-animal-related; artificial; inorganic'],
    ['textile', 'non-textile'],
    ['plant-related', 'non-plant-related'],
    ['house-related; furnishing-related',
    'non-house-related; non-furnishing-related; outdoors-related'],
    ['valuable; precious', 'non-valuable; inexpensive; cheap'],
    ['transportation; movement-related', 'non-transportation; movement-related'],
    ['body; people-related', 'non-body; people-related'],
    ['wood-related; brown', 'non-wood-related; non-brown'],
    ['electronics; technology', 'non-electronics; technology'],
    ['colorful; playful', 'monotone; dull'],
    ['outdoors', 'non-outdoors'],
    ['circular; round', 'non-circular; angular'],
    ['paper-related; flat', 'non-paper-related; bumpy'],
    ['hobby-related; game-related; playing-related',
    'non-hobby-related; non-game-related; non-playing-related'],
    ['tools-related; handheld; elongated',
    'non-tools-related; non-handheld; shortened'],
    ['fluid-related; drink-related', 'solid-related; non-fluid-related'],
    ['water-related', 'dry; non-water-related'],
    ['oriented; many; plenty', 'non-oriented; few; scarce'],
    ['powdery; earth-related; waste-related',
    'non-powdery; non-earth-related; non-waste-related'],
    ['white', 'black; dark'],
    ['coarse-scale pattern; many things', 'non-coarse-scalepattern; few; scarce'],
    ['red', 'non-red'],
    ['long; thin', 'short; thick'],
    ['weapon-related; war-related; dangerous',
    'non-weapon-related; non-war-related; peaceful'],
    ['black', 'non-black; white'],
    ['household-related', 'non-household-related'],
    ['feminine', 'non-feminine; masculine'],
    ['body-part-related', 'non-body-part-related'],
    ['tubular', 'non-tubular'],
    ['music-related; hearing-related; hobby-related; loud',
    'non-music-related; non-hearing-related; non-hobby-related; silent'],
    ['grid-related; grating-related', 'non-grid-related; non-grating-related'],
    ['repetitive; spiky', 'diverse; dull'],
    ['construction-related; craftsmanship-related; housework-related',
    'non-physical-work-related'],
    ['spherical; voluminous', 'angular; small'],
    ['string-related; stringy; curved', 'non-string-related; straight'],
    ['seating; standing; lying-related',
    'non-seating; non-standing; non-lying-related'],
    ['flying-related; sky-related', 'ground-related; non-flying-related'],
    ['bug-related; non-mammalian; disgusting',
    'non-bug-related; mammalian; pleasant'],
    ['transparent; shiny; crystalline', 'opaque; matte; frosted'],
    ['sand-colored', 'non-sand-colored'],
    ['green', 'non-green'],
    ['bathroom-related; wetness-related', 'non-bathroom-related; dry'],
    ['yellow', 'non-yellow'],
    ['heat-related; fire-related; light-related', 'cold-related; dark-related'],
    ['beams-related; mesh-related', 'non-beams-related; non-mesh-related'],
    ['foot-related; walking-related', 'non-foot-related; non-walking-related'],
    ['box-related; container', 'non-box-related; non-container'],
    ['stick-shaped; container', 'non-stick-shaped; non-container'],
    ['head-related', 'non-head-related'],
    ['upright; elongated; volumous', 'non-upright; short; small'],
    ['pointed; spiky', 'non-pointed; blunt'],
    ['child-related; toy-related; cute', 'mature; adult-like'],
    ['farm-related; historical', 'non-farm-related; contemporary'],
    ['seeing-related', 'non-seeing-related; blind'],
    ['medicine-related; health-related',
    'non-medicine-related; non-health-related'],
    ['sweet; dessert-related', 'non-sweet; non-dessert-related'],
    ['orange', 'non-orange'],
    ['thin; flat; wrapping', 'thick; bumpy; non-wrapping'],
    ['cylindrical; conical; cushioning', 'triangular; angular; hardening'],
    ['coldness-related; winter-related', 'warmness-related; non-winter-related '],
    ['measurement-related; numbers-related',
    'non-measurement-related; non-numerical'],
    ['fluffy; soft', 'non-fluffy; rough'],
    ['masculine', 'non-masculine; feminine'],
    ['fine-grained; pattern', 'non-fine-grained; plain']
]
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


def load_rdm(rdm_path):
    rdm = np.load(rdm_path)
    # rdm = rdm[:-1, :, :, :]  # Remove the last participant as it is not used in the model
    # Check for NaN values
    nan_check = np.isnan(rdm).any()

    if nan_check:
        rdm = np.nan_to_num(rdm, copy=False)

    # Check scale of the RDM
    min_val = np.min(rdm)
    max_val = np.max(rdm)


    if min_val != 0 or max_val != 1:
        rdm = (rdm - min_val) / (max_val - min_val)  # Scale to [0, 1]
        rdm_max = np.max(rdm)
        rdm_min = np.min(rdm)
        print(f"RDM scaled to {rdm_min}-{rdm_max}")

    # print(f"RDM loaded and preprocessed, shape: {rdm.shape}")

    return rdm

def get_richness(rdms, zero_ms_position):

    p_richness = []

    for p in range(rdms.shape[0]):

        rdm = rdms[p, :, :, :]
        richness = np.mean(rdm, axis=(1, 2))
        p_richness.append(richness)

    avg_richness = np.mean(p_richness, axis=0)

    avg_richness_min_maxed = (avg_richness - avg_richness.min()) / (avg_richness.max() - avg_richness.min())

    avg_richness_min_maxed[:zero_ms_position] = 0

    # avg_richness_min_maxed = gaussian_filter1d(avg_richness_min_maxed, sigma=3)


    return avg_richness_min_maxed


def compute_average_participant_neural_richness(rdms):
    def compute_richness(rdm):
        # input rdm should be 3 dimensional: (n_timepoints, n_objects, n_objects)

        def flatten_rdm(rdm):
            triu_indices = np.triu_indices(rdm.shape[-1], k=1)
            return rdm[..., triu_indices[0], triu_indices[1]]
        
        def compute_time_rsm(rdm):
            rdm_flat = flatten_rdm(rdm)
            time_rsm = np.corrcoef(rdm_flat)
            return time_rsm
        
        time_rsm = compute_time_rsm(rdm)
        richness = np.mean(time_rsm, axis=1)
        return richness



    # input rdm should be 3 dimensional: (n_participants, n_timepoints, n_objects, n_objects)
    richness_per_participant = []
    for p in range(rdms.shape[0]):
        participant_rdm = rdms[p, :, :, :]
        richness = compute_richness(participant_rdm)
        richness_per_participant.append(richness)

    avg_richness = np.mean(richness_per_participant, axis=0)
    avg_richness = (avg_richness - avg_richness.min()) / (avg_richness.max() - avg_richness.min())
    return avg_richness

class DynamicDataset(Dataset):
    def __init__(self, csv_file, img_dir, rdm_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                 std=[0.27608301, 0.26593025, 0.28238822])
        ])
        
        # Read the CSV file and store the image names and indices
        self.annotations = pd.read_csv(csv_file, index_col=0)
        self.image_names = self.annotations.iloc[:, 0].tolist()
        self.image_indices = self.annotations.index.tolist()
        
        # Load the full RDM matrix
        self.rdms = load_rdm(rdm_dir)
        
        # Create a mapping from image names to indices for efficient lookup
        self.image_name_to_index = {name: idx for name, idx in zip(self.image_names, self.image_indices)}

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        # Get the index of the image_name using the precomputed mapping
        image_name_index = self.image_name_to_index[image_name]

        return image_name, image, image_name_index

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
    


class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, lora_alpha=16, lora_dropout=0.1, seed = 123):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.set_seed(seed)

        self.lora_A = nn.Parameter(torch.randn(self.r, original_layer.out_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r))
        self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))

    def forward(self, x):
        lora_B = self.lora_B.to(dtype=x.dtype)
        lora_A = self.lora_A.to(dtype=x.dtype)
        return self.original_layer(x) + (self.lora_dropout(x) @ lora_B @ lora_A) * self.scaling

    @property
    def weight(self):
        return (self.original_layer.weight.to(self.lora_B.dtype) + (self.lora_B @ self.lora_A) * self.scaling).to(self.original_layer.weight.dtype)

    @property
    def bias(self):
        return self.original_layer.bias

def apply_lora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, lora_dropout=0.1, seed=123):

    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Specific blocks to modify
    block_indices = -n_vision_layers

    for idx in range(block_indices, 0):
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.visual.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a LoRALayer
        lora_layer = LoRALayer(target_layer, r=r, lora_dropout=lora_dropout, seed=seed)
        target_block.attn.out_proj = lora_layer

    block_indices = -n_transformer_layers
    for idx in range(block_indices, 0):
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a LoRALayer
        lora_layer = LoRALayer(target_layer, r=r, lora_dropout=lora_dropout, seed=seed)
        target_block.attn.out_proj = lora_layer



def switch_lora_layers(model, freeze_all=True, lora_state=True):
    """
    Freeze or unfreeze the model's parameters based on the presence of LoRA layers.
    If a LoRALayer is encountered, only its specific LoRA parameters are unfrozen.
    """
    for name, param in model.named_parameters():
        # Initially set requires_grad based on the freeze_all flag
        param.requires_grad = not freeze_all

    if freeze_all:
        # If freezing all parameters, selectively unfreeze LoRA parameters
        def recursive_unfreeze_lora(module):
            for child_name, child in module.named_children():
                if isinstance(child, LoRALayer):
                    # Unfreeze only LoRA-specific parameters within LoRALayer
                    child.lora_A.requires_grad = lora_state
                    child.lora_B.requires_grad = lora_state
                    # Keep the original layer's parameters frozen
                    child.original_layer.weight.requires_grad = False
                    if child.original_layer.bias is not None:
                        child.original_layer.bias.requires_grad = False
                else:
                    recursive_unfreeze_lora(child)

        # Apply selective unfreezing to the entire model
        if isinstance(model, torch.nn.DataParallel):
            recursive_unfreeze_lora(model.module)
        else:
            recursive_unfreeze_lora(model)


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



def switch_dora_layers(model, freeze_all=True, dora_state=True):
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
                    child.m.requires_grad = dora_state
                    child.delta_D_A.requires_grad = dora_state
                    child.delta_D_B.requires_grad = dora_state
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




def unfreeze_weighting_parameters(model):
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    model_module.clip_model.weighting_matrix.requires_grad = True
    model_module.clip_model.noise_level.requires_grad = False


def freeze_text_encoder(model):
    """
    Freeze the text encoder of the model.
    """
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    for name, param in model_module.clip_model.transformer.named_parameters():
            param.requires_grad = False

def unfreeze_parameters(model):
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    model_module.clip_model.logit_scale.requires_grad = True
    for param in model_module.clip_model.dim_weights_matrix:
        param.requires_grad = True


def get_logit_scale_parameter(model):
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    return model_module.clip_model.logit_scale

def get_dim_weight_parameter(model):
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model
    
    return model_module.clip_model.dim_weights_matrix




def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return sum(p.numel() for p in model.parameters())



def calculate_cosine_rdm(predictions):
    # Calculate the pairwise cosine similarity between predictions
    similarity_matrix = torch.nn.functional.cosine_similarity(predictions.unsqueeze(1), predictions.unsqueeze(0), dim=2)
    
    # Convert cosine similarity to dissimilarity (distance)
    dissimilarity_matrix = 1 - similarity_matrix
    
    # Set the diagonal elements to zero
    rdm = dissimilarity_matrix.fill_diagonal_(0)
    
    return rdm


def calculate_pearson_rdm(predictions):
    # Calculate the mean for each prediction (batch of vectors)
    mean_predictions = torch.mean(predictions, dim=1, keepdim=True)
    
    # Subtract the mean from predictions (centering)
    centered_predictions = predictions - mean_predictions
    
    # Calculate the pairwise dot product of centered predictions
    dot_product = torch.mm(centered_predictions, centered_predictions.t())
    
    # Calculate the norms (magnitude) of each prediction
    norms = torch.norm(centered_predictions, dim=1, keepdim=True)
    
    # Calculate the Pearson correlation
    similarity_matrix = dot_product / (norms * norms.t())
    
    # Convert Pearson correlation to Pearson distance
    dissimilarity_matrix = 1 - similarity_matrix
    
    # Set the diagonal elements to zero
    rdm = dissimilarity_matrix.fill_diagonal_(0)
    
    return rdm



def ms_to_timepoints(ms, ms_start=-100, ms_step=5):
    timepoint = (ms - ms_start) // ms_step
    return timepoint

def compute_rdm_generalization(rdm, zero_ms_position): 
    def flatten_rdm(rdm):
        triu_indices = np.triu_indices(rdm.shape[-1], k=1)
        return rdm[..., triu_indices[0], triu_indices[1]]
    
    def compute_time_rsm(rdm):
        rdm_flat = flatten_rdm(rdm)
        time_rsm = np.corrcoef(rdm_flat)
        # time_rsm = 1 - squareform(pdist(rdm_flat, metric='euclidean'))
        return time_rsm
    

    p_generalization = []
    for p in range(rdm.shape[0]):
        time_rsm = compute_time_rsm(rdm[p, :, :, :])
        generalization = np.mean(time_rsm, axis=1)
        generalization_min_maxed = (generalization - generalization.min()) / (generalization.max() - generalization.min())
        p_generalization.append(generalization_min_maxed)


    generalization = np.mean(p_generalization, axis=0)

    generalization[:zero_ms_position] = 0

    # generalization = gaussian_filter1d(generalization, sigma=3)

    return generalization
    
class PearsonMSELoss3D(nn.Module):
    def __init__(self, dynamic_penalty, initial_pearson_loss=1, initial_mse_loss=1, initial_generalization_loss=1, p_weight=1, m_weight=1, g_weight=1):
        super(PearsonMSELoss3D, self).__init__()
        self.initial_pearson_loss = initial_pearson_loss
        self.initial_mse_loss = initial_mse_loss
        self.initial_generalization_loss = initial_generalization_loss
        self.dynamic_penalty = torch.tensor(dynamic_penalty/dynamic_penalty.sum(), dtype=torch.float)
        self.p_weight = p_weight
        self.m_weight = m_weight
        self.g_weight = g_weight
        # print(f"Dynamic Penalty: {dynamic_penalty.shape}")
        

    
    def pearson_loss(self, x_flat, y_flat):
        mean_x = torch.mean(x_flat)
        mean_y = torch.mean(y_flat)
        xm = x_flat - mean_x
        ym = y_flat - mean_y
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
        correlation = r_num / r_den
        pearson_loss = 1 - correlation
        return pearson_loss

    def pearson_mse_weighted_loss(self, pred, target):
        # pred and target shapes are [n_timepoints, n_objects, n_objects]
    
        n_timepoints, batch, _ = pred.shape
        device = pred.device
        triu_indices = torch.triu_indices(batch, batch, offset=1).to(device)
        
        pred_upper = pred[:, triu_indices[0], triu_indices[1]]
        target_upper = target[:, triu_indices[0], triu_indices[1]]

        # Compute Pearson loss across timepoints
        pearson_losses = torch.stack([
            self.pearson_loss(pred_upper[i], target_upper[i])
            for i in range(n_timepoints)
        ])

        # Compute MSE loss across each time point
        mse_losses_within_timepoints = torch.stack([
            F.mse_loss(pred_upper[i], target_upper[i])
            for i in range(n_timepoints)
        ])

        weighted_pearson_losses = torch.sum(pearson_losses * self.dynamic_penalty.to(pearson_losses.device))
        weighted_mse_losses = torch.sum(mse_losses_within_timepoints * self.dynamic_penalty.to(mse_losses_within_timepoints.device))

        # average_pearson_loss = torch.mean(pearson_losses)
        # average_mse_loss = torch.mean(mse_losses_within_timepoints)


        return weighted_pearson_losses, weighted_mse_losses
    
    def compute_time_generalization(self, rdm):
        def flatten_rdm(rdm):
            triu_indices = torch.triu_indices(rdm.shape[-1], rdm.shape[-1], offset=1)
            return rdm[..., triu_indices[0], triu_indices[1]]

        def compute_time_rsm(rdm):
            rdm_flat = flatten_rdm(rdm)
            rdm_mean = rdm_flat - rdm_flat.mean(dim=-1, keepdim=True)
            rdm_std = rdm_flat.std(dim=-1, unbiased=False, keepdim=True)
            rdm_flat = rdm_mean / rdm_std
            time_rsm = torch.matmul(rdm_flat, rdm_flat.transpose(-1, -2))
            time_rsm /= rdm_flat.shape[-1] - 1
            return time_rsm

        rsm = compute_time_rsm(rdm)
        generalization = torch.mean(rsm, dim=1)
        
        return generalization
    

    def forward(self, x, y):
        p_loss, mse_loss = self.pearson_mse_weighted_loss(x, y)

        x_generalization = self.compute_time_generalization(x)
        y_generalization = self.compute_time_generalization(y)
        g_loss = self.pearson_loss(x_generalization, y_generalization) + nn.MSELoss()(x_generalization, y_generalization)

        total_loss = self.p_weight * p_loss/self.initial_pearson_loss + self.m_weight * mse_loss/self.initial_mse_loss + self.g_weight * g_loss/self.initial_generalization_loss


        return total_loss, mse_loss, p_loss, g_loss
    

class PearsonMSELongLoss(nn.Module):
    def __init__(self, initial_pearson_loss=1, initial_mse_loss=1, initial_generalization_loss = 1, p_weight=1, m_weight=1, g_weight = 1):
        super(PearsonMSELongLoss, self).__init__()
        self.initial_pearson_loss = initial_pearson_loss
        self.initial_mse_loss = initial_mse_loss
        self.initial_generalization_loss = initial_generalization_loss
        self.p_weight = p_weight
        self.m_weight = m_weight
        self.g_weight = g_weight

    
    def pearson_loss(self, x_flat, y_flat):
        mean_x = torch.mean(x_flat)
        mean_y = torch.mean(y_flat)
        xm = x_flat - mean_x
        ym = y_flat - mean_y
        r_num = torch.sum(xm * ym)
        r_den = torch.sqrt(torch.sum(xm ** 2) * torch.sum(ym ** 2))
        correlation = r_num / r_den
        pearson_loss = 1 - correlation
        return pearson_loss

    def pearson_mse_long_loss(self, pred, target):
        # pred and target shapes are [n_timepoints, n_objects, n_objects]
    
        n_timepoints, batch, _ = pred.shape
        device = pred.device
        triu_indices = torch.triu_indices(batch, batch, offset=1).to(device)
        
        pred_upper = pred[:, triu_indices[0], triu_indices[1]]
        target_upper = target[:, triu_indices[0], triu_indices[1]]

        # flatten pred_upper and target_upper all the way to 1d long vector
        pred_long = pred_upper.view(-1)
        target_long = target_upper.view(-1)

        # Compute Pearson loss across timepoints
        pearson_loss = self.pearson_loss(pred_long, target_long)

        # mse_loss = F.mse_loss(pred_long, target_long)

        # slice by slice mse
        slice_mse = []
        for t in range(n_timepoints):
            slice_mse.append(F.mse_loss(pred_upper[t], target_upper[t]))
        
        mse_loss = torch.mean(torch.stack(slice_mse))

        return pearson_loss, mse_loss
    

    def compute_time_generalization(self, rdm):
        def flatten_rdm(rdm):
            triu_indices = torch.triu_indices(rdm.shape[-1], rdm.shape[-1], offset=1)
            return rdm[..., triu_indices[0], triu_indices[1]]

        def compute_time_rsm(rdm):
            rdm_flat = flatten_rdm(rdm)
            rdm_mean = rdm_flat - rdm_flat.mean(dim=-1, keepdim=True)
            rdm_std = rdm_flat.std(dim=-1, unbiased=False, keepdim=True)
            rdm_flat = rdm_mean / rdm_std
            time_rsm = torch.matmul(rdm_flat, rdm_flat.transpose(-1, -2))
            time_rsm /= rdm_flat.shape[-1] - 1
            return time_rsm

        rsm = compute_time_rsm(rdm)
        generalization = torch.mean(rsm, dim=1)
        
        return generalization
    

    def forward(self, x, y):
        p_loss, mse_loss = self.pearson_mse_long_loss(x, y)

        # just to show the generalization loss
        x_generalization = self.compute_time_generalization(x)
        y_generalization = self.compute_time_generalization(y)
        # g_loss = self.pearson_loss(x_generalization, y_generalization) + nn.MSELoss()(x_generalization, y_generalization)
        # g_loss = self.pearson_loss(x_generalization, y_generalization) + 0.2 * (torch.mean(x_generalization) - torch.mean(y_generalization))**2
        g_loss = self.pearson_loss(x_generalization, y_generalization)


        total_loss = self.p_weight * p_loss/self.initial_pearson_loss + self.m_weight * mse_loss/self.initial_mse_loss + self.g_weight * g_loss/self.initial_generalization_loss
        
        return total_loss, mse_loss, p_loss, g_loss

    








def train_model(model, train_loader, test_loader, device, criterion, p_weight, m_weight, g_weight, optimizer_0, optimizer_1, epochs, fw_tuning_epochs, rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, early_stopping_patience=5, checkpoint_path='clip_hba_model_cv.pth'):

    model.train()
    best_test_loss = p_weight + m_weight + g_weight
    best_pearson_loss = 999
    epochs_no_improve = 0
    optimizer = optimizer_0

    starting_timepoint = ms_to_timepoints(sample_timepoints[0], ms_start, ms_step)
    ending_timepoint = ms_to_timepoints(sample_timepoints[-1], ms_start, ms_step) + 1
    target_rdms = rdms[:, starting_timepoint:ending_timepoint, :, :]
    # print("target rdm shape: ", target_rdms.shape)

    train_rdms = np.mean(target_rdms[:-1, :, :, :], axis=0) # first 2 participants' MEG Rdms for training
    test_rdms = np.mean(target_rdms[:-1, :, :, :], axis=0) # using 3rd participant's RDM for testing, 4th quality is bad
    # Convert train_loader to list so we can shuffle it later
    train_data = list(train_loader)


    # Initial evaluation
    torch.save(model.state_dict(), checkpoint_path)
    print("\n--- Initial Evaluation Starting ---")
    t, m, p, g = evaluate_model(model, test_loader, device, criterion, test_rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, optimizer)
    best_pearson_loss = p
    criterion = PearsonMSELongLoss(initial_mse_loss=m, initial_pearson_loss=p, initial_generalization_loss=g ,p_weight=p_weight, m_weight=m_weight, g_weight=g_weight)
    print(f"Initial Validation Loss: T={best_test_loss:.4f}, M={m:.4f}, P={p:.4f}, G={g:.4f}")
    print("--- Initial Evaluation Complete ---\n")


    print("--- Training Starting ---")
    for epoch in range(epochs):
        
        if fw_tuning_epochs is not None:

            if epoch > fw_tuning_epochs - 1:
                optimizer = optimizer_1
            else:
                optimizer = optimizer_0
                epochs_no_improve = 0

            if epoch == fw_tuning_epochs:
                print("\n\n*********************************")
                print(f"ViT Starts training at epoch {epoch+1}")
                print("*********************************\n\n")

                # load the latest checkpoint
                model.load_state_dict(torch.load(checkpoint_path))
        
        total_loss = 0.0 
        total_iterations = len(train_loader)

        # Shuffle train data at the start of each epoch
        random.shuffle(train_data)


        progress_bar = tqdm(total=total_iterations, desc=f"Epoch {epoch+1}/{epochs}")


        for batch_idx, (image_name, images, indices) in enumerate(train_data):

            # Shuffle the images and indices within the batch
            perm = torch.randperm(images.size(0))  # Generate a random permutation
            images = images[perm]
            indices = [indices[i] for i in perm.tolist()]

            # print(f"Indices: {indices}")

            images = images.to(device)

            optimizer.zero_grad()

            pred_emb_3d, pred_rdm_3d, _ = model(images)

            target_rdm_3d = train_rdms[:, np.ix_(indices, indices)[0], np.ix_(indices, indices)[1]]
            target_rdm_3d = torch.tensor(target_rdm_3d, dtype=torch.float).to(device)


            loss, mse_loss, pearson_loss, g_loss = criterion(pred_rdm_3d, target_rdm_3d)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix({'Loss': loss.item(), 'M': mse_loss.item(), 'P': pearson_loss.item(), 'G': g_loss.item()})
            progress_bar.update(1)
        progress_bar.close()
                

                
        avg_train_loss = total_loss / total_iterations  # Average loss for the epoch
        progress_bar.close()

        # Evaluate after every epoch
        avg_test_loss, avg_mse_loss, avg_pearson_loss, avg_g_loss = evaluate_model(model, test_loader, device, criterion, test_rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, optimizer)
        print(f"Epoch {epoch+1}: Training Loss: T={avg_train_loss:.4f}, Validation Loss: T={avg_test_loss:.4f}, M={avg_mse_loss:.4f}, P={avg_pearson_loss:.4f}, G={avg_g_loss:.4f}")
        
        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss and avg_pearson_loss < best_pearson_loss:
            best_test_loss = avg_test_loss
            best_pearson_loss = avg_pearson_loss
            epochs_no_improve = 0
            # Save the model checkpoint
            torch.save(model.state_dict(), checkpoint_path)
            print("\n\n-----------------------------------")
            print(f"Checkpoint saved for epoch {epoch+1}")
            print("-----------------------------------\n\n")
            # break # test break
        else:
            epochs_no_improve += 1

        # if epochs_no_improve == early_stopping_patience:
        #     print("\n\n*********************************")
        #     print(f"Early stopping triggered at epoch {epoch+1}")
        #     print("*********************************\n\n")
        #     break

        if epochs_no_improve == early_stopping_patience:
            
            if optimizer == optimizer_0:
                print("\n\n*********************************")
                print(f"ViT Starts training at epoch {epoch+1}")
                print("*********************************\n\n")
                optimizer = optimizer_1
                epochs_no_improve = 0
                model.load_state_dict(torch.load(checkpoint_path))
                fw_tuning_epochs = None
            else:
                print("\n\n*********************************")
                print(f"Early stopping triggered at epoch {epoch+1}")
                print("*********************************\n\n")
                break

    print("--- Training Complete ---\n")

    
def evaluate_model(model, data_loader, device, criterion, test_rdms, sample_timepoints, ms_start, ms_end, ms_step, window_size, optimizer):
    model.eval()
    total_loss = 0.0
    total_mse_loss = 0.0
    total_pearson_loss = 0.0
    total_g_loss = 0.0
    total_iterations = len(data_loader)
    progress_bar = tqdm(total=total_iterations, desc="Evaluating")
    
    with torch.no_grad():
        for batch_idx, (_, images, indices) in enumerate(data_loader):
            images = images.to(device)

            # optimizer.zero_grad()

            pred_emb_3d, pred_rdm_3d, _ = model(images)
        

            target_rdm_3d = test_rdms[:, np.ix_(indices, indices)[0], np.ix_(indices, indices)[1]]
            target_rdm_3d = torch.tensor(target_rdm_3d, dtype=torch.float).to(device)

            loss, mse_loss, pearson_loss, g_loss = criterion(pred_rdm_3d, target_rdm_3d)

            total_loss += loss.item()
            total_mse_loss += mse_loss.item()
            total_pearson_loss += pearson_loss.item()
            total_g_loss += g_loss.item()

            progress_bar.set_postfix({'Loss': loss.item(), 'M': mse_loss.item(), 'P': pearson_loss.item(), 'G': g_loss.item()})
            progress_bar.update(1)

    progress_bar.close()
    avg_loss = total_loss / total_iterations
    avg_mse_loss = total_mse_loss / total_iterations
    avg_pearson_loss = total_pearson_loss / total_iterations
    avg_g_loss = total_g_loss / total_iterations

    return avg_loss, avg_mse_loss, avg_pearson_loss, avg_g_loss


def compute_noise_level (alpha, threshold=0.75, scale = 0.1):

    noise_level = -1 * alpha + 1

    noise_level = gaussian_filter1d(noise_level, sigma=3)

    # set everything below the threshold to 0
    # noise_level[noise_level <= threshold] = threshold

    # min max it to 0-1
    noise_level = (noise_level - noise_level.min()) / (noise_level.max() - noise_level.min()) * scale

    return noise_level



if __name__ == '__main__':


    ##############################
    csv_file = "./Data/hebart66_embedding_rescaled_1806train.csv"
    img_dir = "./Data/Things1854"
    rdm_dir = "./Data/ThingsMEG_RDMs/ThingsMEG_RDM_4P.npy"
    n_dim = 66 # options: 49, 66
    ##############################
    backbone = 'ViT-L/14' #or options: RN50, ViT-B/32, ViT-B/16, ViT-L/14
    epochs = 50
    fw_tuning_epochs = None
    batch_size = 50
    train_portion = 1456/1806
    ms_start, ms_end, ms_step = -100, 1300, 5
    train_start, train_end, train_step, train_window_size = -100, 1300, 5, 15 # (start, end, step) (total 281 timepoints from -100ms to 1300ms every 5ms) #this is the time window for neural finetuning
    lr = 5e-5
    fw_lr= 1e-4
    stage_1_lr_multiplier = 100
    early_stopping_patience = 5
    checkpoint_path = './models/cliphba_dynamic_official_v9_step.pth'
    random_seed = 1
    vision_layers = 24
    transormer_layers = 1
    cuda = 0 #option, 0,1, -1
    pretrained_text_encoder = True
    text_encoder_path = "./models/partial/cliphba_dora66_text_encoder.pth"
    freeze_text = True
    p_weight = 1
    m_weight = 0.1
    g_weight = 0.15
    rank = 32
    noise_scale= 1
    sample_timepoints = list(range(train_start, train_end+1, train_step))
    ##############################

    # print each parameter line by line: batch size, learning rate, random seed, p_weight, m_weight, sample_timepoints
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Random seed: {random_seed}")
    print(f"Pearson Weight: {p_weight}")
    print(f"MSE Weight: {m_weight}")
    print(f"Generalization Weight: {g_weight}")
    print(f"Sample Timepoints: {len(sample_timepoints)}")

    print(f"\nModel will be saved at: {checkpoint_path}\n")


    
    # set random seed for the entire pipeline
    seed_everything(random_seed)

    classnames = classnames66
    classnames = [x[0] for x in classnames]

    dataset = DynamicDataset(csv_file=csv_file, img_dir=img_dir, rdm_dir=rdm_dir)
    rdms = dataset.rdms

    print(f"RDMs shape: {rdms.shape}")

    # define curves
    n_timepoints = len(sample_timepoints)
    zero_ms_position = (0 - ms_start) // ms_step + 1
    beta = compute_rdm_generalization(rdms, zero_ms_position)
    # beta[:zero_ms_position] = 0
    alpha = get_richness(rdms, zero_ms_position)
    # alpha[:zero_ms_position] = 0

    noise_level = compute_noise_level(beta, scale=noise_scale)

    # plot richness, noise_level, visual_scaler
    plt.plot(sample_timepoints, beta, label='beta')
    plt.plot(sample_timepoints, noise_level, label='Noise Level')
    plt.plot(sample_timepoints, alpha, label='Visual Scaler')
    plt.legend()
    plt.savefig('./curves/things_group_curves.png')

    # Split the dataset into training and testing
    train_size = int(train_portion * len(dataset))
    test_size = len(dataset) - train_size
    print(f"\nTrain size: {train_size}, Test size: {test_size}\n")
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if backbone == 'RN50':
        pos_embedding = False
    if backbone == 'ViT-B/16' or backbone == 'ViT-B/32' or backbone == 'ViT-L/14': 
        pos_embedding = True

    
    weighting_matrix = None
    
    # Initialize model
    model = CLIPHBA(classnames=classnames, weighting_matrix=weighting_matrix, backbone_name=backbone, pos_embedding=pos_embedding, ms_start=ms_start, ms_step=ms_step, ms_end=ms_end, train_start=train_start, train_step=train_step, train_end=train_end, train_window_size=train_window_size, beta=beta, noise_level=noise_level, visual_scaler=alpha)

    assert model.clip_model.weighting_matrix.shape[0] == model.clip_model.noise_level.shape[0] == model.clip_model.beta.shape[0] == model.clip_model.visual_scaler.shape[0] == len(sample_timepoints), f"Weighting matrix shape mismatch\n {model.clip_model.weighting_matrix.shape[0]} != {model.clip_model.noise_level.shape[0]} != {model.clip_model.beta.shape[0]} != {model.clip_model.visual_scaler.shape[0]} != {len(sample_timepoints)}"

    # Move the model to GPU
    if cuda == -1:
        device = torch.device("cuda")
        print(f"Using {torch.cuda.device_count()} GPUs")
    elif cuda == 0:
        device = torch.device("cuda:0")
        print("Using GPU 0")
    elif cuda == 1:
        device = torch.device("cuda:1")
        print("Using GPU 1")
    else:
        device = torch.device("cpu")
        print("Using CPU")


    # apply_lora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=16, lora_dropout=0.1, seed=random_seed)
    # switch_lora_layers(model, freeze_all=True, lora_state=True)
    # unfreeze_weighting_parameters(model)

    # apply_lora_to_ViT(model, n_vision_layers=0, n_transformer_layers=transormer_layers, r=16, lora_dropout=0.1, seed=random_seed)
    apply_dora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=rank, dora_dropout=0.1, seed=random_seed)
    switch_dora_layers(model, freeze_all=True, dora_state=True)
    unfreeze_weighting_parameters(model)



    if pretrained_text_encoder:
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
        
    if freeze_text:
        freeze_text_encoder(model)
        print("Model text encoder frozen\n")

    else:
        print("Model is not pretrained with text encoder\n")

    if cuda == -1:
       print(f"Using {torch.cuda.device_count()} GPUs")
       model = DataParallel(model)

    model.to(device)  # Move model to GPU if available

    # optimizer = AdamW(model.parameters(), lr=lr)
    # Define optimizer with separate learning rates
    optimizer_0 = AdamW([
        {'params': [p for n, p in model.named_parameters() if n != 'clip_model.weighting_matrix'], 'lr': 0},  # Default learning rate
        {'params': [model.clip_model.weighting_matrix], 'lr': fw_lr * stage_1_lr_multiplier}  # Higher learning rate for weighting_matrix
    ])

    optimizer_1 = AdamW([
        {'params': [p for n, p in model.named_parameters() if n != 'clip_model.weighting_matrix'], 'lr': lr},  # Default learning rate
        {'params': [model.clip_model.weighting_matrix], 'lr': fw_lr}  # Higher learning rate for weighting_matrix
    ])

    criterion = PearsonMSELongLoss(p_weight=p_weight, m_weight=m_weight, g_weight=g_weight)

    print("Updating layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}\n\n")

    sample_timepoints = list(range(train_start, train_end+1, train_step))


    # print(model)
    train_model(model, train_loader, test_loader, device, criterion, p_weight, m_weight, g_weight, optimizer_0, optimizer_1, epochs, fw_tuning_epochs, rdms, sample_timepoints, ms_start, ms_end, ms_step, train_window_size, early_stopping_patience, checkpoint_path)