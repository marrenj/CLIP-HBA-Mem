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
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from sklearn.model_selection import KFold
from torch.utils.data import Subset
import random
import math

import sys
sys.path.append('../../')
from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from mmedit.models.components.clip_hba_no_softmax import clip
from mmedit.models.components.clip_hba_no_softmax.simple_tokenizer import SimpleTokenizer as _Tokenizer

import shutil
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

import matplotlib.pyplot as plt



# from mmedit.models.components.clip_baseline import clip
# from mmedit.models.components.clip_baseline.simple_tokenizer import SimpleTokenizer as _Tokenizer


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Prompt pairs for the model
classnames49 = [
    ["made of metal / artificial / hard", "made of nonmetal / natural / soft"],
    ["food-related /eating-related / kitchen-related", "non-food-related / non-eating-related / non-kitchen-related"],
    ["animal-related / organic", "non-animal-related / artificial / inorganic"],
    ["clothing-related / fabric / covering", "non-clothing-related / non-fabric"],
    ["furniture-related / household-related / artifact", "non-furniture-related / non-household-related / outdoors-related"],
    ["plant-related / green", "non-plant-related / non-green"],
    ["outdoors-related", "indoors-related"],
    ["transportation / motorized / dynamic", "non-transportation / non-motorized / slow"],
    ["wood-related / brown", "non-wood-related / non-brown"],
    ["body part-related", "non-body-part-related"],
    ["colorful", "monotone"],
    ["valuable special occasion-related", "common / generic / non-valuable"],
    ["electronic / technology", "non-electronic / nature"],
    ["sport-related / recreation-related", "non-sport-related / non-recreation-related"],
    ["disc-shaped / round", "angular / confusing shapes / protruding"],
    ["tool-related", "non-tool-related"],
    ["many small things / coarse pattern", "smooth pattern / singular object"],
    ["paper-related / thin / flat / text-related", "non-paper-related / wide / textured / non-text-related"],
    ["fluid-related / drink-related", "solid-related / non-fluid-related / non-drink-related"],
    ["long / thin", "short / wide"],
    ["water-related / blue", "dry / non-blue / non-water-related"],
    ["powdery / fine-scale pattern", "singular object / smooth / large pieces"],
    ["red", "non-red"],
    ["feminine (stereotypically) / decorative", "masculine (stereotypically) / minimalist"],
    ["bathroom-related / sanitary", "non-bathroom-related / unsanitary"],
    ["black / noble", "unrefined / lowly"],
    ["weapon / danger-related / violence", "calming / nurturing / safety-related"],
    ["musical instrument-related / noise-related", "quiet / silence / non-musical-instrument-related"],
    ["sky-related / flying-related / floating-related", "non-sky-related / ground-related / non-flying-related"],
    ["spherical / ellipsoid / rounded / voluminous", "angular / spiky / sharp"],
    ["repetitive", "varied / disorganized"],
    ["flat / patterned", "bumpy / rough / disorganized"],
    ["white", "black"],
    ["thin / flat", "wide / rough / bumpy"],
    ["disgusting / bugs", "cute / pleasant / non-bugs"],
    ["string-related", "non-string-related"],
    ["arms/legs/skin-related", "non-skin-related"],
    ["shiny / tranparent", "muted / opaque"],
    ["construction-related / physical work-related", "non-physical-work-related"],
    ["fire-related / heat-related", "non-fire-related / cold-related"],
    ["head-related / face-related", "non-head-related / non-face-related"],
    ["beams-related", "non-beams-related"],
    ["eating-related / put things on top", "non-eating-related"],
    ["container-related / hollow", "solid / full / non-container-related"],
    ["child-related / toy-related", "non-child-related / mature"],
    ["medicine-related", "non-medicine-related"],
    ["has grating", "no grating"],
    ["handicraft-related", "non-handicraft-related"],
    ["cylindrical / conical", "triangular / angular / non-cylindrical / non-conical"]
]

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


class ThingsDataset(Dataset):
    def __init__(self, csv_file, img_dir, train_size, seed=1):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load and filter annotations based on the 'set' column
        self.annotations = pd.read_csv(csv_file, index_col=0)
        # randomly sample train_size number of rows from the annotations
        self.annotations = self.annotations.sample(n=train_size, random_state=seed)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_name = self.annotations.iloc[index, 0]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)


        targets = torch.tensor(self.annotations.iloc[index, 1:].values.astype('float32'))
        
        return image_name, image, targets

def load_clip_to_cpu(backbone_name):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class CLIPHBA(nn.Module):
    def __init__(self, classnames, backbone_name='RN50', pos_embedding=False):
        super().__init__()

        self.num_clip = len(classnames)
        self.clip_model = load_clip_to_cpu(backbone_name)
        self.clip_model.float()
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
        pred_score = self.clip_model(image, tokenized_prompts, self.pos_embedding)

        pred_score = pred_score.float()  # Adjust the dimensions accordingly

        # print(f"pred_score: {pred_score}")

        return pred_score



class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=8, lora_alpha=16, lora_dropout=0.1):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        self.lora_A = nn.Parameter(torch.randn(self.r, original_layer.out_features))
        self.lora_B = nn.Parameter(torch.zeros(original_layer.in_features, self.r))
        self.scaling = self.lora_alpha / self.r

        self.reset_parameters()

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

def apply_lora_to_ViT(model, n_vision_layers=1, n_transformer_layers=1, r=8, lora_dropout=0.1):
    """
    Applies LoRA to the 'out_proj' of the 11th and the last (23rd) ResidualAttentionBlock in the
    VisionTransformer's transformer.

    :param model: The PyTorch model to modify.
    :param r: The rank of the LoRA approximation.
    :param lora_dropout: The dropout rate for LoRA layers.
    """
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
        lora_layer = LoRALayer(target_layer, r=r, lora_dropout=lora_dropout)
        target_block.attn.out_proj = lora_layer

    block_indices = -n_transformer_layers
    for idx in range(block_indices, 0):
        # Access the specific ResidualAttentionBlock
        target_block = model_module.clip_model.transformer.resblocks[idx]
        # Access the 'out_proj' within the MultiheadAttention of the block
        target_layer = target_block.attn.out_proj
        # Replace the original layer with a LoRALayer
        lora_layer = LoRALayer(target_layer, r=r, lora_dropout=lora_dropout)
        target_block.attn.out_proj = lora_layer



def unfreeze_lora_layers(model, freeze_all=True):
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
                    child.lora_A.requires_grad = True
                    child.lora_B.requires_grad = True
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



def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # return sum(p.numel() for p in model.parameters())


def unfreeze_image_layers(model):
    # Handle DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Unfreezing the last layer of the image encoder
        
    for param in model_module.clip_model.visual.layer3.parameters():
        param.requires_grad = True

    for param in model_module.clip_model.visual.layer4.parameters():
        param.requires_grad = True

    for param in model_module.clip_model.visual.attnpool.parameters():
        param.requires_grad = True


def unfreeze_image_layers_all(model):
    # Handle DataParallel wrapper
    if isinstance(model, torch.nn.DataParallel):
        model_module = model.module
    else:
        model_module = model

    # Unfreezing the last layer of the image encoder
        
    for param in model_module.clip_model.visual.parameters():
        param.requires_grad = True




def evaluate_model(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0

    # Wrap data_loader with tqdm for a progress bar
    with torch.no_grad(), tqdm(enumerate(data_loader), total=len(data_loader), desc="Evaluating") as progress_bar:
        for batch_idx, (_, images, targets) in progress_bar:
            images = images.to(device)
            targets = targets.to(device)

            predictions = model(images)

            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            total_loss += loss.item() * images.size(0) 

    avg_loss = total_loss / len(data_loader.dataset)
    return avg_loss


def train_model(model, train_loader, test_loader, device, optimizer, criterion, epochs, early_stopping_patience=5, checkpoint_path='clip_hba_model_cv.pth', scheduler=None):
    model.train()
    best_test_loss = float('inf')
    epochs_no_improve = 0
    loss_data = []  # To store loss data for plotting
    if scheduler is not None:
        print("Using scheduler")

    # initial evaluation
    print("*********************************")
    print("Evaluating initial model")
    best_test_loss = evaluate_model(model, test_loader, device, criterion)
    print(f"Initial Validation Loss: {best_test_loss:.4f}")
    print("*********************************\n")


    for epoch in range(epochs):
        total_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (_, images, targets) in progress_bar:

            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            
            loss = criterion(predictions, targets)
            progress_bar.set_postfix({'loss': loss.item()})
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * images.size(0)
        avg_train_loss = total_loss / len(train_loader.dataset)

        # Evaluate after every epoch
        avg_test_loss = evaluate_model(model, test_loader, device, criterion)
        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_test_loss:.4f}")

        if scheduler is not None:
            scheduler.step(avg_test_loss)

        loss_data.append({'epoch': epoch + 1, 'loss': avg_train_loss, 'type': 'Train'})
        loss_data.append({'epoch': epoch + 1, 'loss': avg_test_loss, 'type': 'Test'})

        # Check for early stopping and saving checkpoint
        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            epochs_no_improve = 0
            # Save the model checkpoint
            torch.save(model.state_dict(),checkpoint_path)
            print("\n\n-----------------------------------")
            print(f"Checkpoint saved for epoch {epoch+1}")
            print("-----------------------------------\n\n")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == early_stopping_patience:
            print("\n\n*********************************")
            print(f"Early stopping triggered at epoch {epoch+1}")
            print("*********************************\n\n")
            break


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
    
def run_image(model, data_loader, embedding_save_folder, device=torch.device("cuda:0")):
    model.eval()
    model.to(device)
    image_names = []
    predictions = []
    
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Processing images")
    
    with torch.no_grad():
        for batch_idx, (batch_image_names, batch_images) in progress_bar:
            batch_images = batch_images.to(device)
            batch_outputs = model(batch_images)
            
            predictions.extend(batch_outputs.cpu().numpy())
            image_names.extend(batch_image_names)

        # min max to 0-1
        predictions = np.array(predictions)
        # predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min())

        hba_embedding = pd.DataFrame(predictions)
        hba_embedding['image'] = image_names
        hba_embedding = hba_embedding[['image'] + [col for col in hba_embedding if col != 'image']]
        emb_save_path = f"{embedding_save_folder}/static_embedding.csv"
        hba_embedding.to_csv(emb_save_path, index=False)
        print(f"Embedding saved to {emb_save_path}")

    return hba_embedding

def upper_triangular_spearmanr(rdm1, rdm2):

    # Extract the upper triangular part of the RDMs
    rdm1 = rdm1[np.triu_indices(rdm1.shape[0], k=1)]
    rdm2 = rdm2[np.triu_indices(rdm2.shape[0], k=1)]

    # Calculate the Spearman correlation
    correlation, p_value = spearmanr(rdm1, rdm2)

    return correlation, p_value




if __name__ == '__main__':


    # create a dict: keys: train_size, values: spearmanr, p
    spearmanr_dict = {}

    train_size_list = list(range(100, 1800, 100))
    train_size_list = [10, 25, 50, 75] + train_size_list
    n_dim = 66 # options: 49, 66
    csv_file = f'../Data/hebart{n_dim}_embedding_rescaled100_1806train.csv'
    img_dir = '../Data/Things1854'
    backbone = 'ViT-L/14' #or options: RN50, ViT-B/32, ViT-B/16, ViT-L/14
    epochs = 500
    # batch_size = 64
    train_portion = 0.8
    lr = 3e-4
    scheduler = False
    early_stopping_patience = 15
    random_seed = 1
    vision_layers = 2
    transormer_layers = 1
    rank = 32
    criterion = nn.MSELoss() # nn.MSELoss()
    cuda = 0

    seed_everything(random_seed)
    
    if n_dim == 49:
        classnames = classnames49
        if len(pd.read_csv(csv_file).columns) != n_dim+2:
            raise ValueError("CSV file must have 50 columns for 49 dimensions")
        else:
            print("\nUsing 49 dimensions")
    elif n_dim == 66:
        classnames = classnames66
        if len(pd.read_csv(csv_file).columns) != n_dim+2:
            raise ValueError("CSV file must have 67 columns for 66 dimensions")
        else:
            print("\nUsing 66 dimensions")
    else: 
        raise ValueError("n_dim must be 49 or 66")
    

    # choose only the first element of each class names
    classnames = [x[0] for x in classnames]
    # print(f"Classnames: {classnames}")

    for train_size in train_size_list:

        print(f"Training size: {train_size}")
        # print current ./ path
        print(os.getcwd())

        checkpoint_path = f'../models/size_search/behavioral/cliphba_behavior_size{train_size}.pth'

        # using the img_dir and the train_size, randomly sample train_size number of images into the temp_sample_folder


        dataset = ThingsDataset(csv_file=csv_file, img_dir=img_dir, train_size=train_size, seed=random_seed)

        # Split the dataset into training and testing
        train_split = int(train_portion * len(dataset))
        test_split = len(dataset) - train_split
        train_dataset, test_dataset = random_split(dataset, [train_split, test_split])

        batch_size = min(train_split//4, 100)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        if backbone == 'RN50':
            pos_embedding = False
            print("pos_embedding is False")
        if backbone == 'ViT-B/16' or backbone == 'ViT-B/32' or backbone == 'ViT-L/14': 
            pos_embedding = True
            print("pos_embedding is True")

        # Initialize your model
        model = CLIPHBA(classnames=classnames, backbone_name=backbone, pos_embedding=pos_embedding)


        # Move the model to GPU
        if cuda == -1:
            device = torch.device("cuda")
        elif cuda == 0:
            device = torch.device("cuda:0")
        elif cuda == 1:
            device = torch.device("cuda:1")
        else:
            device = torch.device("cpu")


        # apply_lora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=64, lora_dropout=0.1)
        # unfreeze_lora_layers(model, freeze_all=True)

        apply_dora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=rank, dora_dropout=0.1)
        switch_dora_layers(model, freeze_all=True, dora_state=True)

        # Use DataParallel to utilize multiple GPUs
        if cuda == -1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = DataParallel(model)

        model.to(device)  # Move model to GPU if available


        # Set up optimizer to only update parameters where requires_grad is True
        optimizer = AdamW(model.parameters(), lr=lr)


        if scheduler == True:
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
        else: 
            scheduler = None

        train_model(model, train_loader, test_loader, device, optimizer, criterion, epochs, early_stopping_patience, checkpoint_path, scheduler=scheduler)



############################################################################################################
        batch_size = 256
        load_hba = True
        model_path = checkpoint_path
        save_folder = f'../../output/size_search/cliphba_behavior_size{train_size}/things1854/'

        # Create the directory if it doesn't exist
        embedding_save_folder = f"{save_folder}"
        if os.path.exists(embedding_save_folder):
            shutil.rmtree(embedding_save_folder)
        os.makedirs(embedding_save_folder)

        model = CLIPHBA(classnames=classnames, backbone_name=backbone, pos_embedding=pos_embedding)

        if load_hba:
            apply_dora_to_ViT(model, n_vision_layers=vision_layers, n_transformer_layers=transormer_layers, r=rank, dora_dropout=0.1)
            model_state_dict = torch.load(model_path)
            adjusted_state_dict = {key.replace("module.", ""): value for key, value in model_state_dict.items()}
            model.load_state_dict(adjusted_state_dict)
            # print(f"image weight: {model_state_dict['clip_model.contribution_weight']}, text weight: {2 - model_state_dict['clip_model.contribution_weight']}")
            # print(f"logit scale: {model_state_dict['clip_model.logit_scale'].exp()}")
        else:
            print(f"Using Original CLIP {backbone}")

        device = torch.device(cuda)
        
        # Load the dataset
        dataset = ImageDataset(img_dir)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Run the model and save output embeddings
        # print(model)
        hba_emb = run_image(model, data_loader, embedding_save_folder, device=device)
############################################################################################################
        triplet48_rdm = np.array(loadmat("../Data/RDM48_triplet.mat")['RDM48_triplet'])

        hba_emb = hba_emb.iloc[:, 1:].values
        hba_rdm = 1 - cosine_similarity(hba_emb)
        index_path = "../Data/triplet48_index.npy"
        sample_index = np.load(index_path)
        hba48_rdm = hba_rdm[sample_index][:, sample_index]

        correlation, p_value = upper_triangular_spearmanr(triplet48_rdm, hba48_rdm)
        spearmanr_dict[train_size] = (correlation, p_value)
        print(f"Correlation: {correlation}, p-value: {p_value}")
        print("******************************************")
        print("\n\n\n\n")

        # save the spearmanr_dict to a csv file. columns: train_size, spearmanr, p_value
        spearmanr_df = pd.DataFrame.from_dict(spearmanr_dict, orient='index', columns=['spearmanr', 'p_value'])
        spearmanr_df.index.name = 'train_size'
        spearmanr_df.to_csv(f'./cliphba_behavioral_sizes.csv')

        # reinitialize the canvas and plot the latest results from spearmaanr_dict if there are more than 1 results
        if len(spearmanr_df) > 1:
            plt.close()
            fig, ax = plt.subplots()
            ax.plot(spearmanr_df.index, spearmanr_df['spearmanr'], marker='o')
            ax.set_xlabel('Training Size')
            ax.set_ylabel('Spearman r')
            ax.set_title('Spearman r between HBA and Triplet48 RDMs')
            plt.savefig(f'./cliphba_behavioral_sizes.png')


    
        