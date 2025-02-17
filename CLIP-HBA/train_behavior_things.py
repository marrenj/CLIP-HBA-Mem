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
sys.path.append('../')
from src.models.backbones.sr_backbones.rrdb_net import RRDB
from src.models.builder import build_component
from src.models.common import PixelShufflePack, make_layer
from src.models.registry import BACKBONES
from src.utils import get_root_logger
from src.models.components.clip_hba_no_softmax import clip
from src.models.components.clip_hba_no_softmax.simple_tokenizer import SimpleTokenizer as _Tokenizer

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
    def __init__(self, csv_file, img_dir):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.52997664, 0.48070561, 0.41943838],
                                            std=[0.27608301, 0.26593025, 0.28238822])
                    ])

        # Load and filter annotations based on the 'set' column
        self.annotations = pd.read_csv(csv_file, index_col=0)

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





if __name__ == '__main__':

    '''
    configure training parameters here
    '''


    ##############################
    n_dim = 66 # options: 49, 66
    csv_file = f'./Data/hebart{n_dim}_embedding_rescaled100_1806train.csv' # image target embedding with image names and indices
    img_dir = './Data/Things1854' # image input directory
    ##############################
    backbone = 'ViT-L/14' #or options: RN50, ViT-B/32, ViT-B/16, ViT-L/14
    epochs = 500
    batch_size = 64
    train_portion = 0.8
    lr = 3e-4
    scheduler = False
    early_stopping_patience = 20
    checkpoint_path = f'./models/cliphba_dora{n_dim}_test.pth'
    random_seed = 1
    vision_layers = 2
    transormer_layers = 1
    rank = 32
    criterion = nn.MSELoss() # nn.MSELoss()
    cuda = 0 # -1 for all GPUs, 0 for GPU 0, 1 for GPU 1, 2 for CPU
    ##############################





    

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
    



    print(f"\n\nBackbone: {backbone} \nTotal Epochs: {epochs} \nBatch Size: {batch_size} \nTrain Portion: {train_portion} \nLearning Rate: {lr} \nScheduler: {scheduler} \nEarly Stopping Patience: {early_stopping_patience} \nModel checkpoint Path: {checkpoint_path}\nRandomSeed: {random_seed} \nvision_layers: {vision_layers} \ntransformer_layers: {transormer_layers}\n\n")
    
    dataset = ThingsDataset(csv_file=csv_file, img_dir=img_dir)

    # Split the dataset into training and testing
    train_size = int(train_portion * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

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

    # unfreeze_image_layers_all(model)

    # Set up optimizer to only update parameters where requires_grad is True
    optimizer = AdamW(model.parameters(), lr=lr)


    if scheduler == True:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6)
    else: 
        scheduler = None



    print("Updating layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    print(f"Number of trainable parameters: {count_trainable_parameters(model)}\n\n")

    print(model)
    # train_model(model, train_loader, test_loader, device, optimizer, criterion, epochs, early_stopping_patience, checkpoint_path, scheduler=scheduler)