from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, return_token=False, pos_embedding=False):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), size=(x.size(0), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        if return_token:
            return x[0], x[1:]
        else:
            return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_token=False, pos_embedding=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if return_token:
            x, tokens = self.attnpool(x, return_token, pos_embedding)
            return x, tokens
        else:
            x = self.attnpool(x, return_token, pos_embedding)
            return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, return_token=False, pos_embedding=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        if pos_embedding:
            positional_embedding_resize = F.interpolate(self.positional_embedding.unsqueeze(0).unsqueeze(0), size=(x.size(1), x.size(2)), mode='bicubic').squeeze(0).squeeze(0)
            x = x + positional_embedding_resize.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        layer_outputs = []
        for layer in self.transformer.resblocks:
            x = layer(x)
            layer_outputs.append(x.permute(1, 0, 2))  # store layer output as NLD
        
        x = self.ln_post(layer_outputs[-1][:, 0, :])
        
        if self.proj is not None:
            x = x @ self.proj
        return x, layer_outputs



class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                    ms_start: int,
                    ms_end: int,
                    ms_step: int,
                    train_start: int,
                    train_end: int,
                    train_window_size: int,
                    n_dim: int,
                    weighting_matrix: np.ndarray,
                    richness: list, 
                    dynamic_sigma: list
                 ):
        super().__init__()

        self.context_length = context_length


        self.n_dim = n_dim
        self.ms_start = ms_start
        self.ms_end = ms_end
        self.ms_step = ms_step
        self.train_start = train_start
        self.train_end = train_end
        self.train_window_size = train_window_size
        self.weighting_matrix = weighting_matrix
        self.richness = richness
        self.dynamic_sigma = dynamic_sigma

        self.n_timepoints = (self.ms_end - self.ms_start) // self.ms_step + 1
        self.n_train_timepoints = (self.train_end - self.train_start) // self.ms_step + 1

        # print(f"ms_start: {self.ms_start}ms, ms_end: {self.ms_end}ms, ms_step: {self.ms_step}ms, \ntrain_start: {self.train_start}ms, train_end: {self.train_end}ms, train_window_size: +/-{self.train_window_size}ms")
        # print(f"\nn_timepoints: {self.n_timepoints}, n_train_timepoints: {self.n_train_timepoints}\n")

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, pos_embedding):
        return self.visual(image.type(self.dtype), pos_embedding=pos_embedding)


    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, tokenized_prompts, time_input, pos_embedding=False):
        time_input = time_input.to(image.device)

        # Encode the image features
        image_features = self.encode_image(image, pos_embedding)

        # Ensure tokenized_prompts is a tensor and move it to the same device as the image
        tokenized_prompts = tokenized_prompts.to(image.device)

        # Encode each text prompt individually and stack them into a single tensor
        text_features = [self.encode_text(prompt) for prompt in tokenized_prompts]
        text_features = torch.stack(text_features)

        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = self.time_distortion(image_features, time_input) # Apply time distortion to image features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        

        # compute the cosine similarity between the image and text features
        logit_scale = self.logit_scale.exp()

        # Stack text_features along the batch dimension
        text_features_stacked = text_features.view(-1, text_features.size(-1))

        # Compute logits per image for each text feature
        logits = logit_scale * (image_features @ text_features_stacked.t())

        # Reshape logits to separate the pairs
        logits = logits.view(image_features.size(0), text_features.size(0), 2)

        # Apply softmax to the last dimension
        probs = logits.softmax(dim=-1)

        # Extract the first probability from each pair
        pred_emb = probs[:, :, 0]
        
        return pred_emb
    


def gaussian_kernel(n, sigma=3, device=None):
    # Create a Gaussian kernel
    gauss_kernel = torch.exp(-0.5 * ((torch.arange(n, dtype=torch.float32, device=device) - (n - 1) / 2) ** 2) / (sigma ** 2))
    gauss_kernel = gauss_kernel / gauss_kernel.sum()
    return gauss_kernel



class ModifiedCLIP(CLIP):
    def __init__(self, *args, **kwargs):
        super(ModifiedCLIP, self).__init__(*args, **kwargs)
        # self.time_scalar_layer = nn.Parameter(torch.zeros(self.n_timepoints))
        self.richness = nn.Parameter(torch.tensor(self.richness, dtype=torch.float32))
        self.dynamic_sigma = nn.Parameter(torch.tensor(self.dynamic_sigma, dtype=torch.float32))
        self.weighting_matrix = nn.Parameter(torch.tensor(self.weighting_matrix, dtype=torch.float32))


    def forward(self, image, tokenized_prompts, time_input, pos_embedding=False):
        assert self.richness.shape[0] == self.n_timepoints, f"Richness scaler shape: {self.richness.shape[0]}, n_timepoints: {self.n_timepoints}"
        
        time_input = time_input.to(image.device)
        batch_size = image.shape[0]

        _, layer_outputs = self.encode_image(image, pos_embedding)


        # Select the weights for the given timepoints
        time_index = float((time_input[0].item()) - self.ms_start) // self.ms_step
        start_index = max(0, int(time_index - self.train_window_size//self.ms_step))
        end_index = min(int(time_index + self.train_window_size//self.ms_step + 1), self.n_timepoints)

        # Gaussian Kernel
        n = end_index - start_index
        sigma = self.dynamic_sigma[int(time_index)]

        gauss_kernel = gaussian_kernel(n, sigma, device=image.device)

        # print(np.round(gauss_kernel, 3))

        # use gaussian kernel to find the richness scaler and apply to the final embedding
        richness_scaler = (self.richness[start_index:end_index] * gauss_kernel).sum()

        # weights = self.weighting_matrix[start_index:end_index].mean(dim=0)
        weights = (self.weighting_matrix[start_index:end_index] * gauss_kernel.unsqueeze(1)).sum(dim=0)
        weights = weights.clamp(min=0)
        weights = weights.to(image.device).expand(batch_size, -1)



        # Compute the weighted average of the layer outputs
        image_features = torch.zeros_like(layer_outputs[0][:, 0, :])
        for i in range(len(layer_outputs)):
            image_features += layer_outputs[i][:, 0, :] * weights[:, i].unsqueeze(1)

        # Apply final LayerNorm and projection to match the expected input dimensions
        image_features = image_features @ self.visual.proj

        tokenized_prompts = tokenized_prompts.to(image.device)
        text_features = [self.encode_text(prompt) for prompt in tokenized_prompts]
        text_features = torch.stack(text_features)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        text_features_stacked = text_features.view(-1, text_features.size(-1))


        # merging modalities here
        logits = logit_scale * (image_features @ text_features_stacked.t())
        logits = logits.view(image_features.size(0), text_features.size(0), 2)
        
        probs = logits.softmax(dim=-1) #softmax function normalizes the cross-modal similarity into (p, 1-p)

        pred_emb = probs[:, :, 0]

        pred_emb = pred_emb * richness_scaler

        return pred_emb


def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

    

# Initialize your model with the modified CLIP
def build_model(state_dict: dict, ms_start=-100, ms_step=5, ms_end=1300, train_start=100, train_end=800, train_window_size=30, richness=None, dynamic_sigma=None, weighting_matrix=None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    n_dim = 66

    if richness is None:
        richness = torch.ones((ms_end - ms_start) // ms_step + 1)  # Default to ones if richness is not provided
    
    if dynamic_sigma is None:
        dynamic_sigma = torch.ones((ms_end - ms_start) // ms_step + 1) # Default to ones if dynamic_sigma is not provided

    assert weighting_matrix is not None, "Weighting matrix must be provided"

    model = ModifiedCLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        ms_start=ms_start,
        ms_end=ms_end,
        ms_step=ms_step,
        train_start=train_start,
        train_end=train_end,
        train_window_size=train_window_size,
        n_dim=n_dim,
        weighting_matrix = weighting_matrix,
        richness=richness,
        dynamic_sigma=dynamic_sigma
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    model.load_state_dict(state_dict, strict=False)

    return model
