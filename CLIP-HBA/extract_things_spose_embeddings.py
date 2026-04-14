"""Extract 66-dim SPoSE embeddings from the CLIP-HBA-Behavior backbone for THINGS images.

Builds the CLIPHBA backbone with DoRA weights from epoch97_dora_params.pth,
runs each THINGS image through the model's forward pass (which scores the image
against 66 SPoSE semantic text prompts), and saves results as a CSV with columns:
    image_name, 0, 1, 2, ..., 65

The THINGS image directory is expected to contain 1854 concept subfolders, each
holding one or more .jpg files:
    <img_dir>/aardvark/aardvark_01b.jpg
    <img_dir>/abacus/abacus_01b.jpg
    ...

Usage
-----
    python extract_things_spose_embeddings.py \
        --img_dir "Z:/multimodal_brain_inspired/THINGS_images" \
        --backbone_checkpoint ./Data/lamem/epoch97_dora_params.pth \
        --out_dir "Z:/multimodal_brain_inspired/marren/CLIP-HBA-Mem/CLIP-HBA/THINGS"
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
import torch
from functions.spose_dimensions import classnames66
from functions.train_behavior_things_pipeline import (
    CLIPHBA,
    apply_dora_to_ViT,
    seed_everything,
)
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


def discover_things_images(img_dir: str) -> list[str]:
    """Walk ``img_dir`` and return sorted relative paths (``concept/file.jpg``)."""
    img_dir = pathlib.Path(img_dir)
    rel_paths: list[str] = []
    for concept_dir in sorted(img_dir.iterdir()):
        if not concept_dir.is_dir():
            continue
        for img_path in sorted(concept_dir.glob("*.jpg")):
            rel_paths.append(f"{concept_dir.name}/{img_path.name}")
    return rel_paths


class ThingsImageDataset(Dataset):
    """Minimal dataset that loads THINGS images by relative path (no targets needed).

    Each ``image_rel_path`` is of the form ``concept/filename.jpg`` relative to
    ``img_dir``.  The returned ``image_name`` is just the filename (e.g.
    ``aardvark_01b.jpg``) for CSV output compatibility.
    """

    def __init__(self, image_rel_paths: list[str], img_dir: str) -> None:
        self.image_rel_paths = image_rel_paths
        self.img_dir = pathlib.Path(img_dir)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.52997664, 0.48070561, 0.41943838],
                    std=[0.27608301, 0.26593025, 0.28238822],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_rel_paths)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        rel_path = self.image_rel_paths[index]
        img_path = self.img_dir / rel_path
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Return just the filename (e.g. aardvark_01b.jpg) as the image name
        image_name = pathlib.Path(rel_path).name
        return image_name, image


def extract_spose_embeddings(
    image_rel_paths: list[str],
    img_dir: str,
    backbone_checkpoint: str,
    device: torch.device,
    batch_size: int = 64,
    # num_workers: int = 4,
    backbone_name: str = "ViT-L/14",
    vision_layers: int = 2,
    transformer_layers: int = 1,
    rank: int = 32,
) -> pd.DataFrame:
    """Extract 66-dim SPoSE embeddings for THINGS images.

    Args:
        image_rel_paths: Relative paths (``concept/filename.jpg``) within ``img_dir``.
        img_dir: Root directory containing concept subfolders with THINGS images.
        backbone_checkpoint: Path to the CLIP-HBA DoRA checkpoint.
        device: Torch device to run inference on.
        batch_size: Batch size for inference.
        num_workers: DataLoader worker count.
        backbone_name: CLIP backbone variant string.
        vision_layers: Number of DoRA-patched vision layers.
        transformer_layers: Number of DoRA-patched text layers.
        rank: DoRA rank.

    Returns:
        DataFrame with columns ``image_name, 0, 1, ..., 65``.
    """
    # Build backbone
    pos_embedding = backbone_name != "RN50"
    model = CLIPHBA(
        classnames=classnames66,
        backbone_name=backbone_name,
        pos_embedding=pos_embedding,
    )
    apply_dora_to_ViT(
        model,
        n_vision_layers=vision_layers,
        n_transformer_layers=transformer_layers,
        r=rank,
    )

    # Load DoRA checkpoint
    state_dict = torch.load(backbone_checkpoint, map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    print(f"[Backbone] Loaded {len(state_dict)} keys from {backbone_checkpoint}")

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # Build dataset and loader
    dataset = ThingsImageDataset(image_rel_paths, img_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=num_workers,
        # pin_memory=(device.type == 'cuda'),
    )

    all_names: list[str] = []
    all_embeddings: list[np.ndarray] = []

    with torch.no_grad(), tqdm(loader, desc="Extracting SPoSE embeddings") as pbar:
        for names, images in pbar:
            images = images.to(device)
            preds = model(images)  # [B, 66]
            all_names.extend(names)
            all_embeddings.append(preds.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 66]
    df = pd.DataFrame(embeddings, columns=[str(i) for i in range(66)])
    df.insert(0, "image_name", all_names)

    # Free GPU memory
    model.cpu()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 66-dim SPoSE embeddings from CLIP-HBA-Behavior for THINGS images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img_dir",
        default="Z:/multimodal_brain_inspired/THINGS_images",
        help="Root directory with 1854 concept subfolders, each containing .jpg images.",
    )
    parser.add_argument(
        "--backbone_checkpoint",
        default="./Data/lamem/epoch97_dora_params.pth",
        help="Path to the CLIP-HBA DoRA checkpoint.",
    )
    parser.add_argument(
        "--out_dir",
        default="Z:/multimodal_brain_inspired/marren/CLIP-HBA-Mem/CLIP-HBA/THINGS",
        help="Directory to save the output CSV.",
    )
    parser.add_argument(
        "--out_filename",
        default="things_spose_embeddings_66d.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for backbone inference.",
    )
    # parser.add_argument(
    #    '--num_workers', type=int, default=4,
    #    help='DataLoader worker count.',
    # )
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="GPU index. Use -1 for CPU.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed.",
    )
    parser.add_argument(
        "--vision_layers",
        type=int,
        default=2,
        help="Number of DoRA-patched vision layers.",
    )
    parser.add_argument(
        "--transformer_layers",
        type=int,
        default=1,
        help="Number of DoRA-patched text layers.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=32,
        help="DoRA rank.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cpu") if args.cuda == -1 else torch.device(f"cuda:{args.cuda}")

    # Discover all .jpg images under concept subfolders
    image_rel_paths = discover_things_images(args.img_dir)
    print(f"Discovered {len(image_rel_paths)} images across concept subfolders in {args.img_dir}")

    # Extract embeddings
    df = extract_spose_embeddings(
        image_rel_paths=image_rel_paths,
        img_dir=args.img_dir,
        backbone_checkpoint=args.backbone_checkpoint,
        device=device,
        batch_size=args.batch_size,
        # num_workers=args.num_workers,
        vision_layers=args.vision_layers,
        transformer_layers=args.transformer_layers,
        rank=args.rank,
    )

    # Save
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_filename
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows x {df.shape[1]} columns -> {out_path}")


if __name__ == "__main__":
    main()
