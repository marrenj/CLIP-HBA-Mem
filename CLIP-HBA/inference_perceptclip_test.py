"""Run PerceptCLIP inference on all 10 test splits of the combined LaMem+MemCat dataset.

Downloads the pre-trained PerceptCLIP checkpoint from HuggingFace
(PerceptCLIP/PerceptCLIP_Memorability) and scores each test split.

Usage:
    # All 10 folds (default)
    python inference_perceptclip_test.py

    # Specific folds
    python inference_perceptclip_test.py --folds 1 2 3

    # Custom data directory
    python inference_perceptclip_test.py --data_dir /path/to/Data

    # Use a local HuggingFace cache to avoid re-downloading
    python inference_perceptclip_test.py --hf_cache_dir /scratch/hf_cache
"""

import argparse
import datetime
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from functions.train_behavior_things_pipeline import seed_everything
from functions.train_mem_pipeline import PerceptCLIPDataset


_HF_REPO_ID = "PerceptCLIP/PerceptCLIP_Memorability"


def _download_checkpoint(cache_dir: str | None) -> tuple[str, str]:
    """Download modeling.py and pytorch_model.bin from HuggingFace.

    Returns:
        (modeling_path, ckpt_path) — local filesystem paths to each file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed. "
            "Run: pip install huggingface_hub"
        )

    kwargs = {"repo_id": _HF_REPO_ID}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    print(f"[HuggingFace] Downloading from {_HF_REPO_ID} …")
    modeling_path = hf_hub_download(filename="modeling.py", **kwargs)
    ckpt_path = hf_hub_download(filename="pytorch_model.bin", **kwargs)
    print(f"[HuggingFace] modeling.py  -> {modeling_path}")
    print(f"[HuggingFace] pytorch_model.bin -> {ckpt_path}")
    return modeling_path, ckpt_path


def _load_model(modeling_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate clip_lora_model from HuggingFace's modeling.py and load weights.

    Uses importlib so the architecture is guaranteed to match the released checkpoint.
    """
    spec = importlib.util.spec_from_file_location("perceptclip_modeling", modeling_path)
    modeling_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling_mod)

    print("[Model] Initialising clip_lora_model …")
    model: torch.nn.Module = modeling_mod.clip_lora_model()

    state_dict = torch.load(ckpt_path, map_location="cpu")
    # Strip DataParallel 'module.' prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Model] Warning — missing keys ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"[Model] Warning — unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    print(f"[Model] Checkpoint loaded: {ckpt_path}")

    model.to(device)
    model.eval()
    return model


def run_inference(config: dict) -> None:
    seed_everything(config["seed"])

    device = torch.device(config["device"])
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(config["output_dir"], timestamp)
    os.makedirs(out_dir, exist_ok=True)

    modeling_path, ckpt_path = _download_checkpoint(config["hf_cache_dir"])
    model = _load_model(modeling_path, ckpt_path, device)

    data_dir = config["data_dir"]
    img_root = {
        "lamem": os.path.join(data_dir, "lamem", "images"),
        "memcat": os.path.join(data_dir, "memcat", "images"),
    }
    memcat_meta_csv = os.path.join(data_dir, "memcat", "memcat_image_data.csv")

    summary_rows: list[dict] = []

    for fold in config["folds"]:
        csv_path = os.path.join(
            data_dir,
            "combined_lamem_memcat",
            f"lamem_memcat_test_split_{fold:02d}.csv",
        )
        if not os.path.exists(csv_path):
            print(f"[Fold {fold}] Test CSV not found, skipping: {csv_path}")
            continue

        ds = PerceptCLIPDataset(
            csv_file=csv_path,
            img_root=img_root,
            memcat_meta_csv=memcat_meta_csv,
        )
        print(f"\n[Fold {fold:02d}] {len(ds):,} images — {csv_path}")

        loader = DataLoader(
            ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=(device.type == "cuda"),
        )

        all_paths: list[str] = []
        all_preds: list[float] = []
        all_targets: list[float] = []

        with torch.no_grad():
            for image_paths, images, targets in tqdm(loader, desc=f"Fold {fold:02d}"):
                images = images.to(device)
                preds = model(images).squeeze(1).cpu().numpy()
                all_paths.extend(image_paths)
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.numpy().tolist())

        preds_arr = np.array(all_preds)
        targets_arr = np.array(all_targets)
        rho, p_val = spearmanr(preds_arr, targets_arr)
        mse = float(np.mean((preds_arr - targets_arr) ** 2))

        print(f"[Fold {fold:02d}] Spearman ρ = {rho:.4f} (p={p_val:.2e})  |  MSE = {mse:.6f}")
        print(
            f"[Fold {fold:02d}] Pred range: [{preds_arr.min():.4f}, {preds_arr.max():.4f}]  "
            f"std: {preds_arr.std():.4f}"
        )

        results_df = pd.DataFrame(
            {"image_path": all_paths, "pred_score": preds_arr, "true_score": targets_arr}
        )
        pred_csv = os.path.join(out_dir, f"perceptclip_test_fold{fold:02d}_predictions.csv")
        results_df.to_csv(pred_csv, index=False)
        print(f"[Fold {fold:02d}] Saved predictions -> {pred_csv}")

        summary_rows.append(
            {
                "fold": fold,
                "n_images": len(ds),
                "spearman_rho": rho,
                "spearman_p": p_val,
                "mse": mse,
                "pred_std": float(preds_arr.std()),
                "pred_min": float(preds_arr.min()),
                "pred_max": float(preds_arr.max()),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "perceptclip_test_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[Summary] Saved -> {summary_path}")
    print(summary_df.to_string(index=False))

    if len(summary_rows) > 1:
        mean_rho = summary_df["spearman_rho"].mean()
        std_rho = summary_df["spearman_rho"].std()
        mean_mse = summary_df["mse"].mean()
        print(f"\n[Summary] Mean Spearman ρ = {mean_rho:.4f} ± {std_rho:.4f}  |  Mean MSE = {mean_mse:.6f}")

    print("\nDone.")


def main() -> None:
    _data_dir_default = os.environ.get("DATA_DIR", "./Data")

    parser = argparse.ArgumentParser(
        description="Run pre-trained PerceptCLIP inference on combined LaMem+MemCat test splits."
    )
    parser.add_argument(
        "--data_dir",
        default=_data_dir_default,
        help="Root data directory containing lamem/, memcat/, and combined_lamem_memcat/. "
             "Also settable via DATA_DIR env var.",
    )
    parser.add_argument(
        "--output_dir",
        default="./preds/perceptclip_test/",
        help="Directory to save per-fold prediction CSVs and the summary CSV.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        default=None,
        help="HuggingFace cache directory for downloaded model files. "
             "Defaults to ~/.cache/huggingface/hub.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=list(range(1, 11)),
        help="Fold numbers to run inference on (default: 1 2 … 10).",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string (e.g. cuda:0, cuda:1, cpu).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader worker processes.",
    )
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "hf_cache_dir": args.hf_cache_dir,
        "folds": sorted(args.folds),
        "batch_size": args.batch_size,
        "device": args.device,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }

    run_inference(config)


if __name__ == "__main__":
    main()
