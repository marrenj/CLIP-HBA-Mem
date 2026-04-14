"""Unit tests for CLIP-HBA-Mem training pipeline components.

All tests run on CPU with synthetic data — no real images, checkpoints,
or CLIP weights are required.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# conftest.py adds CLIP-HBA/ to sys.path
from functions.train_behavior_things_pipeline import seed_everything
from functions.train_mem_pipeline import (
    EmbeddingDataset,
    MLPOnlyHead,
    evaluate_mem_model,
)


# ---------------------------------------------------------------------------
# MLPOnlyHead
# ---------------------------------------------------------------------------

class TestMLPOnlyHead:
    def test_forward_output_shape(self) -> None:
        """Output tensor must be [B, 1] for any batch size."""
        model = MLPOnlyHead(hidden_dims=(64, 32), dropout_rate=0.0, input_dim=128)
        model.eval()
        x = torch.randn(8, 128)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 1), f"Expected (8, 1), got {out.shape}"

    def test_forward_single_sample(self) -> None:
        """Forward pass on a single sample should not raise."""
        model = MLPOnlyHead(hidden_dims=(32,), dropout_rate=0.0, input_dim=64)
        model.eval()
        x = torch.randn(1, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_all_params_trainable(self) -> None:
        """All MLPOnlyHead parameters should require gradients (it has no frozen backbone)."""
        model = MLPOnlyHead(hidden_dims=(64, 32), dropout_rate=0.0)
        for name, param in model.named_parameters():
            assert param.requires_grad, f"Parameter {name} unexpectedly frozen"

    def test_invalid_activation_raises(self) -> None:
        with pytest.raises(ValueError, match="activation must be one of"):
            MLPOnlyHead(activation="sigmoid")


# ---------------------------------------------------------------------------
# EmbeddingDataset
# ---------------------------------------------------------------------------

class TestEmbeddingDataset:
    def _make_pt_file(self, tmp_path: Path, n: int = 20, dim: int = 768) -> Path:
        """Write a minimal .pt payload that EmbeddingDataset expects."""
        pt_path = tmp_path / "test_embeddings.pt"
        torch.save({
            "embeddings": torch.randn(n, dim),
            "scores": torch.rand(n),
            "image_paths": [f"img_{i:04d}.jpg" for i in range(n)],
        }, pt_path)
        return pt_path

    def test_len(self, tmp_path: Path) -> None:
        pt = self._make_pt_file(tmp_path, n=20)
        ds = EmbeddingDataset(str(pt))
        assert len(ds) == 20

    def test_getitem_shapes(self, tmp_path: Path) -> None:
        pt = self._make_pt_file(tmp_path, n=10, dim=768)
        ds = EmbeddingDataset(str(pt))
        path, emb, score = ds[0]
        assert isinstance(path, str)
        assert emb.shape == (768,)
        assert score.shape == ()  # scalar tensor

    def test_scores_in_range(self, tmp_path: Path) -> None:
        """Scores written as rand() should all be in [0, 1]."""
        pt = self._make_pt_file(tmp_path)
        ds = EmbeddingDataset(str(pt))
        assert ds.scores.min() >= 0.0
        assert ds.scores.max() <= 1.0


# ---------------------------------------------------------------------------
# evaluate_mem_model
# ---------------------------------------------------------------------------

class TestEvaluateMemModel:
    """evaluate_mem_model should return three finite floats."""

    def _make_loader(self, n: int = 40, dim: int = 32) -> DataLoader:
        embeddings = torch.randn(n, dim)
        scores = torch.rand(n)
        # Wrap in a fake dataset that yields (path, emb, score) tuples
        class _FakeDS(torch.utils.data.Dataset):
            def __len__(self):
                return n
            def __getitem__(self, i):
                return f"img_{i}.jpg", embeddings[i], scores[i]
        return DataLoader(_FakeDS(), batch_size=8, shuffle=False)

    def test_returns_three_floats(self) -> None:
        model = MLPOnlyHead(hidden_dims=(16,), dropout_rate=0.0, input_dim=32)
        loader = self._make_loader(n=40, dim=32)
        loss, rho, std = evaluate_mem_model(
            model, loader, torch.device("cpu"), nn.MSELoss()
        )
        assert isinstance(loss, float)
        assert isinstance(rho, float)
        assert isinstance(std, float)

    def test_loss_is_non_negative(self) -> None:
        model = MLPOnlyHead(hidden_dims=(16,), dropout_rate=0.0, input_dim=32)
        loader = self._make_loader(n=32, dim=32)
        loss, _, _ = evaluate_mem_model(
            model, loader, torch.device("cpu"), nn.MSELoss()
        )
        assert loss >= 0.0


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

class TestSeedEverything:
    def test_deterministic_torch(self) -> None:
        """Two calls with the same seed must produce identical random tensors."""
        seed_everything(42)
        t1 = torch.randn(10)
        seed_everything(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2), "seed_everything not making torch deterministic"

    def test_deterministic_numpy(self) -> None:
        seed_everything(99)
        a1 = np.random.rand(10)
        seed_everything(99)
        a2 = np.random.rand(10)
        assert np.allclose(a1, a2), "seed_everything not making numpy deterministic"

    def test_different_seeds_differ(self) -> None:
        seed_everything(1)
        t1 = torch.randn(50)
        seed_everything(2)
        t2 = torch.randn(50)
        assert not torch.allclose(t1, t2), "Different seeds produced identical tensors"


# ---------------------------------------------------------------------------
# Checkpoint dict structure
# ---------------------------------------------------------------------------

class TestCheckpointStructure:
    """Verify that the checkpoint saved by train_mem_model contains required keys."""

    REQUIRED_KEYS = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "val_loss",
        "spearman_rho",
        "rng_state",
        "numpy_rng_state",
        "timestamp",
    }

    def test_checkpoint_keys_present(self, tmp_path: Path) -> None:
        """Run one training step and confirm checkpoint dict has all required keys."""
        from functions.train_mem_pipeline import train_mem_model

        model = MLPOnlyHead(hidden_dims=(16,), dropout_rate=0.0, input_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        n = 20

        class _FakeDS(torch.utils.data.Dataset):
            def __len__(self):
                return n
            def __getitem__(self, i):
                return f"img_{i}.jpg", torch.randn(32), torch.rand(1).squeeze()

        loader = DataLoader(_FakeDS(), batch_size=8)
        chk_prefix = str(tmp_path / "chk")

        train_mem_model(
            model=model,
            train_loader=loader,
            val_loader=loader,
            device=torch.device("cpu"),
            optimizer=optimizer,
            criterion=criterion,
            epochs=1,
            early_stopping_patience=5,
            checkpoint_path=chk_prefix,
            fold=1,
            save_checkpoint=True,
        )

        # Find the saved checkpoint
        chk_files = list(tmp_path.glob("*.pth"))
        assert len(chk_files) == 1, f"Expected 1 checkpoint, found {len(chk_files)}"
        chk = torch.load(chk_files[0], map_location="cpu", weights_only=False)
        missing = self.REQUIRED_KEYS - set(chk.keys())
        assert not missing, f"Checkpoint missing keys: {missing}"
