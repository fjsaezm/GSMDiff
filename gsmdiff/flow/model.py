"""Adapters for pretrained optimal-transport flow-matching velocity models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from gsmdiff.flow.pnp_unet import PnPFlowUNet


class PretrainedPnPFlowModel:
    """Expose the public PnP-Flow CelebA U-Net as ``velocity(x_t, t)``."""

    def __init__(self, model: PnPFlowUNet) -> None:
        self.model = model
        self.sample_size = int(model.input_height)
        self.in_channels = int(model.input_channels)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
        download: bool = False,
        google_drive_id: str | None = None,
        sha256: str | None = None,
        input_channels: int = 3,
        input_height: int = 128,
        channels: int = 32,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        residual_blocks: int = 6,
        attention_resolutions: tuple[int, ...] = (16, 8),
    ) -> PretrainedPnPFlowModel:
        path = Path(checkpoint).expanduser()
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f"PnP-Flow checkpoint not found at {path}. Set model.download=true "
                    "to fetch the configured public checkpoint."
                )
            if not google_drive_id:
                raise ValueError("google_drive_id is required when download is enabled.")
            try:
                import gdown
            except ImportError as error:
                raise ImportError(
                    "Automatic checkpoint download requires the 'flow' extra: "
                    "pip install -e '.[flow]'."
                ) from error
            path.parent.mkdir(parents=True, exist_ok=True)
            downloaded = gdown.download(id=google_drive_id, output=str(path), quiet=False)
            if downloaded is None or not path.exists():
                raise RuntimeError(f"Failed to download the PnP-Flow checkpoint to {path}.")

        if sha256 is not None:
            digest = hashlib.sha256()
            with path.open("rb") as checkpoint_file:
                for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256.lower() != str(sha256).lower():
                raise ValueError(
                    f"Checkpoint SHA-256 mismatch for {path}: expected {sha256}, "
                    f"got {actual_sha256}."
                )

        model = PnPFlowUNet(
            input_channels=input_channels,
            input_height=input_height,
            ch=channels,
            ch_mult=channel_multipliers,
            num_res_blocks=residual_blocks,
            attn_resolutions=attention_resolutions,
        )
        loaded: Any = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict) and "state_dict" in loaded:
            loaded = loaded["state_dict"]
        if not isinstance(loaded, dict):
            raise TypeError(f"Expected a state dictionary in {path}.")
        if loaded and all(str(key).startswith("module.") for key in loaded):
            loaded = {str(key).removeprefix("module."): value for key, value in loaded.items()}
        model.load_state_dict(loaded, strict=True)
        model.to(device=device, dtype=dtype)
        model.eval()
        return cls(model)

    @torch.no_grad()
    def velocity(self, sample: Tensor, time: Tensor | float) -> Tensor:
        scalar_time = torch.as_tensor(time, device=sample.device, dtype=sample.dtype)
        timesteps = scalar_time.reshape(1).expand(sample.shape[0])
        return self.model(sample, timesteps)
