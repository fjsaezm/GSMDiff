"""Adapters that expose Diffusers denoisers as mathematical score models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class DiffusionPrediction:
    score: Tensor
    denoised: Tensor
    alpha: Tensor
    sigma: Tensor


class DiffusersScoreModel:
    """Convert a Diffusers UNet prediction into ``score(x_t, t)``.

    The adapter supports all prediction conventions exposed by the standard
    Diffusers schedulers: epsilon, clean sample, and v-prediction.
    """

    def __init__(self, model: Any, scheduler: Any) -> None:
        if not hasattr(scheduler, "alphas_cumprod"):
            raise TypeError("The scheduler must expose alphas_cumprod.")
        self.model = model
        self.scheduler = scheduler

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        revision: str | None = None,
        use_safetensors: bool | None = None,
    ) -> DiffusersScoreModel:
        try:
            from diffusers import DDPMPipeline
        except ImportError as error:
            raise ImportError(
                "The CelebA experiment requires the 'diffusion' extra: "
                "pip install -e '.[diffusion]'."
            ) from error

        options: dict[str, Any] = {"torch_dtype": dtype}
        if revision is not None:
            options["revision"] = revision
        if use_safetensors is not None:
            options["use_safetensors"] = use_safetensors
        pipeline = DDPMPipeline.from_pretrained(model_id, **options)
        pipeline.unet.to(device)
        pipeline.unet.eval()
        return cls(pipeline.unet, pipeline.scheduler)

    def set_timesteps(self, num_inference_steps: int, device: torch.device) -> None:
        self.scheduler.set_timesteps(num_inference_steps, device=device)

    def alpha_sigma(
        self, timestep: Tensor | int, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        index = int(timestep.item()) if isinstance(timestep, Tensor) else int(timestep)
        alpha_bar = self.scheduler.alphas_cumprod[index].to(device=device, dtype=dtype)
        alpha = alpha_bar.sqrt()
        sigma = (1.0 - alpha_bar).clamp_min(0.0).sqrt()
        return alpha, sigma

    @torch.no_grad()
    def predict(self, sample: Tensor, timestep: Tensor | int) -> DiffusionPrediction:
        model_input = self.scheduler.scale_model_input(sample, timestep)
        model_output = self.model(model_input, timestep).sample
        alpha, sigma = self.alpha_sigma(
            timestep, device=sample.device, dtype=sample.dtype
        )
        prediction_type = str(getattr(self.scheduler.config, "prediction_type", "epsilon"))
        sigma_safe = sigma.clamp_min(torch.finfo(sample.dtype).eps)

        if prediction_type == "epsilon":
            score = -model_output / sigma_safe
            denoised = (sample - sigma * model_output) / alpha
        elif prediction_type == "sample":
            denoised = model_output
            score = -(sample - alpha * denoised) / sigma_safe.square()
        elif prediction_type == "v_prediction":
            denoised = alpha * sample - sigma * model_output
            epsilon = alpha * model_output + sigma * sample
            score = -epsilon / sigma_safe
        else:
            raise ValueError(f"Unsupported prediction_type: {prediction_type!r}.")

        return DiffusionPrediction(score, denoised, alpha, sigma)
