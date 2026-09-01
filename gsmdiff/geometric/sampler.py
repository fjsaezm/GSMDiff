"""Reverse diffusion samplers for geometric diffusion/GSM mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from gsmdiff.geometric.diffusion import DiffusersScoreModel
from gsmdiff.geometric.filtered_score import FilteredGSMNoisyScore


class BlendSchedule(Protocol):
    def __call__(self, step_index: int, num_steps: int) -> float: ...


@dataclass(frozen=True)
class GeometricStepDiagnostics:
    timestep: int
    blend: float
    diffusion_score_norm: float
    prior_score_norm: float
    cg_iterations: int
    cg_maximum_relative_residual: float
    cg_total_iterations: int = 0
    mean_field_iterations: int = 0
    cg_solves_at_iteration_limit: int = 0
    score_combination: str = "geometric"
    prior_weight: float = 0.0
    weighted_prior_score_norm: float = 0.0
    score_cosine_similarity: float = 0.0


@dataclass(frozen=True)
class GeometricSnapshot:
    """CPU snapshot captured after a requested number of reverse steps."""

    completed_steps: int
    timestep: int
    state: Tensor
    denoised: Tensor


@dataclass(frozen=True)
class GeometricSamplingResult:
    samples: Tensor
    initial_noise: Tensor
    steps: tuple[GeometricStepDiagnostics, ...]
    snapshots: tuple[GeometricSnapshot, ...] = ()


def _root_mean_square(value: Tensor) -> float:
    return float(value.square().mean().sqrt().detach().cpu())


class GeometricDiffusionSampler:
    """Algorithm 2 geometric DDPM sampler with a filtered GGSM prior."""

    def __init__(
        self,
        diffusion: DiffusersScoreModel,
        prior_score: FilteredGSMNoisyScore,
        blend_schedule: BlendSchedule,
        *,
        guidance_schedule: BlendSchedule | None = None,
        score_combination: str = "geometric",
        method: str = "reverse_sde",
    ) -> None:
        if method not in {"reverse_sde", "ddim"}:
            raise ValueError("method must be 'reverse_sde' or 'ddim'.")
        if score_combination not in {"geometric", "additive"}:
            raise ValueError("score_combination must be 'geometric' or 'additive'.")
        if score_combination == "additive" and guidance_schedule is None:
            raise ValueError("guidance_schedule is required for additive score combination.")
        self.diffusion = diffusion
        self.prior_score = prior_score
        self.blend_schedule = blend_schedule
        self.guidance_schedule = guidance_schedule
        self.score_combination = score_combination
        self.method = method

    @torch.no_grad()
    def sample(
        self,
        *,
        batch_size: int,
        num_inference_steps: int,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
        initial_noise: Tensor | None = None,
        blend_override: float | None = None,
        guidance_override: float | None = None,
        snapshot_every: int | None = None,
    ) -> GeometricSamplingResult:
        if batch_size <= 0 or num_inference_steps <= 0:
            raise ValueError("batch_size and num_inference_steps must be positive.")
        if snapshot_every is not None and snapshot_every <= 0:
            raise ValueError("snapshot_every must be positive when provided.")
        self.diffusion.set_timesteps(num_inference_steps, device)
        model = self.diffusion.model
        sample_size = model.config.sample_size
        if isinstance(sample_size, int):
            height = width = sample_size
        else:
            height, width = (int(size) for size in sample_size)
        shape = (batch_size, int(model.config.in_channels), height, width)
        if initial_noise is None:
            sample = torch.randn(shape, device=device, dtype=dtype, generator=generator)
            sample *= float(getattr(self.diffusion.scheduler, "init_noise_sigma", 1.0))
            initial_noise = sample.clone()
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise has shape {tuple(initial_noise.shape)}, expected {shape}."
                )
            sample = initial_noise.to(device=device, dtype=dtype).clone()
            initial_noise = sample.clone()

        timesteps = self.diffusion.scheduler.timesteps
        diagnostics: list[GeometricStepDiagnostics] = []
        snapshots: list[GeometricSnapshot] = []
        prior_solution: Tensor | None = None
        alpha_bars = self.diffusion.scheduler.alphas_cumprod
        for step_index, timestep in enumerate(timesteps):
            prediction = self.diffusion.predict(sample, timestep)
            if self.score_combination == "geometric":
                blend = (
                    float(blend_override)
                    if blend_override is not None
                    else float(self.blend_schedule(step_index, len(timesteps)))
                )
                if not 0.0 <= blend <= 1.0:
                    raise ValueError(f"The blend schedule returned {blend}, outside [0, 1].")
                prior_weight = 1.0 - blend
            else:
                assert self.guidance_schedule is not None
                blend = 1.0
                prior_weight = (
                    float(guidance_override)
                    if guidance_override is not None
                    else float(self.guidance_schedule(step_index, len(timesteps)))
                )
                if prior_weight < 0.0:
                    raise ValueError(
                        f"The guidance schedule returned {prior_weight}; gamma must be non-negative."
                    )
            if prior_weight == 0.0:
                prior_score = torch.zeros_like(prediction.score)
                cg_iterations = 0
                cg_residual = 0.0
                cg_total_iterations = 0
                mean_field_iterations = 0
                cg_solves_at_iteration_limit = 0
                prior_solution = None
            else:
                prior = self.prior_score.predict(
                    sample,
                    prediction.denoised,
                    prediction.alpha,
                    prediction.sigma,
                    initial_solution=(
                        prior_solution if self.prior_score.warm_start else None
                    ),
                )
                prior_score = prior.score
                prior_solution = prior.linear_solution
                cg_iterations = prior.diagnostics.iterations
                cg_residual = prior.diagnostics.maximum_relative_residual
                cg_total_iterations = prior.diagnostics.total_iterations
                mean_field_iterations = prior.diagnostics.mean_field_iterations
                cg_solves_at_iteration_limit = (
                    prior.diagnostics.solves_at_iteration_limit
                )
            total_score = blend * prediction.score + prior_weight * prior_score
            total_denoised = (
                sample + prediction.sigma.square() * total_score
            ) / prediction.alpha

            timestep_index = int(timestep.item())
            if step_index + 1 < len(timesteps):
                previous_timestep = int(timesteps[step_index + 1].item())
                previous_alpha_bar = alpha_bars[previous_timestep].to(
                    device=device, dtype=dtype
                )
            else:
                previous_alpha_bar = torch.ones((), device=device, dtype=dtype)

            if self.method == "ddim":
                previous_alpha = previous_alpha_bar.sqrt()
                previous_sigma = (1.0 - previous_alpha_bar).clamp_min(0.0).sqrt()
                sample = (
                    previous_alpha * total_denoised
                    - previous_sigma * prediction.sigma * total_score
                )
            else:
                current_alpha_bar = alpha_bars[timestep_index].to(
                    device=device, dtype=dtype
                )
                transition_alpha = (current_alpha_bar / previous_alpha_bar).clamp(
                    min=torch.finfo(dtype).eps, max=1.0
                )
                transition_beta = 1.0 - transition_alpha
                sample = (sample + transition_beta * total_score) / transition_alpha.sqrt()
                if step_index + 1 < len(timesteps):
                    current_sigma_squared = (1.0 - current_alpha_bar).clamp_min(0.0)
                    previous_sigma_squared = (1.0 - previous_alpha_bar).clamp_min(0.0)
                    posterior_variance = (
                        previous_sigma_squared
                        / current_sigma_squared.clamp_min(torch.finfo(dtype).eps)
                        * transition_beta
                    ).clamp_min(0.0)
                    noise = torch.randn(
                        sample.shape,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                    sample = sample + posterior_variance.sqrt() * noise

            completed_steps = step_index + 1
            should_snapshot = snapshot_every is not None and (
                completed_steps % snapshot_every == 0 or completed_steps == len(timesteps)
            )
            if should_snapshot:
                snapshots.append(
                    GeometricSnapshot(
                        completed_steps=completed_steps,
                        timestep=timestep_index,
                        state=sample.detach().float().cpu().clone(),
                        denoised=total_denoised.detach().float().cpu().clone(),
                    )
                )

            diagnostics.append(
                GeometricStepDiagnostics(
                    timestep=timestep_index,
                    blend=blend,
                    diffusion_score_norm=_root_mean_square(prediction.score),
                    prior_score_norm=_root_mean_square(prior_score),
                    cg_iterations=cg_iterations,
                    cg_maximum_relative_residual=cg_residual,
                    cg_total_iterations=cg_total_iterations,
                    mean_field_iterations=mean_field_iterations,
                    cg_solves_at_iteration_limit=cg_solves_at_iteration_limit,
                    score_combination=self.score_combination,
                    prior_weight=prior_weight,
                    weighted_prior_score_norm=_root_mean_square(prior_weight * prior_score),
                    score_cosine_similarity=float(
                        torch.nn.functional.cosine_similarity(
                            prediction.score.flatten(start_dim=1),
                            prior_score.flatten(start_dim=1),
                            dim=1,
                            eps=torch.finfo(prediction.score.dtype).eps,
                        )
                        .mean()
                        .detach()
                        .cpu()
                    ),
                )
            )

        return GeometricSamplingResult(
            sample,
            initial_noise,
            tuple(diagnostics),
            tuple(snapshots),
        )
