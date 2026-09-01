"""Algorithm 5 OT-flow sampler for pretrained-flow/GGSM mixtures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor

from gsmdiff.geometric.filtered_score import FilteredGSMNoisyScore


class VelocityModel(Protocol):
    sample_size: int
    in_channels: int

    def velocity(self, sample: Tensor, time: Tensor | float) -> Tensor: ...


class FieldCoefficientSchedule(Protocol):
    def __call__(self, step_index: int, num_steps: int) -> float: ...


@dataclass(frozen=True)
class GeometricFlowStepDiagnostics:
    time: float
    flow_coefficient: float
    prior_coefficient: float
    learned_velocity_norm: float
    prior_velocity_norm: float
    weighted_learned_velocity_norm: float
    weighted_prior_velocity_norm: float
    velocity_cosine_similarity: float
    cg_iterations: int
    cg_maximum_relative_residual: float
    cg_total_iterations: int
    mean_field_iterations: int
    cg_solves_at_iteration_limit: int


@dataclass(frozen=True)
class GeometricFlowSnapshot:
    completed_steps: int
    time: float
    state: Tensor
    clean_estimate: Tensor


@dataclass(frozen=True)
class GeometricFlowSamplingResult:
    samples: Tensor
    initial_noise: Tensor
    steps: tuple[GeometricFlowStepDiagnostics, ...]
    snapshots: tuple[GeometricFlowSnapshot, ...] = ()


def _rms(value: Tensor) -> float:
    return float(value.square().mean().sqrt().detach().cpu())


class GeometricFlowSampler:
    """Euler implementation of report Algorithm 5 with independent weights.

    The report's ``lambda`` and ``1-lambda`` are generalized to
    ``flow_coefficient`` and ``prior_coefficient``. No sum-to-one constraint is
    imposed, so the update is ``v_r=lambda_q*v_theta + lambda_p*v_p``.
    """

    def __init__(
        self,
        flow: VelocityModel,
        prior_velocity: FilteredGSMNoisyScore,
        flow_coefficient: FieldCoefficientSchedule,
        prior_coefficient: FieldCoefficientSchedule,
    ) -> None:
        self.flow = flow
        self.prior_velocity = prior_velocity
        self.flow_coefficient = flow_coefficient
        self.prior_coefficient = prior_coefficient

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
        flow_coefficient_override: float | None = None,
        prior_coefficient_override: float | None = None,
        snapshot_every: int | None = None,
    ) -> GeometricFlowSamplingResult:
        if batch_size <= 0 or num_inference_steps <= 0:
            raise ValueError("batch_size and num_inference_steps must be positive.")
        if snapshot_every is not None and snapshot_every <= 0:
            raise ValueError("snapshot_every must be positive when provided.")
        shape = (
            batch_size,
            int(self.flow.in_channels),
            int(self.flow.sample_size),
            int(self.flow.sample_size),
        )
        if initial_noise is None:
            state = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"initial_noise has shape {tuple(initial_noise.shape)}, expected {shape}."
                )
            state = initial_noise.to(device=device, dtype=dtype).clone()
        initial_noise = state.clone()
        clean_estimate = torch.zeros_like(state)
        prior_solution: Tensor | None = None
        step_size = 1.0 / num_inference_steps
        diagnostics: list[GeometricFlowStepDiagnostics] = []
        snapshots: list[GeometricFlowSnapshot] = []

        for step_index in range(num_inference_steps):
            time_value = step_index * step_size
            time = torch.tensor(time_value, device=device, dtype=dtype)
            learned_velocity = self.flow.velocity(state, time)
            flow_weight = (
                float(flow_coefficient_override)
                if flow_coefficient_override is not None
                else float(self.flow_coefficient(step_index, num_inference_steps))
            )
            prior_weight = (
                float(prior_coefficient_override)
                if prior_coefficient_override is not None
                else float(self.prior_coefficient(step_index, num_inference_steps))
            )
            if flow_weight < 0.0 or prior_weight < 0.0:
                raise ValueError("Vector-field coefficients must be non-negative.")

            total_cg_iterations = 0
            solves_at_limit = 0
            final_cg_iterations = 0
            final_cg_residual = 0.0
            prior_velocity = torch.zeros_like(state)
            combined_velocity = flow_weight * learned_velocity
            if prior_weight == 0.0:
                prior_solution = None
                clean_estimate = state + (1.0 - time) * combined_velocity
                refinement_count = 0
            else:
                refinement_count = self.prior_velocity.mean_field_iterations
                solution = prior_solution if self.prior_velocity.warm_start else None
                for _ in range(refinement_count):
                    prediction = self.prior_velocity.predict_flow_velocity(
                        state,
                        clean_estimate,
                        time,
                        initial_solution=solution,
                    )
                    solution = prediction.linear_solution
                    prior_velocity = prediction.velocity
                    combined_velocity = (
                        flow_weight * learned_velocity + prior_weight * prior_velocity
                    )
                    clean_estimate = state + (1.0 - time) * combined_velocity
                    final_cg_iterations = prediction.diagnostics.iterations
                    final_cg_residual = prediction.diagnostics.maximum_relative_residual
                    total_cg_iterations += prediction.diagnostics.iterations
                    solves_at_limit += prediction.diagnostics.solves_at_iteration_limit
                prior_solution = solution

            state = state + step_size * combined_velocity
            completed_steps = step_index + 1
            if snapshot_every is not None and (
                completed_steps % snapshot_every == 0 or completed_steps == num_inference_steps
            ):
                snapshots.append(
                    GeometricFlowSnapshot(
                        completed_steps=completed_steps,
                        time=time_value,
                        state=state.detach().float().cpu().clone(),
                        clean_estimate=clean_estimate.detach().float().cpu().clone(),
                    )
                )
            diagnostics.append(
                GeometricFlowStepDiagnostics(
                    time=time_value,
                    flow_coefficient=flow_weight,
                    prior_coefficient=prior_weight,
                    learned_velocity_norm=_rms(learned_velocity),
                    prior_velocity_norm=_rms(prior_velocity),
                    weighted_learned_velocity_norm=_rms(flow_weight * learned_velocity),
                    weighted_prior_velocity_norm=_rms(prior_weight * prior_velocity),
                    velocity_cosine_similarity=float(
                        torch.nn.functional.cosine_similarity(
                            learned_velocity.flatten(start_dim=1),
                            prior_velocity.flatten(start_dim=1),
                            dim=1,
                            eps=torch.finfo(dtype).eps,
                        )
                        .mean()
                        .cpu()
                    ),
                    cg_iterations=final_cg_iterations,
                    cg_maximum_relative_residual=final_cg_residual,
                    cg_total_iterations=total_cg_iterations,
                    mean_field_iterations=refinement_count,
                    cg_solves_at_iteration_limit=solves_at_limit,
                )
            )

        return GeometricFlowSamplingResult(
            state, initial_noise, tuple(diagnostics), tuple(snapshots)
        )
