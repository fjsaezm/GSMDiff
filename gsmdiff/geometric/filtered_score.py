"""Noise-aware score for the Equation 5 filtered GSM image prior."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from gsmdiff.distributions import FirstDifferenceFilterProduct


def _batch_inner(left: Tensor, right: Tensor) -> Tensor:
    return (left * right).flatten(start_dim=1).sum(dim=1)


def _expand_batch(value: Tensor, reference: Tensor) -> Tensor:
    return value.reshape((-1,) + (1,) * (reference.ndim - 1))


@dataclass(frozen=True)
class CGDiagnostics:
    """Diagnostics for the final and accumulated mean-field linear solves."""

    iterations: int
    maximum_relative_residual: float
    total_iterations: int
    mean_field_iterations: int
    solves_at_iteration_limit: int


@dataclass(frozen=True)
class FilteredPriorPrediction:
    score: Tensor
    conditional_mean: Tensor
    linear_solution: Tensor
    diagnostics: CGDiagnostics


@dataclass(frozen=True)
class FilteredPriorVelocityPrediction:
    """One Algorithm 5 GGSM fixed-point velocity update."""

    velocity: Tensor
    conditional_mean: Tensor
    linear_solution: Tensor
    diagnostics: CGDiagnostics


class FilteredGSMNoisyScore:
    """Mean-field approximation to the noisy score of a filtered GSM prior.

    Given plug-in filter precisions, this solves

    ``(alpha^2 I + sigma^2 F^T Omega F) y = x_t``

    and returns the Tweedie score formed from ``E[x_0|x_t] = alpha*y``.
    Algorithm 2 alternates precision updates and linear solves for
    ``mean_field_iterations`` rounds. The solve is matrix-free and batched.
    """

    def __init__(
        self,
        prior: FirstDifferenceFilterProduct,
        *,
        cg_max_iterations: int = 50,
        cg_tolerance: float = 1e-4,
        numerical_epsilon: float = 1e-8,
        warm_start: bool = False,
        mean_field_iterations: int = 1,
    ) -> None:
        if cg_max_iterations <= 0:
            raise ValueError("cg_max_iterations must be positive.")
        if cg_tolerance <= 0 or numerical_epsilon <= 0:
            raise ValueError("CG tolerance and numerical epsilon must be positive.")
        if mean_field_iterations <= 0:
            raise ValueError("mean_field_iterations must be positive.")
        self.prior = prior
        self.cg_max_iterations = int(cg_max_iterations)
        self.cg_tolerance = float(cg_tolerance)
        self.numerical_epsilon = float(numerical_epsilon)
        self.warm_start = bool(warm_start)
        self.mean_field_iterations = int(mean_field_iterations)

    def _weights(self, denoised: Tensor) -> tuple[Tensor, Tensor]:
        horizontal, vertical = self.prior._responses(denoised)
        distribution = self.prior.coefficient_distribution
        horizontal_weight = distribution.expected_precision(horizontal)
        vertical_weight = distribution.expected_precision(vertical)
        if not torch.isfinite(horizontal_weight).all() or not torch.isfinite(vertical_weight).all():
            raise FloatingPointError(
                "The plug-in GSM precision is non-finite. Use a positive smoothing "
                "parameter for singular GSM families."
            )
        return horizontal_weight, vertical_weight

    def _precision_action(
        self,
        value: Tensor,
        horizontal_weight: Tensor,
        vertical_weight: Tensor,
    ) -> Tensor:
        horizontal, vertical = self.prior._responses(value)
        horizontal = horizontal_weight * horizontal
        vertical = vertical_weight * vertical
        result = torch.zeros_like(value)
        result[..., :, :-1] += horizontal
        result[..., :, 1:] -= horizontal
        result[..., :-1, :] += vertical
        result[..., 1:, :] -= vertical

        pixel_count = value.shape[-2] * value.shape[-1]
        spatial_mean = value.mean(dim=(-2, -1), keepdim=True)
        result += spatial_mean / (self.prior.mean_scale**2 * pixel_count)
        return result

    def _precision_diagonal(
        self,
        reference: Tensor,
        horizontal_weight: Tensor,
        vertical_weight: Tensor,
    ) -> Tensor:
        diagonal = torch.zeros_like(reference)
        diagonal[..., :, :-1] += horizontal_weight
        diagonal[..., :, 1:] += horizontal_weight
        diagonal[..., :-1, :] += vertical_weight
        diagonal[..., 1:, :] += vertical_weight
        pixel_count = reference.shape[-2] * reference.shape[-1]
        diagonal += 1.0 / (self.prior.mean_scale**2 * pixel_count**2)
        return diagonal

    def _solve(
        self,
        right_hand_side: Tensor,
        alpha_squared: Tensor,
        sigma_squared: Tensor,
        horizontal_weight: Tensor,
        vertical_weight: Tensor,
        initial_solution: Tensor | None = None,
    ) -> tuple[Tensor, CGDiagnostics]:
        def action(value: Tensor) -> Tensor:
            return alpha_squared * value + sigma_squared * self._precision_action(
                value, horizontal_weight, vertical_weight
            )

        diagonal = alpha_squared + sigma_squared * self._precision_diagonal(
            right_hand_side, horizontal_weight, vertical_weight
        )
        diagonal = diagonal.clamp_min(self.numerical_epsilon)
        if initial_solution is None:
            solution = torch.zeros_like(right_hand_side)
        else:
            if initial_solution.shape != right_hand_side.shape:
                raise ValueError("initial_solution must have the same shape as right_hand_side.")
            solution = initial_solution.to(right_hand_side).clone()
        residual = right_hand_side - action(solution)
        preconditioned = residual / diagonal
        direction = preconditioned.clone()
        residual_preconditioned = _batch_inner(residual, preconditioned)
        rhs_norm = (
            _batch_inner(right_hand_side, right_hand_side).sqrt().clamp_min(self.numerical_epsilon)
        )
        relative_residual = _batch_inner(residual, residual).sqrt() / rhs_norm
        maximum_relative_residual = float(relative_residual.max().detach().cpu())
        if maximum_relative_residual <= self.cg_tolerance:
            return solution, CGDiagnostics(0, maximum_relative_residual, 0, 1, 0)

        for iteration in range(1, self.cg_max_iterations + 1):
            action_direction = action(direction)
            denominator = _batch_inner(direction, action_direction).clamp_min(
                self.numerical_epsilon
            )
            step_size = residual_preconditioned / denominator
            solution = solution + _expand_batch(step_size, solution) * direction
            residual = residual - _expand_batch(step_size, residual) * action_direction
            relative_residual = _batch_inner(residual, residual).sqrt() / rhs_norm
            maximum_relative_residual = float(relative_residual.max().detach().cpu())
            if maximum_relative_residual <= self.cg_tolerance:
                break
            next_preconditioned = residual / diagonal
            next_residual_preconditioned = _batch_inner(residual, next_preconditioned)
            coefficient = next_residual_preconditioned / residual_preconditioned.clamp_min(
                self.numerical_epsilon
            )
            direction = next_preconditioned + _expand_batch(coefficient, direction) * direction
            preconditioned = next_preconditioned
            residual_preconditioned = next_residual_preconditioned

        return solution, CGDiagnostics(
            iteration,
            maximum_relative_residual,
            iteration,
            1,
            int(iteration >= self.cg_max_iterations),
        )

    @torch.no_grad()
    def predict(
        self,
        noisy: Tensor,
        denoised_for_weights: Tensor,
        alpha: Tensor,
        sigma: Tensor,
        initial_solution: Tensor | None = None,
    ) -> FilteredPriorPrediction:
        self.prior._validate(noisy)
        alpha_squared = alpha.square()
        sigma_squared = sigma.square()
        estimate = denoised_for_weights
        solution = initial_solution
        total_iterations = 0
        solves_at_iteration_limit = 0
        diagnostics: CGDiagnostics | None = None
        for _ in range(self.mean_field_iterations):
            horizontal_weight, vertical_weight = self._weights(estimate)
            solution, diagnostics = self._solve(
                noisy,
                alpha_squared,
                sigma_squared,
                horizontal_weight,
                vertical_weight,
                # The previous inner solution is a valid initial guess for the
                # next fixed-point solve. ``warm_start`` controls only reuse
                # across reverse-diffusion timesteps in the outer sampler.
                solution,
            )
            total_iterations += diagnostics.iterations
            solves_at_iteration_limit += diagnostics.solves_at_iteration_limit
            estimate = alpha * solution

        assert diagnostics is not None
        diagnostics = CGDiagnostics(
            diagnostics.iterations,
            diagnostics.maximum_relative_residual,
            total_iterations,
            self.mean_field_iterations,
            solves_at_iteration_limit,
        )
        conditional_mean = estimate
        sigma_squared_safe = sigma_squared.clamp_min(self.numerical_epsilon)
        score = -(noisy - alpha * conditional_mean) / sigma_squared_safe
        return FilteredPriorPrediction(score, conditional_mean, solution, diagnostics)

    @torch.no_grad()
    def predict_flow_velocity(
        self,
        state: Tensor,
        clean_estimate_for_weights: Tensor,
        time: Tensor | float,
        initial_solution: Tensor | None = None,
    ) -> FilteredPriorVelocityPrediction:
        """Evaluate one GGSM velocity refinement from Algorithm 5.

        For the OT path ``x_t=(1-t)x_0+t x_1``, this solves
        ``[t² I + (1-t)² P] w = x_t`` and returns
        ``v_p=t w-(1-t)P w``. The outer sampler owns the fixed-point loop
        because its next clean estimate uses the *combined* velocity field.
        """
        self.prior._validate(state)
        scalar_time = torch.as_tensor(time, device=state.device, dtype=state.dtype)
        if scalar_time.numel() != 1 or not 0.0 <= float(scalar_time) < 1.0:
            raise ValueError("time must be a scalar in [0, 1).")
        horizontal_weight, vertical_weight = self._weights(clean_estimate_for_weights)
        one_minus_time = 1.0 - scalar_time
        solution, diagnostics = self._solve(
            state,
            scalar_time.square(),
            one_minus_time.square(),
            horizontal_weight,
            vertical_weight,
            initial_solution,
        )
        precision_solution = self._precision_action(solution, horizontal_weight, vertical_weight)
        velocity = scalar_time * solution - one_minus_time * precision_solution
        return FilteredPriorVelocityPrediction(
            velocity=velocity,
            conditional_mean=scalar_time * solution,
            linear_solution=solution,
            diagnostics=diagnostics,
        )
