"""ULA and MALA kernels for score-addressable GSM targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


class ScoreTarget:
    """Structural interface documented without imposing inheritance on targets."""

    def score(self, value: Tensor) -> Tensor: ...

    def log_density(self, value: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class SamplerDiagnostics:
    acceptance_rate: float
    mean_squared_jump: float
    steps: int


@dataclass(frozen=True)
class ChainResult:
    """Saved states have shape ``(draw, chain, *event_shape)``."""

    samples: Tensor
    final_state: Tensor
    diagnostics: SamplerDiagnostics


def _noise_like(value: Tensor, generator: torch.Generator | None) -> Tensor:
    return torch.randn(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    )


def _sum_event(value: Tensor) -> Tensor:
    return value.flatten(start_dim=1).sum(dim=1)


class _LangevinSampler:
    def __init__(self, step_size: float, name: str) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}.")
        self.step_size = float(step_size)
        self.name = str(name)

    def step(
        self,
        target: ScoreTarget,
        state: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
        self,
        target: ScoreTarget,
        initial_state: Tensor,
        *,
        num_draws: int = 1,
        burn_in: int = 0,
        thinning: int = 1,
        generator: torch.Generator | None = None,
    ) -> ChainResult:
        if initial_state.ndim < 2:
            raise ValueError("initial_state must have shape (chains, *event_shape).")
        if num_draws <= 0 or burn_in < 0 or thinning <= 0:
            raise ValueError(
                "num_draws and thinning must be positive; burn_in cannot be negative."
            )

        state = initial_state.detach().clone()
        saved: list[Tensor] = []
        accepted = torch.zeros(state.shape[0], dtype=torch.float64, device=state.device)
        squared_jump = torch.zeros_like(accepted)
        total_steps = burn_in + num_draws * thinning
        for index in range(total_steps):
            previous = state
            state, accepted_step = self.step(target, state, generator=generator)
            accepted += accepted_step.to(torch.float64)
            squared_jump += _sum_event((state - previous).square()).to(torch.float64)
            if index >= burn_in and (index - burn_in + 1) % thinning == 0:
                saved.append(state.clone())

        diagnostics = SamplerDiagnostics(
            acceptance_rate=float((accepted / total_steps).mean().cpu()),
            mean_squared_jump=float((squared_jump / total_steps).mean().cpu()),
            steps=total_steps,
        )
        return ChainResult(torch.stack(saved), state, diagnostics)


class ULASampler(_LangevinSampler):
    """Unadjusted Langevin Algorithm using the report's epsilon convention."""

    def __init__(self, step_size: float, name: str = "ula") -> None:
        super().__init__(step_size, name)

    def step(
        self,
        target: ScoreTarget,
        state: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        proposal = (
            state
            + 0.5 * self.step_size * target.score(state)
            + self.step_size**0.5 * _noise_like(state, generator)
        )
        accepted = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
        return proposal, accepted


class MALASampler(_LangevinSampler):
    """Metropolis-adjusted Langevin with exact or score-path density ratios."""

    def __init__(
        self,
        step_size: float,
        density_ratio: str = "exact",
        quadrature_points: int = 16,
        name: str = "mala",
    ) -> None:
        super().__init__(step_size, name)
        if density_ratio not in {"exact", "path_integral"}:
            raise ValueError("density_ratio must be 'exact' or 'path_integral'.")
        if quadrature_points <= 0:
            raise ValueError("quadrature_points must be positive.")
        self.density_ratio = density_ratio
        self.quadrature_points = int(quadrature_points)

    def _target_log_ratio(self, target: ScoreTarget, start: Tensor, end: Tensor) -> Tensor:
        if self.density_ratio == "exact":
            return target.log_density(end) - target.log_density(start)

        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_points)
        nodes_tensor = torch.as_tensor(
            (nodes + 1.0) / 2.0, dtype=start.dtype, device=start.device
        )
        weights_tensor = torch.as_tensor(
            weights / 2.0, dtype=start.dtype, device=start.device
        )
        direction = end - start
        ratio = torch.zeros(start.shape[0], dtype=start.dtype, device=start.device)
        for node, weight in zip(nodes_tensor, weights_tensor):
            point = start + node * direction
            ratio += weight * _sum_event(direction * target.score(point))
        return ratio

    def step(
        self,
        target: ScoreTarget,
        state: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        score = target.score(state)
        forward_mean = state + 0.5 * self.step_size * score
        proposal = forward_mean + self.step_size**0.5 * _noise_like(state, generator)
        proposal_score = target.score(proposal)
        reverse_mean = proposal + 0.5 * self.step_size * proposal_score

        target_ratio = self._target_log_ratio(target, state, proposal)
        forward_error = _sum_event((proposal - forward_mean).square())
        reverse_error = _sum_event((state - reverse_mean).square())
        proposal_ratio = (forward_error - reverse_error) / (2.0 * self.step_size)
        log_acceptance = target_ratio + proposal_ratio
        uniforms = torch.rand(
            (state.shape[0],), dtype=state.dtype, device=state.device, generator=generator
        )
        accepted = torch.log(uniforms) < torch.minimum(
            log_acceptance, torch.zeros_like(log_acceptance)
        )
        mask = accepted.reshape((-1,) + (1,) * (state.ndim - 1))
        return torch.where(mask, proposal, state), accepted
