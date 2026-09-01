"""Image priors built by applying coefficient densities to filter responses."""

from __future__ import annotations

import torch
from torch import Tensor

from gsmdiff.distributions.base import SuperGaussian


class FirstDifferenceFilterProduct:
    """Product prior on horizontal and vertical first differences.

    For a coefficient density ``f``, this target implements

    ``p(x) proportional to prod_i f((F_h x)_i) prod_i f((F_v x)_i)``.

    First differences do not constrain an image's constant component, so a
    Gaussian factor is placed on each channel's spatial mean. This is the
    minimal term needed to make the image target proper while leaving its
    high-pass statistics governed by ``coefficient_distribution``.
    """

    def __init__(
        self,
        coefficient_distribution: SuperGaussian,
        mean_scale: float = 10.0,
        name: str = "first_difference_filter_product",
    ) -> None:
        if mean_scale <= 0:
            raise ValueError(f"mean_scale must be positive, got {mean_scale}.")
        self.coefficient_distribution = coefficient_distribution
        self.mean_scale = float(mean_scale)
        self.name = str(name)

    @staticmethod
    def _validate(value: Tensor) -> None:
        if value.ndim < 3:
            raise ValueError(
                "value must have shape (chains, ..., height, width); "
                f"got {tuple(value.shape)}."
            )
        if value.shape[-2] < 2 or value.shape[-1] < 2:
            raise ValueError("height and width must both be at least two.")

    @staticmethod
    def _responses(value: Tensor) -> tuple[Tensor, Tensor]:
        horizontal = value[..., :, :-1] - value[..., :, 1:]
        vertical = value[..., :-1, :] - value[..., 1:, :]
        return horizontal, vertical

    def log_density(self, value: Tensor) -> Tensor:
        """Return one filter-product log density per chain."""
        self._validate(value)
        horizontal, vertical = self._responses(value)
        spatial_mean = value.mean(dim=(-2, -1))
        mean_log_density = -0.5 * (spatial_mean / self.mean_scale).square()
        return (
            self.coefficient_distribution.log_density(horizontal)
            + self.coefficient_distribution.log_density(vertical)
            + mean_log_density.flatten(start_dim=1).sum(dim=1)
        )

    def filter_energy(self, value: Tensor) -> Tensor:
        """Return ``sum_i rho((F x)_i)`` per image, excluding the DC factor."""
        self._validate(value)
        horizontal, vertical = self._responses(value)
        return (
            self.coefficient_distribution.potential(horizontal)
            .flatten(start_dim=1)
            .sum(dim=1)
            + self.coefficient_distribution.potential(vertical)
            .flatten(start_dim=1)
            .sum(dim=1)
        )

    def energy(self, value: Tensor) -> Tensor:
        """Return the full negative log density (up to constants) per image."""
        spatial_mean = value.mean(dim=(-2, -1))
        mean_energy = 0.5 * (spatial_mean / self.mean_scale).square()
        return self.filter_energy(value) + mean_energy.flatten(start_dim=1).sum(dim=1)

    def score(self, value: Tensor) -> Tensor:
        """Apply coefficient scores and the adjoint difference filters."""
        self._validate(value)
        horizontal, vertical = self._responses(value)
        horizontal_score = self.coefficient_distribution.score(horizontal)
        vertical_score = self.coefficient_distribution.score(vertical)

        score = torch.zeros_like(value)
        score[..., :, :-1] += horizontal_score
        score[..., :, 1:] -= horizontal_score
        score[..., :-1, :] += vertical_score
        score[..., 1:, :] -= vertical_score

        pixel_count = value.shape[-2] * value.shape[-1]
        spatial_mean = value.mean(dim=(-2, -1), keepdim=True)
        score -= spatial_mean / (self.mean_scale**2 * pixel_count)
        return score
