"""Interfaces shared by probability targets used by the samplers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor


class SuperGaussian(ABC):
    """Element-wise super-Gaussian target with independent image coefficients.

    The first tensor axis indexes independent Markov chains. All remaining axes
    are event (for example channel, height, and width) axes. Normalizing
    constants are deliberately omitted because ULA and MALA only need scores
    and log-density differences.
    """

    @abstractmethod
    def potential(self, value: Tensor) -> Tensor:
        """Return the element-wise negative log density, up to a constant."""

    @abstractmethod
    def score(self, value: Tensor) -> Tensor:
        """Return the element-wise score ``grad(log p(value))``."""

    @abstractmethod
    def expected_precision(self, value: Tensor) -> Tensor:
        """Return ``E[w | value]`` from the GSM score identity."""

    def elementwise_log_density(self, value: Tensor) -> Tensor:
        return -self.potential(value)

    def log_density(self, value: Tensor) -> Tensor:
        """Return one joint log density per chain."""
        if value.ndim == 0:
            raise ValueError("A chain axis is required; got a scalar tensor.")
        return self.elementwise_log_density(value).flatten(start_dim=1).sum(dim=1)

    def check_identity(self, value: Tensor, *, atol: float = 1e-6) -> None:
        """Raise if the analytic score violates ``score=-x E[w|x]``."""
        score = self.score(value)
        expected = -value * self.expected_precision(value)
        # For non-smooth densities such as exact Laplace, E[w|x=0] is infinite
        # while the score uses the conventional zero subgradient. The product
        # identity is understood by its limit there.
        expected = torch.where(value == 0, score, expected)
        if not torch.allclose(score, expected, atol=atol, rtol=1e-5):
            raise RuntimeError("The GSM score identity is not satisfied.")
