"""Super-Gaussian families used by the original GSM image loss.

All formulas are analytic, differentiable almost everywhere, device agnostic,
and preserve the input tensor's dtype. The optional smoothing parameters make
the conditional precision finite at the origin, where the exact Laplace GSM
has an infinite conditional mean precision.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from gsmdiff.distributions.base import SuperGaussian


def _validate_scale(scale: float) -> None:
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}.")


def _validate_smoothing(smoothing: float) -> None:
    if smoothing < 0:
        raise ValueError(f"smoothing must be non-negative, got {smoothing}.")


class Gaussian(SuperGaussian):
    """The ``l2`` model: ``rho(x) = x^2 / (2 scale^2)``."""

    def __init__(self, scale: float = 1.0, name: str = "l2") -> None:
        _validate_scale(scale)
        self.scale = float(scale)
        self.name = str(name)

    def potential(self, value: Tensor) -> Tensor:
        return 0.5 * (value / self.scale).square()

    def score(self, value: Tensor) -> Tensor:
        return -value / self.scale**2

    def expected_precision(self, value: Tensor) -> Tensor:
        return torch.full_like(value, 1.0 / self.scale**2)


class Laplace(SuperGaussian):
    """The ``l1`` model, optionally with smooth absolute value.

    ``smoothing=0`` is the exact Laplace distribution. A positive value uses
    ``sqrt(x^2+smoothing^2)`` and avoids an infinite precision at zero.
    """

    def __init__(
        self,
        scale: float = 1.0,
        smoothing: float = 0.0,
        name: str = "l1",
    ) -> None:
        _validate_scale(scale)
        _validate_smoothing(smoothing)
        self.scale = float(scale)
        self.smoothing = float(smoothing)
        self.name = str(name)

    def _radius(self, value: Tensor) -> Tensor:
        if self.smoothing == 0:
            return value.abs()
        return torch.sqrt(value.square() + self.smoothing**2)

    def potential(self, value: Tensor) -> Tensor:
        return self._radius(value) / self.scale

    def expected_precision(self, value: Tensor) -> Tensor:
        radius = self._radius(value)
        if self.smoothing == 0:
            return torch.where(radius > 0, 1.0 / (self.scale * radius), torch.inf)
        return 1.0 / (self.scale * radius)

    def score(self, value: Tensor) -> Tensor:
        if self.smoothing == 0:
            return -value.sign() / self.scale
        return -value * self.expected_precision(value)


class GeneralizedGaussian(SuperGaussian):
    """Generalized-Gaussian family, including the legacy ``l0.8`` model.

    The potential is ``(r / scale)^shape / shape`` with
    ``r=sqrt(x^2+smoothing^2)``. Dividing by ``shape`` gives exactly the
    ``r^(shape-2)`` precision convention used by the legacy GSM loss. The
    default positive smoothing handles the singular score of shapes below one.
    """

    def __init__(
        self,
        shape: float = 0.8,
        scale: float = 1.0,
        smoothing: float = 1e-3,
        name: str = "l0.8",
    ) -> None:
        if shape <= 0:
            raise ValueError(f"shape must be positive, got {shape}.")
        _validate_scale(scale)
        _validate_smoothing(smoothing)
        self.shape = float(shape)
        self.scale = float(scale)
        self.smoothing = float(smoothing)
        self.name = str(name)

    def _radius(self, value: Tensor) -> Tensor:
        if self.smoothing == 0:
            return value.abs()
        return torch.sqrt(value.square() + self.smoothing**2)

    def potential(self, value: Tensor) -> Tensor:
        return (self._radius(value) / self.scale).pow(self.shape) / self.shape

    def expected_precision(self, value: Tensor) -> Tensor:
        radius = self._radius(value)
        precision = radius.pow(self.shape - 2.0) / self.scale**self.shape
        return precision

    def score(self, value: Tensor) -> Tensor:
        if self.smoothing == 0:
            radius = self._radius(value)
            magnitude = radius.pow(self.shape - 1.0) / self.scale**self.shape
            return torch.where(
                radius == 0, torch.zeros_like(value), -value.sign() * magnitude
            )
        return -value * self.expected_precision(value)


class HyperbolicSecant(SuperGaussian):
    """The ``-logsech`` model: ``p(x) proportional to sech(x / scale)``."""

    def __init__(self, scale: float = 1.0, name: str = "negative_log_sech") -> None:
        _validate_scale(scale)
        self.scale = float(scale)
        self.name = str(name)

    def potential(self, value: Tensor) -> Tensor:
        z = value / self.scale
        # log(cosh(z)) written without overflow for large coefficients.
        return torch.logaddexp(z, -z) - math.log(2.0)

    def score(self, value: Tensor) -> Tensor:
        return -torch.tanh(value / self.scale) / self.scale

    def expected_precision(self, value: Tensor) -> Tensor:
        z = value / self.scale
        ratio = torch.where(z == 0, torch.ones_like(z), torch.tanh(z) / z)
        return ratio / self.scale**2


class SmoothedPowerLaw(SuperGaussian):
    """A proper smoothed version of the legacy ``log`` penalty.

    The density is ``p(x) proportional to (x^2 + scale^2)^(-tail_exponent/2)``.
    It is normalizable exactly when ``tail_exponent > 1``. With exponent two,
    its conditional precision has the same inverse-square behavior as the
    ``log`` branch in the legacy ``theta_function``.
    """

    def __init__(
        self,
        scale: float = 1e-2,
        tail_exponent: float = 2.0,
        name: str = "log",
    ) -> None:
        _validate_scale(scale)
        if tail_exponent <= 1:
            raise ValueError(
                "tail_exponent must be > 1 for a proper distribution; "
                f"got {tail_exponent}."
            )
        self.scale = float(scale)
        self.tail_exponent = float(tail_exponent)
        self.name = str(name)

    def potential(self, value: Tensor) -> Tensor:
        return 0.5 * self.tail_exponent * torch.log(value.square() + self.scale**2)

    def expected_precision(self, value: Tensor) -> Tensor:
        return self.tail_exponent / (value.square() + self.scale**2)

    def score(self, value: Tensor) -> Tensor:
        return -value * self.expected_precision(value)
