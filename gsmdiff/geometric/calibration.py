"""Dataset-specific calibration of first-difference prior scales."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


def first_difference_coefficients(samples: Tensor) -> Tensor:
    """Pool signed horizontal and vertical differences from clean NCHW images."""
    if samples.ndim != 4:
        raise ValueError(f"samples must have NCHW shape, got {tuple(samples.shape)}.")
    if samples.shape[-2] < 2 or samples.shape[-1] < 2:
        raise ValueError("sample height and width must both be at least two.")
    samples = samples.detach().float().cpu()
    horizontal = samples[..., :, :-1] - samples[..., :, 1:]
    vertical = samples[..., :-1, :] - samples[..., 1:, :]
    return torch.cat((horizontal.reshape(-1), vertical.reshape(-1)))


def coefficient_standard_deviation(sample_batches: Iterable[Tensor]) -> tuple[float, int]:
    """Return the population std of pooled dataset coefficients without concatenating them."""
    count = 0
    total = 0.0
    total_squared = 0.0
    for samples in sample_batches:
        values = first_difference_coefficients(samples).double()
        count += values.numel()
        total += float(values.sum())
        total_squared += float(values.square().sum())
    if count == 0:
        raise ValueError("at least one non-empty sample batch is required.")
    mean = total / count
    variance = max(0.0, total_squared / count - mean * mean)
    return math.sqrt(variance), count


def distribution_scale_from_standard_deviation(
    standard_deviation: float,
    distribution: str,
    *,
    shape: float | None = None,
) -> float:
    """Moment-match a configured density scale to a coefficient standard deviation."""
    standard_deviation = float(standard_deviation)
    if standard_deviation <= 0.0:
        raise ValueError("standard_deviation must be positive.")
    distribution = distribution.lower().replace("-", "_")
    if distribution in {"hyperbolic_secant", "sech", "negative_log_sech"}:
        # sech(x / scale) / (pi * scale) has variance pi^2 * scale^2 / 4.
        return 2.0 * standard_deviation / math.pi
    if distribution in {"gaussian", "l2"}:
        return standard_deviation
    if distribution in {"laplace", "l1"}:
        return standard_deviation / math.sqrt(2.0)
    if distribution in {"generalized_gaussian", "generalized"}:
        if shape is None or shape <= 0.0:
            raise ValueError("a positive shape is required for generalized_gaussian.")
        variance_factor = shape ** (2.0 / shape) * math.gamma(3.0 / shape) / math.gamma(
            1.0 / shape
        )
        return standard_deviation / math.sqrt(variance_factor)
    raise ValueError(
        f"standard-deviation matching is not implemented for distribution {distribution!r}."
    )
