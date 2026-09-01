"""Schedules for geometric-mixture weights and additive prior guidance."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _validate_weight(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}.")
    return value


def _validate_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def _progress(step_index: int, num_steps: int) -> float:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0 <= step_index < num_steps:
        raise ValueError(f"step_index must be in [0, {num_steps}), got {step_index}.")
    return 1.0 if num_steps == 1 else step_index / (num_steps - 1)


@dataclass(frozen=True)
class ConstantLambda:
    """Use one diffusion-model weight throughout denoising."""

    value: float = 0.95
    name: str = "constant"

    def __post_init__(self) -> None:
        _validate_weight(self.value, "value")

    def __call__(self, step_index: int, num_steps: int) -> float:
        _progress(step_index, num_steps)
        return float(self.value)


@dataclass(frozen=True)
class LinearLambda:
    """Linearly interpolate from ``start`` at noise to ``end`` at the image."""

    start: float = 1.0
    end: float = 0.9
    name: str = "linear"

    def __post_init__(self) -> None:
        _validate_weight(self.start, "start")
        _validate_weight(self.end, "end")

    def __call__(self, step_index: int, num_steps: int) -> float:
        progress = _progress(step_index, num_steps)
        return float(self.start + progress * (self.end - self.start))


@dataclass(frozen=True)
class CosineLambda:
    """Cosine interpolation from ``start`` at noise to ``end`` at the image."""

    start: float = 1.0
    end: float = 0.9
    name: str = "cosine"

    def __post_init__(self) -> None:
        _validate_weight(self.start, "start")
        _validate_weight(self.end, "end")

    def __call__(self, step_index: int, num_steps: int) -> float:
        progress = _progress(step_index, num_steps)
        interpolation = 0.5 - 0.5 * math.cos(math.pi * progress)
        return float(self.start + interpolation * (self.end - self.start))


@dataclass(frozen=True)
class PowerDecayLambda:
    """Decay as ``1 - progress**power`` from diffusion to the GSM prior."""

    power: float = 5.0
    name: str = "power_decay"

    def __post_init__(self) -> None:
        if self.power <= 0:
            raise ValueError(f"power must be positive, got {self.power}.")

    def __call__(self, step_index: int, num_steps: int) -> float:
        progress = _progress(step_index, num_steps)
        return float(1.0 - progress**self.power)


@dataclass(frozen=True)
class ConstantGamma:
    """Use a constant additive filtered-prior guidance strength."""

    value: float = 0.0
    name: str = "constant"

    def __post_init__(self) -> None:
        _validate_nonnegative(self.value, "value")

    def __call__(self, step_index: int, num_steps: int) -> float:
        _progress(step_index, num_steps)
        return float(self.value)


@dataclass(frozen=True)
class PowerGrowthGamma:
    """Increase additive guidance from zero to ``maximum`` near clean time."""

    maximum: float = 0.01
    power: float = 5.0
    name: str = "power_growth"

    def __post_init__(self) -> None:
        _validate_nonnegative(self.maximum, "maximum")
        if self.power <= 0:
            raise ValueError(f"power must be positive, got {self.power}.")

    def __call__(self, step_index: int, num_steps: int) -> float:
        progress = _progress(step_index, num_steps)
        return float(self.maximum * progress**self.power)


@dataclass(frozen=True)
class ConstantFieldCoefficient:
    """A non-negative coefficient for one independently weighted vector field."""

    value: float = 1.0
    name: str = "constant"

    def __post_init__(self) -> None:
        _validate_nonnegative(self.value, "value")

    def __call__(self, step_index: int, num_steps: int) -> float:
        _progress(step_index, num_steps)
        return float(self.value)


@dataclass(frozen=True)
class LinearFieldCoefficient:
    """Linearly vary an independently weighted vector-field coefficient."""

    start: float = 1.0
    end: float = 1.0
    name: str = "linear"

    def __post_init__(self) -> None:
        _validate_nonnegative(self.start, "start")
        _validate_nonnegative(self.end, "end")

    def __call__(self, step_index: int, num_steps: int) -> float:
        progress = _progress(step_index, num_steps)
        return float(self.start + progress * (self.end - self.start))
