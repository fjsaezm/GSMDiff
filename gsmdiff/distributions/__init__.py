"""Analytic super-Gaussian distributions."""

from gsmdiff.distributions.base import SuperGaussian
from gsmdiff.distributions.filter_product import FirstDifferenceFilterProduct
from gsmdiff.distributions.super_gaussian import (
    Gaussian,
    GeneralizedGaussian,
    HyperbolicSecant,
    Laplace,
    SmoothedPowerLaw,
)

__all__ = [
    "FirstDifferenceFilterProduct",
    "Gaussian",
    "GeneralizedGaussian",
    "HyperbolicSecant",
    "Laplace",
    "SmoothedPowerLaw",
    "SuperGaussian",
]
