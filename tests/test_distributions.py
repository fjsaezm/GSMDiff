import pytest
import torch

from gsmdiff.distributions import (
    FirstDifferenceFilterProduct,
    Gaussian,
    GeneralizedGaussian,
    HyperbolicSecant,
    Laplace,
    SmoothedPowerLaw,
)


@pytest.mark.parametrize(
    "distribution",
    [
        Gaussian(scale=1.3),
        GeneralizedGaussian(shape=0.8, scale=1.1, smoothing=0.05),
        Laplace(scale=0.8, smoothing=0.05),
        HyperbolicSecant(scale=1.2),
        SmoothedPowerLaw(scale=0.2, tail_exponent=2.5),
    ],
)
def test_score_matches_autograd_and_gsm_identity(distribution):
    value = torch.tensor([[-1.3, -0.2, 0.0, 0.7]], dtype=torch.float64, requires_grad=True)
    log_density = distribution.log_density(value).sum()
    autodiff_score = torch.autograd.grad(log_density, value)[0]
    assert torch.allclose(distribution.score(value), autodiff_score, atol=1e-9)
    distribution.check_identity(value.detach(), atol=1e-9)


def test_log_model_must_be_a_proper_distribution():
    with pytest.raises(ValueError, match="proper distribution"):
        SmoothedPowerLaw(tail_exponent=1.0)


def test_exact_laplace_identity_handles_the_origin():
    Laplace(smoothing=0.0).check_identity(torch.tensor([[0.0, 1.0]]))


def test_filter_product_score_matches_autograd():
    coefficient_distribution = HyperbolicSecant(scale=1.2)
    target = FirstDifferenceFilterProduct(coefficient_distribution, mean_scale=7.0)
    value = torch.randn(2, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    autodiff_score = torch.autograd.grad(target.log_density(value).sum(), value)[0]
    assert torch.allclose(target.score(value), autodiff_score, atol=1e-10)


def test_filter_product_mean_prior_resolves_constant_null_space():
    target = FirstDifferenceFilterProduct(Gaussian(), mean_scale=5.0)
    value = torch.zeros(2, 1, 3, 4, dtype=torch.float64)
    shifted = value + 2.0
    expected_change = torch.full((2,), -0.5 * (2.0 / 5.0) ** 2, dtype=torch.float64)
    assert torch.allclose(target.log_density(shifted) - target.log_density(value), expected_change)
    assert torch.allclose(target.score(shifted), torch.full_like(shifted, -2.0 / (25.0 * 12)))


def test_filter_product_energy_is_sum_of_filter_potentials_and_dc_term():
    target = FirstDifferenceFilterProduct(Gaussian(scale=2.0), mean_scale=5.0)
    value = torch.tensor([[[[0.0, 2.0], [4.0, 6.0]]]], dtype=torch.float64)
    # Horizontal differences: -2, -2; vertical differences: -4, -4.
    expected_filter = 2 * (2.0**2 / 8.0) + 2 * (4.0**2 / 8.0)
    expected_dc = 0.5 * (3.0 / 5.0) ** 2
    assert target.filter_energy(value).item() == pytest.approx(expected_filter)
    assert target.energy(value).item() == pytest.approx(expected_filter + expected_dc)
    assert target.energy(value).item() == pytest.approx(-target.log_density(value).item())
