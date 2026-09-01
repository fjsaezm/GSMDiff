import torch

from gsmdiff.distributions import Gaussian
from gsmdiff.samplers import MALASampler, ULASampler


def test_ula_recovers_standard_gaussian_moments():
    generator = torch.Generator().manual_seed(12)
    initial = torch.zeros(256, 1, dtype=torch.float64)
    result = ULASampler(step_size=0.05).sample(
        Gaussian(), initial, burn_in=500, num_draws=20, thinning=10, generator=generator
    )
    samples = result.samples.flatten()
    assert abs(float(samples.mean())) < 0.1
    assert abs(float(samples.std()) - 1.0) < 0.1
    assert result.diagnostics.acceptance_rate == 1.0


def test_mala_exact_and_path_integral_are_seed_equivalent_for_gaussian():
    initial = torch.linspace(-1, 1, 32, dtype=torch.float64).reshape(32, 1)
    exact_generator = torch.Generator().manual_seed(7)
    path_generator = torch.Generator().manual_seed(7)
    exact = MALASampler(step_size=0.2, density_ratio="exact").sample(
        Gaussian(), initial, burn_in=10, num_draws=3, thinning=2, generator=exact_generator
    )
    path = MALASampler(
        step_size=0.2, density_ratio="path_integral", quadrature_points=4
    ).sample(
        Gaussian(), initial, burn_in=10, num_draws=3, thinning=2, generator=path_generator
    )
    assert torch.allclose(exact.samples, path.samples, atol=1e-10)
    assert 0.0 <= exact.diagnostics.acceptance_rate <= 1.0

