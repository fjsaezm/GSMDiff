from types import SimpleNamespace

import pytest
import torch

from gsmdiff.distributions import FirstDifferenceFilterProduct, Gaussian
from gsmdiff.geometric import (
    ConstantGamma,
    ConstantFieldCoefficient,
    ConstantLambda,
    CosineLambda,
    DiffusersScoreModel,
    FilteredGSMNoisyScore,
    GeometricDiffusionSampler,
    GeometricFlowSampler,
    LinearLambda,
    PowerDecayLambda,
    PowerGrowthGamma,
)
from gsmdiff.geometric.calibration import (
    coefficient_standard_deviation,
    distribution_scale_from_standard_deviation,
)


class _FixedModel:
    def __init__(self, output):
        self.output = output

    def __call__(self, sample, timestep):
        return SimpleNamespace(sample=torch.full_like(sample, self.output))


class _Scheduler:
    def __init__(self, prediction_type):
        self.alphas_cumprod = torch.tensor([0.64], dtype=torch.float64)
        self.config = SimpleNamespace(prediction_type=prediction_type)

    @staticmethod
    def scale_model_input(sample, timestep):
        return sample


@pytest.mark.parametrize("prediction_type", ["epsilon", "sample", "v_prediction"])
def test_diffusers_adapter_converts_prediction_to_consistent_score(prediction_type):
    noisy = torch.full((1, 1, 2, 2), 0.25, dtype=torch.float64)
    alpha = 0.8
    sigma = 0.6
    target_denoised = 0.5
    epsilon = (0.25 - alpha * target_denoised) / sigma
    if prediction_type == "epsilon":
        model_output = epsilon
    elif prediction_type == "sample":
        model_output = target_denoised
    else:
        model_output = alpha * epsilon - sigma * target_denoised

    adapter = DiffusersScoreModel(_FixedModel(model_output), _Scheduler(prediction_type))
    prediction = adapter.predict(noisy, 0)
    assert torch.allclose(prediction.denoised, torch.full_like(noisy, target_denoised))
    assert torch.allclose(prediction.score, torch.full_like(noisy, -epsilon / sigma))


def test_filtered_noisy_score_matches_dense_gaussian_conditioning():
    prior = FirstDifferenceFilterProduct(Gaussian(scale=1.3), mean_scale=4.0)
    noisy_score = FilteredGSMNoisyScore(
        prior, cg_max_iterations=50, cg_tolerance=1e-12, numerical_epsilon=1e-14
    )
    noisy = torch.tensor([[[[0.2, -0.7, 0.5], [1.1, -0.3, 0.4]]]], dtype=torch.float64)
    denoised = torch.zeros_like(noisy)
    alpha = torch.tensor(0.8, dtype=torch.float64)
    sigma = torch.tensor(0.6, dtype=torch.float64)

    horizontal_weight, vertical_weight = noisy_score._weights(denoised)
    dimension = noisy[0].numel()
    basis = torch.eye(dimension, dtype=torch.float64).reshape(dimension, *noisy.shape[1:])
    precision_columns = noisy_score._precision_action(
        basis, horizontal_weight, vertical_weight
    ).reshape(dimension, dimension)
    precision = precision_columns.transpose(0, 1)
    system = alpha.square() * torch.eye(dimension, dtype=torch.float64) + sigma.square() * precision
    dense_solution = torch.linalg.solve(system, noisy.flatten())
    expected_mean = (alpha * dense_solution).reshape_as(noisy)
    expected_score = -(noisy - alpha * expected_mean) / sigma.square()

    prediction = noisy_score.predict(noisy, denoised, alpha, sigma)
    assert prediction.diagnostics.maximum_relative_residual < 1e-10
    assert torch.allclose(prediction.conditional_mean, expected_mean, atol=1e-10)
    assert torch.allclose(prediction.score, expected_score, atol=1e-10)


def test_filtered_noisy_score_accepts_previous_solution_as_exact_warm_start():
    prior = FirstDifferenceFilterProduct(Gaussian(scale=1.3), mean_scale=4.0)
    noisy_score = FilteredGSMNoisyScore(
        prior,
        cg_max_iterations=50,
        cg_tolerance=1e-12,
        numerical_epsilon=1e-14,
        warm_start=True,
    )
    noisy = torch.tensor([[[[0.2, -0.7], [1.1, -0.3]]]], dtype=torch.float64)
    denoised = torch.zeros_like(noisy)
    alpha = torch.tensor(0.8, dtype=torch.float64)
    sigma = torch.tensor(0.6, dtype=torch.float64)

    cold = noisy_score.predict(noisy, denoised, alpha, sigma)
    warm = noisy_score.predict(
        noisy,
        denoised,
        alpha,
        sigma,
        initial_solution=cold.linear_solution,
    )
    assert cold.diagnostics.iterations > 0
    assert warm.diagnostics.iterations == 0
    assert warm.diagnostics.maximum_relative_residual <= 1e-12
    assert torch.allclose(warm.score, cold.score, atol=1e-12)


def test_filtered_noisy_score_repeats_algorithm_2_mean_field_updates():
    class RecordingScore(FilteredGSMNoisyScore):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_inputs = []

        def _weights(self, denoised):
            self.weight_inputs.append(denoised.clone())
            return super()._weights(denoised)

    prior = FirstDifferenceFilterProduct(Gaussian(scale=1.3), mean_scale=4.0)
    noisy_score = RecordingScore(
        prior,
        cg_max_iterations=50,
        cg_tolerance=1e-12,
        numerical_epsilon=1e-14,
        mean_field_iterations=3,
    )
    noisy = torch.tensor([[[[0.2, -0.7], [1.1, -0.3]]]], dtype=torch.float64)
    initial_estimate = torch.zeros_like(noisy)
    prediction = noisy_score.predict(
        noisy,
        initial_estimate,
        torch.tensor(0.8, dtype=torch.float64),
        torch.tensor(0.6, dtype=torch.float64),
    )

    assert len(noisy_score.weight_inputs) == 3
    assert torch.equal(noisy_score.weight_inputs[0], initial_estimate)
    assert torch.allclose(noisy_score.weight_inputs[-1], prediction.conditional_mean, atol=1e-10)
    assert prediction.diagnostics.mean_field_iterations == 3
    assert prediction.diagnostics.total_iterations >= prediction.diagnostics.iterations


def test_singular_gsm_precision_is_rejected_with_actionable_error():
    from gsmdiff.distributions import Laplace

    prior = FirstDifferenceFilterProduct(Laplace(smoothing=0.0))
    noisy_score = FilteredGSMNoisyScore(prior)
    with pytest.raises(FloatingPointError, match="positive smoothing"):
        noisy_score.predict(
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 2, 2),
            torch.tensor(0.8),
            torch.tensor(0.6),
        )


def test_lambda_schedules_have_documented_endpoints():
    assert ConstantLambda(0.4)(0, 10) == pytest.approx(0.4)
    linear = LinearLambda(start=1.0, end=0.25)
    cosine = CosineLambda(start=1.0, end=0.25)
    for schedule in (linear, cosine):
        assert schedule(0, 11) == pytest.approx(1.0)
        assert schedule(10, 11) == pytest.approx(0.25)
    assert linear(5, 11) == pytest.approx(0.625)
    assert cosine(5, 11) == pytest.approx(0.625)
    power = PowerDecayLambda(power=5.0)
    assert power(0, 11) == pytest.approx(1.0)
    assert power(5, 11) == pytest.approx(1.0 - 0.5**5)
    assert power(10, 11) == pytest.approx(0.0)


def test_lambda_schedule_validates_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ConstantLambda(1.1)
    with pytest.raises(ValueError, match="power"):
        PowerDecayLambda(0.0)


def test_gamma_schedules_are_nonnegative_and_have_documented_endpoints():
    assert ConstantGamma(0.25)(3, 10) == pytest.approx(0.25)
    growth = PowerGrowthGamma(maximum=0.2, power=2.0)
    assert growth(0, 11) == 0.0
    assert growth(5, 11) == pytest.approx(0.05)
    assert growth(10, 11) == pytest.approx(0.2)
    with pytest.raises(ValueError, match="non-negative"):
        ConstantGamma(-0.1)


def test_sech_scale_calibration_uses_its_actual_standard_deviation():
    samples = torch.tensor([[[[0.0, 1.0], [2.0, 4.0]]]])
    standard_deviation, count = coefficient_standard_deviation([samples])
    coefficients = torch.tensor([-1.0, -2.0, -2.0, -3.0])
    assert count == 4
    assert standard_deviation == pytest.approx(float(coefficients.std(correction=0)))
    assert distribution_scale_from_standard_deviation(
        standard_deviation, "hyperbolic_secant"
    ) == pytest.approx(2.0 * standard_deviation / torch.pi)


class _SnapshotDiffusion:
    def __init__(self):
        self.model = SimpleNamespace(config=SimpleNamespace(sample_size=2, in_channels=1))
        self.scheduler = SimpleNamespace(
            alphas_cumprod=torch.tensor([0.9, 0.8, 0.7, 0.6]),
            init_noise_sigma=1.0,
        )

    def set_timesteps(self, num_inference_steps, device):
        assert num_inference_steps == 4
        self.scheduler.timesteps = torch.tensor([3, 2, 1, 0], device=device)

    def predict(self, sample, timestep):
        alpha_bar = self.scheduler.alphas_cumprod[int(timestep)].to(sample)
        alpha = alpha_bar.sqrt()
        sigma = (1.0 - alpha_bar).sqrt()
        return SimpleNamespace(
            score=torch.zeros_like(sample),
            denoised=sample / alpha,
            alpha=alpha,
            sigma=sigma,
        )


def test_additive_guidance_keeps_the_complete_diffusion_score():
    class OneStepDiffusion(_SnapshotDiffusion):
        def __init__(self):
            self.model = SimpleNamespace(config=SimpleNamespace(sample_size=1, in_channels=1))
            self.scheduler = SimpleNamespace(
                alphas_cumprod=torch.tensor([0.5]), init_noise_sigma=1.0
            )

        def set_timesteps(self, num_inference_steps, device):
            self.scheduler.timesteps = torch.tensor([0], device=device)

        def predict(self, sample, timestep):
            return SimpleNamespace(
                score=torch.full_like(sample, 2.0),
                denoised=torch.zeros_like(sample),
                alpha=torch.sqrt(torch.tensor(0.5)).to(sample),
                sigma=torch.sqrt(torch.tensor(0.5)).to(sample),
            )

    class ConstantPrior:
        warm_start = False

        def predict(self, noisy, denoised, alpha, sigma, initial_solution=None):
            return SimpleNamespace(
                score=torch.full_like(noisy, 3.0),
                linear_solution=torch.zeros_like(noisy),
                diagnostics=SimpleNamespace(
                    iterations=1,
                    maximum_relative_residual=0.0,
                    total_iterations=1,
                    mean_field_iterations=1,
                    solves_at_iteration_limit=0,
                ),
            )

    sampler = GeometricDiffusionSampler(
        OneStepDiffusion(),
        ConstantPrior(),
        ConstantLambda(0.1),  # Must be ignored in additive mode.
        guidance_schedule=ConstantGamma(0.5),
        score_combination="additive",
    )
    result = sampler.sample(
        batch_size=1,
        num_inference_steps=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        initial_noise=torch.zeros(1, 1, 1, 1),
    )
    expected_total_score = 2.0 + 0.5 * 3.0
    expected = 0.5 * expected_total_score / torch.sqrt(torch.tensor(0.5))
    assert result.samples.item() == pytest.approx(float(expected))
    assert result.steps[0].blend == 1.0
    assert result.steps[0].prior_weight == 0.5
    assert result.steps[0].weighted_prior_score_norm == pytest.approx(1.5)
    assert result.steps[0].score_cosine_similarity == pytest.approx(1.0)


def test_geometric_sampler_captures_requested_snapshots_on_cpu():
    sampler = GeometricDiffusionSampler(
        _SnapshotDiffusion(),
        SimpleNamespace(),
        ConstantLambda(1.0),
        method="reverse_sde",
    )
    result = sampler.sample(
        batch_size=1,
        num_inference_steps=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(12),
        initial_noise=torch.ones(1, 1, 2, 2),
        snapshot_every=2,
    )
    assert [snapshot.completed_steps for snapshot in result.snapshots] == [2, 4]
    assert [snapshot.timestep for snapshot in result.snapshots] == [2, 0]
    assert all(snapshot.state.device.type == "cpu" for snapshot in result.snapshots)
    assert torch.equal(result.snapshots[-1].state, result.samples.cpu())


def test_geometric_sampler_rejects_nonpositive_snapshot_interval():
    sampler = GeometricDiffusionSampler(
        _SnapshotDiffusion(),
        SimpleNamespace(),
        ConstantLambda(1.0),
    )
    with pytest.raises(ValueError, match="snapshot_every"):
        sampler.sample(
            batch_size=1,
            num_inference_steps=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
            snapshot_every=0,
        )


def test_reverse_sde_uses_ddpm_posterior_variance():
    class TwoStepDiffusion(_SnapshotDiffusion):
        def __init__(self):
            self.model = SimpleNamespace(config=SimpleNamespace(sample_size=1, in_channels=1))
            self.scheduler = SimpleNamespace(
                alphas_cumprod=torch.tensor([0.8, 0.6]),
                init_noise_sigma=1.0,
            )

        def set_timesteps(self, num_inference_steps, device):
            assert num_inference_steps == 2
            self.scheduler.timesteps = torch.tensor([1, 0], device=device)

    seed = 91
    expected_generator = torch.Generator().manual_seed(seed)
    first_noise = torch.randn((1, 1, 1, 1), generator=expected_generator)
    expected = 1.0 / torch.sqrt(torch.tensor(0.75))
    expected = expected + torch.sqrt(torch.tensor(0.125)) * first_noise
    expected = expected / torch.sqrt(torch.tensor(0.8))

    sampler = GeometricDiffusionSampler(
        TwoStepDiffusion(),
        SimpleNamespace(),
        ConstantLambda(1.0),
        method="reverse_sde",
    )
    result = sampler.sample(
        batch_size=1,
        num_inference_steps=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(seed),
        initial_noise=torch.ones(1, 1, 1, 1),
    )
    assert torch.allclose(result.samples, expected.reshape_as(result.samples))


def test_algorithm_five_uses_independent_coefficients_inside_fixed_point_loop():
    class ConstantFlow:
        sample_size = 1
        in_channels = 1

        @staticmethod
        def velocity(sample, time):
            return torch.full_like(sample, 2.0)

    class EstimateDependentPrior:
        mean_field_iterations = 2
        warm_start = False

        @staticmethod
        def predict_flow_velocity(state, clean_estimate, time, initial_solution=None):
            del time, initial_solution
            velocity = clean_estimate + 1.0
            return SimpleNamespace(
                velocity=velocity,
                linear_solution=torch.zeros_like(state),
                diagnostics=SimpleNamespace(
                    iterations=1,
                    maximum_relative_residual=0.0,
                    solves_at_iteration_limit=0,
                ),
            )

    sampler = GeometricFlowSampler(
        ConstantFlow(),
        EstimateDependentPrior(),
        ConstantFieldCoefficient(2.0),
        ConstantFieldCoefficient(3.0),
    )
    result = sampler.sample(
        batch_size=1,
        num_inference_steps=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        initial_noise=torch.zeros(1, 1, 1, 1),
    )
    # refinement 1: v=2*2 + 3*1=7, xhat=7
    # refinement 2: v=2*2 + 3*(7+1)=28, then one Euler step of size one.
    assert result.samples.item() == pytest.approx(28.0)
    assert result.steps[0].flow_coefficient == 2.0
    assert result.steps[0].prior_coefficient == 3.0
    assert result.steps[0].cg_total_iterations == 2


def test_filtered_gsm_flow_velocity_satisfies_algorithm_five_formula():
    prior = FirstDifferenceFilterProduct(Gaussian(scale=1.0), mean_scale=2.0)
    predictor = FilteredGSMNoisyScore(prior, cg_max_iterations=100, cg_tolerance=1e-7)
    state = torch.tensor([[[[0.2, -0.1], [0.4, 0.3]]]], dtype=torch.float64)
    time = torch.tensor(0.4, dtype=torch.float64)
    prediction = predictor.predict_flow_velocity(state, torch.zeros_like(state), time)
    horizontal_weight, vertical_weight = predictor._weights(torch.zeros_like(state))
    expected = time * prediction.linear_solution - (1.0 - time) * predictor._precision_action(
        prediction.linear_solution, horizontal_weight, vertical_weight
    )
    assert torch.allclose(prediction.velocity, expected, atol=1e-8)
    action = time.square() * prediction.linear_solution + (1.0 - time).square() * (
        predictor._precision_action(prediction.linear_solution, horizontal_weight, vertical_weight)
    )
    assert torch.allclose(action, state, atol=1e-6)


def test_flow_only_override_is_the_pretrained_flow_baseline():
    class UnitFlow:
        sample_size = 1
        in_channels = 1

        @staticmethod
        def velocity(sample, time):
            return torch.ones_like(sample)

    sampler = GeometricFlowSampler(
        UnitFlow(),
        SimpleNamespace(),
        ConstantFieldCoefficient(0.2),
        ConstantFieldCoefficient(0.3),
    )
    result = sampler.sample(
        batch_size=1,
        num_inference_steps=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        initial_noise=torch.zeros(1, 1, 1, 1),
        flow_coefficient_override=1.0,
        prior_coefficient_override=0.0,
    )
    assert result.samples.item() == pytest.approx(1.0)
    assert all(step.prior_coefficient == 0.0 for step in result.steps)
