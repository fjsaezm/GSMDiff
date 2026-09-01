from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def test_every_distribution_and_sampler_composes():
    for distribution in (
        "gaussian",
        "generalized_gaussian",
        "laplace",
        "hyperbolic_secant",
        "log",
    ):
        for sampler in ("ula", "mala"):
            with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
                cfg = compose(
                    config_name="config",
                    overrides=[f"distribution={distribution}", f"sampler={sampler}"],
                )
            assert instantiate(cfg.distribution) is not None
            coefficient_distribution = instantiate(cfg.distribution)
            assert (
                instantiate(
                    cfg.prior,
                    coefficient_distribution=coefficient_distribution,
                )
                is not None
            )
            assert instantiate(cfg.sampler) is not None


def test_geometric_celebahq_config_composes_without_downloading_model():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="geometric_celebahq")
    assert instantiate(cfg.distribution) is not None
    assert instantiate(cfg.blend)(0, 2) == 0.95
    assert instantiate(cfg.guidance)(0, 2) == 0.0
    assert cfg.sampling.score_combination == "geometric"
    assert cfg.model.id == "google/ddpm-celebahq-256"


def test_power_decay_blend_config_composes_with_exact_endpoints():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="geometric_celebahq", overrides=["blend=power_decay"])
    schedule = instantiate(cfg.blend)
    assert schedule(0, 1000) == 1.0
    assert schedule(999, 1000) == 0.0


def test_geometric_flow_config_has_independent_field_coefficients():
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="geometric_flow_celeba")
    flow_weight = instantiate(cfg.flow_weight)
    prior_weight = instantiate(cfg.prior_weight)
    assert flow_weight(0, 2) == 1.0
    assert prior_weight(0, 2) == 0.05
    assert flow_weight(0, 2) + prior_weight(0, 2) != 1.0
    assert cfg.model.input_height == 128
    assert cfg.sampling.method == "euler"
