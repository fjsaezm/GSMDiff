"""Sample a CelebA-HQ diffusion/GSM geometric density mixture."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

from gsmdiff.distributions import FirstDifferenceFilterProduct
from gsmdiff.geometric import (
    DiffusersScoreModel,
    FilteredGSMNoisyScore,
    GeometricDiffusionSampler,
)
from gsmdiff.utils.tracking import initialize_wandb
from gsmdiff.visualization import draw_high_pass_contours, high_pass_coefficients


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def _configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed all RNGs and request deterministic PyTorch kernels."""
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def _draw_image(axis, sample: Tensor, contour_cfg: DictConfig | None = None) -> None:
    image = sample.detach().float().cpu().clamp(-1, 1)
    if image.shape[0] == 1:
        axis.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
    elif image.shape[0] == 3:
        axis.imshow(((image.permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1))
    else:
        raise ValueError("Image plots support one or three channels.")
    if contour_cfg is not None:
        threshold = contour_cfg.get("threshold")
        draw_high_pass_contours(
            axis,
            image,
            quantile=float(contour_cfg.quantile),
            threshold=None if threshold is None else float(threshold),
            color=str(contour_cfg.color),
            opacity=float(contour_cfg.opacity),
        )
    axis.axis("off")


def _save_comparison_grid(
    geometric: Tensor,
    baseline: Tensor | None,
    path: Path,
    *,
    max_samples: int,
    contour_cfg: DictConfig | None = None,
    geometric_label: str = "Geometric diffusion + filtered GSM",
    baseline_label: str = "Diffusion baseline (lambda=1)",
) -> None:
    import matplotlib.pyplot as plt

    count = min(max_samples, len(geometric))
    groups = [(geometric_label, geometric)]
    if baseline is not None:
        count = min(count, len(baseline))
        groups.insert(0, (baseline_label, baseline))
    figure_width = max(4.5, 2.5 * count)
    figure, axes = plt.subplots(
        len(groups), count, figsize=(figure_width, 2.8 * len(groups)), squeeze=False
    )
    for row, (label, samples) in enumerate(groups):
        for column in range(count):
            _draw_image(axes[row, column], samples[column], contour_cfg)
        axes[row, 0].set_title(label, loc="left", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _coefficients(samples: Tensor) -> Tensor:
    return torch.cat([high_pass_coefficients(sample) for sample in samples]).float().cpu()


def _standardize(values: Tensor) -> Tensor:
    centered = values - values.mean()
    return centered / centered.std().clamp_min(torch.finfo(values.dtype).eps)


def _coefficient_metrics(samples: Tensor, prefix: str, near_zero: float) -> dict[str, float]:
    raw = _coefficients(samples)
    standardized = _standardize(raw)
    second_moment = standardized.square().mean()
    excess_kurtosis = standardized.pow(4).mean() / second_moment.square() - 3.0
    return {
        f"filters/{prefix}_mean_absolute": float(raw.abs().mean()),
        f"filters/{prefix}_std": float(raw.std()),
        f"filters/{prefix}_standardized_near_zero_fraction": float(
            (standardized.abs() < near_zero).float().mean()
        ),
        f"filters/{prefix}_standardized_excess_kurtosis": float(excess_kurtosis),
    }


def _prior_energy_metrics(
    samples: Tensor,
    prior: FirstDifferenceFilterProduct,
    prefix: str,
) -> dict[str, float]:
    samples = samples.detach().float().cpu()
    filter_energy = prior.filter_energy(samples)
    total_energy = prior.energy(samples)
    channels, height, width = samples.shape[1:]
    coefficient_count = channels * (height * (width - 1) + (height - 1) * width)
    return {
        f"ggsm_energy/{prefix}_filter_sum_mean": float(filter_energy.mean()),
        f"ggsm_energy/{prefix}_filter_sum_std": float(filter_energy.std(correction=0)),
        f"ggsm_energy/{prefix}_filter_mean_per_coefficient": float(
            filter_energy.mean() / coefficient_count
        ),
        f"ggsm_energy/{prefix}_total_mean": float(total_energy.mean()),
        f"ggsm_energy/{prefix}_coefficient_count_per_image": coefficient_count,
    }


def _save_sparsity_plot(
    geometric: Tensor,
    baseline: Tensor | None,
    path: Path,
    cfg: DictConfig,
    geometric_label: str = "Geometric",
    baseline_label: str = "Diffusion baseline",
) -> None:
    import matplotlib.pyplot as plt

    series = [(geometric_label, _standardize(_coefficients(geometric)), "tab:orange")]
    if baseline is not None:
        series.insert(0, (baseline_label, _standardize(_coefficients(baseline)), "tab:blue"))

    limit_values = torch.cat([values.abs() for _, values, _ in series])
    limit = float(torch.quantile(limit_values, float(cfg.range_quantile)))
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for label, values, color in series:
        axes[0].hist(
            values.numpy(),
            bins=int(cfg.bins),
            range=(-limit, limit),
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
            color=color,
        )
        thresholds = torch.linspace(0, float(cfg.cdf_max_threshold), 200)
        fractions = torch.stack(
            [(values.abs() <= threshold).float().mean() for threshold in thresholds]
        )
        axes[1].plot(thresholds.numpy(), fractions.numpy(), label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Standardized signed first-difference coefficient")
    axes[0].set_ylabel("Density (log scale)")
    axes[0].set_title("Peak and heavy tails")
    axes[1].set_xlabel("Absolute standardized coefficient threshold")
    axes[1].set_ylabel("Fraction of coefficients below threshold")
    axes[1].set_title("Near-zero concentration (sparsity)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _aggregate_step_metrics(
    steps,
    *,
    cg_tolerance: float,
    cg_max_iterations: int,
) -> dict[str, float]:
    if not steps:
        return {}
    cg_iterations = np.asarray([step.cg_iterations for step in steps])
    cg_total_iterations = np.asarray([step.cg_total_iterations for step in steps])
    cg_residuals = np.asarray([step.cg_maximum_relative_residual for step in steps])
    mean_field_iterations = np.asarray([step.mean_field_iterations for step in steps])
    cg_solves_at_limit = np.asarray([step.cg_solves_at_iteration_limit for step in steps])
    diffusion_norms = np.asarray([step.diffusion_score_norm for step in steps])
    weighted_prior_norms = np.asarray([step.weighted_prior_score_norm for step in steps])
    norm_ratios = weighted_prior_norms / np.maximum(diffusion_norms, np.finfo(np.float64).eps)
    return {
        "sampling/lambda_start": steps[0].blend,
        "sampling/lambda_end": steps[-1].blend,
        "sampling/prior_weight_start": steps[0].prior_weight,
        "sampling/prior_weight_end": steps[-1].prior_weight,
        "sampling/cg_mean_iterations": float(np.mean(cg_iterations)),
        "sampling/cg_mean_total_iterations": float(np.mean(cg_total_iterations)),
        "sampling/mean_field_iterations": float(np.max(mean_field_iterations)),
        "sampling/cg_total_solves_at_iteration_limit": int(np.sum(cg_solves_at_limit)),
        "sampling/cg_fraction_at_iteration_limit": float(
            np.mean(cg_iterations >= cg_max_iterations)
        ),
        "sampling/cg_fraction_above_tolerance": float(np.mean(cg_residuals > cg_tolerance)),
        "sampling/cg_p95_relative_residual": float(np.quantile(cg_residuals, 0.95)),
        "sampling/cg_max_relative_residual": float(np.max(cg_residuals)),
        "sampling/diffusion_score_rms_mean": float(np.mean(diffusion_norms)),
        "sampling/prior_score_rms_mean": float(np.mean([step.prior_score_norm for step in steps])),
        "sampling/weighted_prior_score_rms_mean": float(np.mean(weighted_prior_norms)),
        "sampling/weighted_prior_to_diffusion_rms_ratio_mean": float(np.mean(norm_ratios)),
        "sampling/weighted_prior_to_diffusion_rms_ratio_p95": float(np.quantile(norm_ratios, 0.95)),
        "sampling/score_cosine_similarity_mean": float(
            np.mean([step.score_cosine_similarity for step in steps])
        ),
    }


def _blend_label(cfg: DictConfig) -> str:
    def compact(value) -> str:
        return f"{float(value):g}".replace(".", "p")

    if str(cfg.name) == "constant":
        return f"lambda{compact(cfg.value)}"
    if str(cfg.name) == "power_decay":
        return f"power_decay_{compact(cfg.power)}"
    return f"{cfg.name}_{compact(cfg.start)}_to_{compact(cfg.end)}"


def _save_score_weight_schedule(steps, path: Path) -> None:
    import matplotlib.pyplot as plt

    additive = steps and steps[0].score_combination == "additive"
    values = np.asarray([step.prior_weight if additive else step.blend for step in steps])
    progress = np.linspace(0.0, 1.0, len(values))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(progress, values, linewidth=2.2)
    axis.set_xlabel("Normalized reverse-process progress")
    axis.set_ylabel(r"Prior guidance $\gamma$" if additive else r"Diffusion weight $\lambda$")
    axis.set_xlim(0.0, 1.0)
    upper = max(1.0 if not additive else 0.0, float(values.max()))
    margin = max(0.02, 0.02 * upper)
    axis.set_ylim(-margin, upper + margin)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_snapshots(
    snapshots,
    final_samples: Tensor,
    run_dir: Path,
    *,
    max_samples: int,
    contour_cfg: DictConfig,
) -> None:
    checkpoint_root = run_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final_step = snapshots[-1].completed_steps if snapshots else None
    for snapshot in snapshots:
        checkpoint_dir = checkpoint_root / f"step_{snapshot.completed_steps:04d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        display_samples = (
            final_samples if snapshot.completed_steps == final_step else snapshot.denoised
        )
        _save_comparison_grid(
            display_samples,
            None,
            checkpoint_dir / "samples.png",
            max_samples=max_samples,
        )
        _save_comparison_grid(
            display_samples,
            None,
            checkpoint_dir / "samples_with_high_pass_contours.png",
            max_samples=max_samples,
            contour_cfg=contour_cfg,
        )


def run(cfg: DictConfig) -> Path:
    seed = int(cfg.runtime.seed)
    _configure_reproducibility(seed, bool(cfg.runtime.deterministic))

    device = _resolve_device(str(cfg.runtime.device))
    dtype = getattr(torch, str(cfg.runtime.dtype))
    diffusion = DiffusersScoreModel.from_pretrained(
        str(cfg.model.id),
        device=device,
        dtype=dtype,
        revision=cfg.model.get("revision"),
        use_safetensors=cfg.model.get("use_safetensors"),
    )
    coefficient_distribution = instantiate(cfg.distribution)
    prior = FirstDifferenceFilterProduct(
        coefficient_distribution,
        mean_scale=float(cfg.prior.mean_scale),
    )
    noisy_prior = FilteredGSMNoisyScore(
        prior,
        cg_max_iterations=int(cfg.prior.cg.max_iterations),
        cg_tolerance=float(cfg.prior.cg.tolerance),
        numerical_epsilon=float(cfg.prior.cg.numerical_epsilon),
        warm_start=bool(cfg.prior.cg.warm_start),
        mean_field_iterations=int(cfg.prior.mean_field_iterations),
    )
    blend_schedule = instantiate(cfg.blend)
    guidance_schedule = instantiate(cfg.guidance)
    sampler = GeometricDiffusionSampler(
        diffusion,
        noisy_prior,
        blend_schedule,
        guidance_schedule=guidance_schedule,
        score_combination=str(cfg.sampling.score_combination),
        method=str(cfg.sampling.method),
    )

    sample_size = diffusion.model.config.sample_size
    height, width = (
        (int(sample_size), int(sample_size))
        if isinstance(sample_size, int)
        else tuple(int(value) for value in sample_size)
    )
    shape = (
        int(cfg.sampling.batch_size),
        int(diffusion.model.config.in_channels),
        height,
        width,
    )
    initial_generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(shape, device=device, dtype=dtype, generator=initial_generator)
    initial_noise *= float(getattr(diffusion.scheduler, "init_noise_sigma", 1.0))
    geometric_generator = torch.Generator(device=device).manual_seed(seed + 1)
    snapshot_every = cfg.checkpoints.get("every_steps")
    result = sampler.sample(
        batch_size=shape[0],
        num_inference_steps=int(cfg.sampling.num_inference_steps),
        device=device,
        dtype=dtype,
        generator=geometric_generator,
        initial_noise=initial_noise,
        snapshot_every=None if snapshot_every is None else int(snapshot_every),
    )
    geometric = result.samples.detach().float().cpu()

    baseline = None
    if bool(cfg.comparison.generate_diffusion_baseline):
        baseline_generator = torch.Generator(device=device).manual_seed(seed + 1)
        baseline_result = sampler.sample(
            batch_size=shape[0],
            num_inference_steps=int(cfg.sampling.num_inference_steps),
            device=device,
            dtype=dtype,
            generator=baseline_generator,
            initial_noise=initial_noise,
            blend_override=(1.0 if str(cfg.sampling.score_combination) == "geometric" else None),
            guidance_override=(0.0 if str(cfg.sampling.score_combination) == "additive" else None),
        )
        baseline = baseline_result.samples.detach().float().cpu()

    run_name = cfg.run.get("name") or (
        f"celebahq_geometric_{cfg.distribution.name}_{_blend_label(cfg.blend)}_seed{seed}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(str(cfg.run.output_dir)) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    torch.save(geometric, run_dir / "geometric_samples.pt")
    if baseline is not None:
        torch.save(baseline, run_dir / "diffusion_baseline_samples.pt")
    (run_dir / "step_diagnostics.json").write_text(
        json.dumps([asdict(step) for step in result.steps], indent=2)
    )

    max_samples = int(cfg.image.num_preview)
    _save_comparison_grid(
        geometric,
        baseline,
        run_dir / "samples.png",
        max_samples=max_samples,
    )
    _save_comparison_grid(
        geometric,
        baseline,
        run_dir / "samples_with_high_pass_contours.png",
        max_samples=max_samples,
        contour_cfg=cfg.image.high_pass_contours,
    )
    _save_score_weight_schedule(result.steps, run_dir / "score_weight_schedule.png")
    # Keep the historical filename for report scripts and existing consumers.
    _save_score_weight_schedule(result.steps, run_dir / "lambda_schedule.png")
    _save_snapshots(
        result.snapshots,
        geometric,
        run_dir,
        max_samples=max_samples,
        contour_cfg=cfg.image.high_pass_contours,
    )
    _save_sparsity_plot(
        geometric,
        baseline,
        run_dir / "filter_sparsity.png",
        cfg.analysis.histogram,
    )

    near_zero = float(cfg.analysis.near_zero_threshold)
    metrics = {
        "samples/count": len(geometric),
        "samples/geometric_mean": float(geometric.mean()),
        "samples/geometric_std": float(geometric.std()),
        **_coefficient_metrics(geometric, "geometric", near_zero),
        **_prior_energy_metrics(geometric, prior, "geometric"),
        **_aggregate_step_metrics(
            result.steps,
            cg_tolerance=float(cfg.prior.cg.tolerance),
            cg_max_iterations=int(cfg.prior.cg.max_iterations),
        ),
    }
    if baseline is not None:
        metrics.update(
            {
                "samples/baseline_mean": float(baseline.mean()),
                "samples/baseline_std": float(baseline.std()),
                **_coefficient_metrics(baseline, "baseline", near_zero),
                **_prior_energy_metrics(baseline, prior, "baseline"),
            }
        )
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    tracking_run = initialize_wandb(cfg, run_dir)
    tracking_run.log(metrics)
    wandb = __import__("wandb")
    tracking_run.log(
        {
            "samples": wandb.Image(str(run_dir / "samples.png")),
            "high_pass_contours": wandb.Image(str(run_dir / "samples_with_high_pass_contours.png")),
            "lambda_schedule": wandb.Image(str(run_dir / "lambda_schedule.png")),
            "score_weight_schedule": wandb.Image(str(run_dir / "score_weight_schedule.png")),
            "filter_sparsity": wandb.Image(str(run_dir / "filter_sparsity.png")),
        }
    )
    tracking_run.finish()
    print(f"Saved {len(geometric)} geometric CelebA-HQ samples to {run_dir}")
    print(json.dumps(metrics, indent=2))
    return run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="geometric_celebahq")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
