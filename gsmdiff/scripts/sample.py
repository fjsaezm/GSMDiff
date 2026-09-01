"""Sample filter-product GSM image tensors with ULA or MALA."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gsmdiff.utils.tracking import initialize_wandb
from gsmdiff.visualization import (
    draw_high_pass_contours,
    high_pass_coefficients,
    high_pass_magnitude,
)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but CUDA is unavailable.")
    return device


def _save_preview(
    samples: torch.Tensor,
    path: Path,
    value_range: float,
    *,
    max_samples: int,
    contour_cfg: DictConfig | None = None,
) -> None:
    import matplotlib.pyplot as plt

    count = min(int(samples.shape[0]), max_samples)
    columns = min(4, count)
    rows = (count + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.4 * columns, 2.4 * rows),
        squeeze=False,
    )
    for axis in axes.flat:
        axis.axis("off")
    for axis, sample in zip(axes.flat, samples[:count]):
        _draw_sample(axis, sample, value_range, contour_cfg)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_sample(
    axis,
    sample: torch.Tensor,
    value_range: float,
    contour_cfg: DictConfig | None = None,
    contour_threshold: float | None = None,
) -> None:
    image = sample.detach().cpu().clamp(-value_range, value_range)
    if image.shape[0] == 1:
        axis.imshow(image[0], cmap="gray", vmin=-value_range, vmax=value_range)
    elif image.shape[0] == 3:
        display_image = (image.permute(1, 2, 0) / value_range + 1.0) / 2.0
        axis.imshow(display_image.clamp(0, 1))
    else:
        raise ValueError("Preview supports one or three image channels.")
    if contour_cfg is not None:
        threshold = contour_cfg.get("threshold")
        if contour_threshold is not None:
            threshold = contour_threshold
        draw_high_pass_contours(
            axis,
            image,
            quantile=float(contour_cfg.quantile),
            threshold=None if threshold is None else float(threshold),
            color=str(contour_cfg.color),
            opacity=float(contour_cfg.opacity),
        )
    axis.axis("off")


def _save_comparison(
    gsm_samples: torch.Tensor,
    baseline_samples: torch.Tensor,
    path: Path,
    value_range: float,
    *,
    max_samples: int,
    contour_cfg: DictConfig | None = None,
) -> None:
    import matplotlib.pyplot as plt

    count = min(len(gsm_samples), len(baseline_samples), max_samples)
    shared_contour_threshold = None
    if contour_cfg is not None and contour_cfg.get("threshold") is None:
        baseline_responses = torch.cat(
            [high_pass_magnitude(sample).reshape(-1) for sample in baseline_samples[:count]]
        )
        shared_contour_threshold = float(
            torch.quantile(baseline_responses, float(contour_cfg.quantile))
        )
    figure, axes = plt.subplots(2, count, figsize=(2.5 * count, 5.2), squeeze=False)
    for column in range(count):
        _draw_sample(
            axes[0, column],
            baseline_samples[column],
            value_range,
            contour_cfg,
            shared_contour_threshold,
        )
        _draw_sample(
            axes[1, column],
            gsm_samples[column],
            value_range,
            contour_cfg,
            shared_contour_threshold,
        )
    axes[0, 0].set_title("Gaussian baseline", loc="left")
    axes[1, 0].set_title("Filtered GSM target", loc="left")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _pooled_filter_coefficients(samples: torch.Tensor) -> torch.Tensor:
    return torch.cat([high_pass_coefficients(sample) for sample in samples]).float().cpu()


def _standardize(coefficients: torch.Tensor) -> torch.Tensor:
    centered = coefficients - coefficients.mean()
    scale = centered.std()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("Cannot standardize constant or non-finite filter coefficients.")
    return centered / scale


def _save_filtered_histogram(
    gsm_samples: torch.Tensor,
    baseline_samples: torch.Tensor,
    path: Path,
    cfg: DictConfig,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    gsm_coefficients = _pooled_filter_coefficients(gsm_samples)
    baseline_coefficients = _pooled_filter_coefficients(baseline_samples)
    if bool(cfg.standardize):
        gsm_coefficients = _standardize(gsm_coefficients)
        baseline_coefficients = _standardize(baseline_coefficients)

    combined_absolute = torch.cat((gsm_coefficients.abs(), baseline_coefficients.abs()))
    limit = float(torch.quantile(combined_absolute, float(cfg.range_quantile)))
    panel_count = 2 if bool(cfg.log_y) else 1
    figure, axes = plt.subplots(
        1,
        panel_count,
        figsize=(6.2 * panel_count, 4.5),
        squeeze=False,
        sharex=True,
    )
    histogram_options = {
        "bins": int(cfg.bins),
        "range": (-limit, limit),
        "density": True,
        "histtype": "step",
        "linewidth": 2.0,
    }
    prefix = "Standardized " if bool(cfg.standardize) else ""
    for axis in axes.flat:
        axis.hist(
            baseline_coefficients.numpy(),
            label="Gaussian (no GSM)",
            color="tab:blue",
            **histogram_options,
        )
        axis.hist(
            gsm_coefficients.numpy(),
            label="Filtered GSM target",
            color="tab:orange",
            **histogram_options,
        )
        axis.set_xlabel(f"{prefix}signed high-pass coefficient")
        axis.set_ylabel("Density")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0, 0].set_title("Central concentration")
    if bool(cfg.log_y):
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_title("Tail behavior (log density)")
    figure.suptitle("Horizontal and vertical filtered-image distributions")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=160)
    plt.close(figure)

    near_zero = float(cfg.near_zero_threshold)
    return {
        "filters/gsm_near_zero_fraction": float(
            (gsm_coefficients.abs() < near_zero).float().mean()
        ),
        "filters/gaussian_near_zero_fraction": float(
            (baseline_coefficients.abs() < near_zero).float().mean()
        ),
        "filters/gsm_mean_absolute_coefficient": float(gsm_coefficients.abs().mean()),
        "filters/gaussian_mean_absolute_coefficient": float(
            baseline_coefficients.abs().mean()
        ),
    }


def run(cfg: DictConfig) -> Path:
    seed = int(cfg.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = _resolve_device(str(cfg.runtime.device))
    dtype = getattr(torch, str(cfg.runtime.dtype))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    event_shape = tuple(int(size) for size in cfg.image.shape)
    initial = float(cfg.initialization.std) * torch.randn(
        (int(cfg.sampling.num_chains), *event_shape),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    coefficient_distribution = instantiate(cfg.distribution)
    target = instantiate(
        cfg.prior,
        coefficient_distribution=coefficient_distribution,
    )
    sampler = instantiate(cfg.sampler)
    result = sampler.sample(
        target,
        initial,
        num_draws=int(cfg.sampling.draws_per_chain),
        burn_in=int(cfg.sampling.burn_in),
        thinning=int(cfg.sampling.thinning),
        generator=generator,
    )
    samples = result.samples.flatten(0, 1).cpu()
    baseline_generator = torch.Generator(device="cpu")
    baseline_generator.manual_seed(seed + int(cfg.comparison.baseline.seed_offset))
    baseline_samples = float(cfg.comparison.baseline.std) * torch.randn(
        samples.shape,
        dtype=samples.dtype,
        generator=baseline_generator,
    )
    if bool(cfg.comparison.baseline.match_gsm_std):
        baseline_samples = (
            (baseline_samples - baseline_samples.mean())
            / baseline_samples.std()
            * samples.std()
        )

    run_name = cfg.run.get("name") or (
        f"{cfg.prior.name}_{cfg.distribution.name}_{cfg.sampler.name}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(str(cfg.run.output_dir)) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    torch.save(samples, run_dir / "samples.pt")
    torch.save(baseline_samples, run_dir / "gaussian_baseline_samples.pt")
    preview_count = int(cfg.image.num_preview)
    preview_range = float(cfg.image.preview_range)
    _save_preview(
        samples,
        run_dir / "samples.png",
        preview_range,
        max_samples=preview_count,
    )
    contour_path = run_dir / "samples_with_high_pass_contours.png"
    if bool(cfg.image.high_pass_contours.enabled):
        _save_preview(
            samples,
            contour_path,
            preview_range,
            max_samples=preview_count,
            contour_cfg=cfg.image.high_pass_contours,
        )
    comparison_path = run_dir / "gsm_vs_gaussian.png"
    comparison_contour_path = run_dir / "gsm_vs_gaussian_with_high_pass_contours.png"
    _save_comparison(
        samples,
        baseline_samples,
        comparison_path,
        preview_range,
        max_samples=preview_count,
    )
    if bool(cfg.image.high_pass_contours.enabled):
        _save_comparison(
            samples,
            baseline_samples,
            comparison_contour_path,
            preview_range,
            max_samples=preview_count,
            contour_cfg=cfg.image.high_pass_contours,
        )
    histogram_path = run_dir / "filtered_coefficients_histogram.png"
    filter_metrics = _save_filtered_histogram(
        samples,
        baseline_samples,
        histogram_path,
        cfg.comparison.histogram,
    )

    metrics = {
        "sampling/acceptance_rate": result.diagnostics.acceptance_rate,
        "sampling/mean_squared_jump": result.diagnostics.mean_squared_jump,
        "sampling/steps": result.diagnostics.steps,
        "samples/mean": float(samples.mean()),
        "samples/std": float(samples.std()),
        "baseline/std": float(baseline_samples.std()),
        **filter_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    tracking_run = initialize_wandb(cfg, run_dir)
    tracking_run.log(metrics)
    tracking_run.log({"samples": __import__("wandb").Image(str(run_dir / "samples.png"))})
    if contour_path.exists():
        tracking_run.log(
            {
                "samples_with_high_pass_contours": __import__("wandb").Image(
                    str(contour_path)
                )
            }
        )
    tracking_run.log(
        {
            "gsm_vs_gaussian": __import__("wandb").Image(str(comparison_path)),
            "filtered_coefficients_histogram": __import__("wandb").Image(
                str(histogram_path)
            ),
        }
    )
    if comparison_contour_path.exists():
        tracking_run.log(
            {
                "gsm_vs_gaussian_with_high_pass_contours": __import__("wandb").Image(
                    str(comparison_contour_path)
                )
            }
        )
    tracking_run.finish()
    print(f"Saved {len(samples)} samples to {run_dir}")
    print(json.dumps(metrics, indent=2))
    return run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
