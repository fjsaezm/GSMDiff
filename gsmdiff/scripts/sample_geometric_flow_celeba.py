"""Sample the Algorithm 5 pretrained-flow/GGSM mixture on CelebA."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from gsmdiff.distributions import FirstDifferenceFilterProduct
from gsmdiff.flow import PretrainedPnPFlowModel
from gsmdiff.geometric import FilteredGSMNoisyScore, GeometricFlowSampler
from gsmdiff.scripts.sample_geometric_celebahq import (
    _coefficient_metrics,
    _configure_reproducibility,
    _prior_energy_metrics,
    _resolve_device,
    _save_comparison_grid,
    _save_sparsity_plot,
)
from gsmdiff.utils.tracking import initialize_wandb


def _compact(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _schedule_label(config: DictConfig) -> str:
    if str(config.name) == "constant":
        return _compact(config.value)
    return f"{config.name}_{_compact(config.start)}_to_{_compact(config.end)}"


def _save_weight_schedule(steps, path: Path) -> None:
    import matplotlib.pyplot as plt

    progress = np.linspace(0.0, 1.0, len(steps))
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(
        progress,
        [step.flow_coefficient for step in steps],
        linewidth=2.2,
        label=r"Learned flow $\lambda_q$",
    )
    axis.plot(
        progress,
        [step.prior_coefficient for step in steps],
        linewidth=2.2,
        label=r"GGSM flow $\lambda_p$",
    )
    axis.set_xlabel("Normalized forward-flow progress")
    axis.set_ylabel("Independent vector-field coefficient")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _aggregate_step_metrics(
    steps, *, cg_tolerance: float, cg_max_iterations: int
) -> dict[str, float]:
    if not steps:
        return {}
    cg_iterations = np.asarray([step.cg_iterations for step in steps])
    cg_residuals = np.asarray([step.cg_maximum_relative_residual for step in steps])
    learned = np.asarray([step.weighted_learned_velocity_norm for step in steps])
    prior = np.asarray([step.weighted_prior_velocity_norm for step in steps])
    ratio = prior / np.maximum(learned, np.finfo(np.float64).eps)
    return {
        "sampling/flow_coefficient_start": steps[0].flow_coefficient,
        "sampling/flow_coefficient_end": steps[-1].flow_coefficient,
        "sampling/prior_coefficient_start": steps[0].prior_coefficient,
        "sampling/prior_coefficient_end": steps[-1].prior_coefficient,
        "sampling/cg_mean_iterations": float(np.mean(cg_iterations)),
        "sampling/cg_mean_total_iterations": float(
            np.mean([step.cg_total_iterations for step in steps])
        ),
        "sampling/mean_field_iterations": int(
            np.max([step.mean_field_iterations for step in steps])
        ),
        "sampling/cg_total_solves_at_iteration_limit": int(
            np.sum([step.cg_solves_at_iteration_limit for step in steps])
        ),
        "sampling/cg_fraction_at_iteration_limit": float(
            np.mean(cg_iterations >= cg_max_iterations)
        ),
        "sampling/cg_fraction_above_tolerance": float(np.mean(cg_residuals > cg_tolerance)),
        "sampling/cg_p95_relative_residual": float(np.quantile(cg_residuals, 0.95)),
        "sampling/cg_max_relative_residual": float(np.max(cg_residuals)),
        "sampling/learned_velocity_rms_mean": float(
            np.mean([step.learned_velocity_norm for step in steps])
        ),
        "sampling/prior_velocity_rms_mean": float(
            np.mean([step.prior_velocity_norm for step in steps])
        ),
        "sampling/weighted_prior_to_learned_rms_ratio_mean": float(np.mean(ratio)),
        "sampling/weighted_prior_to_learned_rms_ratio_p95": float(np.quantile(ratio, 0.95)),
        "sampling/velocity_cosine_similarity_mean": float(
            np.mean([step.velocity_cosine_similarity for step in steps])
        ),
    }


def _save_snapshots(snapshots, final_samples, run_dir: Path, cfg: DictConfig) -> None:
    if not snapshots:
        return
    root = run_dir / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    final_step = snapshots[-1].completed_steps
    for snapshot in snapshots:
        directory = root / f"step_{snapshot.completed_steps:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        shown = final_samples if snapshot.completed_steps == final_step else snapshot.clean_estimate
        _save_comparison_grid(
            shown,
            None,
            directory / "samples.png",
            max_samples=int(cfg.image.num_preview),
            geometric_label="Geometric flow + filtered GSM",
        )
        _save_comparison_grid(
            shown,
            None,
            directory / "samples_with_high_pass_contours.png",
            max_samples=int(cfg.image.num_preview),
            contour_cfg=cfg.image.high_pass_contours,
            geometric_label="Geometric flow + filtered GSM",
        )


def run(cfg: DictConfig) -> Path:
    seed = int(cfg.runtime.seed)
    _configure_reproducibility(seed, bool(cfg.runtime.deterministic))
    device = _resolve_device(str(cfg.runtime.device))
    dtype = getattr(torch, str(cfg.runtime.dtype))
    if str(cfg.sampling.method) != "euler":
        raise ValueError("Algorithm 5 currently requires sampling.method=euler.")

    flow = PretrainedPnPFlowModel.from_pretrained(
        str(cfg.model.checkpoint),
        device=device,
        dtype=dtype,
        download=bool(cfg.model.download),
        google_drive_id=cfg.model.get("google_drive_id"),
        sha256=cfg.model.get("sha256"),
        input_channels=int(cfg.model.input_channels),
        input_height=int(cfg.model.input_height),
        channels=int(cfg.model.channels),
        channel_multipliers=tuple(int(v) for v in cfg.model.channel_multipliers),
        residual_blocks=int(cfg.model.residual_blocks),
        attention_resolutions=tuple(int(v) for v in cfg.model.attention_resolutions),
    )
    coefficient_distribution = instantiate(cfg.distribution)
    prior = FirstDifferenceFilterProduct(
        coefficient_distribution, mean_scale=float(cfg.prior.mean_scale)
    )
    prior_velocity = FilteredGSMNoisyScore(
        prior,
        cg_max_iterations=int(cfg.prior.cg.max_iterations),
        cg_tolerance=float(cfg.prior.cg.tolerance),
        numerical_epsilon=float(cfg.prior.cg.numerical_epsilon),
        warm_start=bool(cfg.prior.cg.warm_start),
        mean_field_iterations=int(cfg.prior.mean_field_iterations),
    )
    sampler = GeometricFlowSampler(
        flow,
        prior_velocity,
        instantiate(cfg.flow_weight),
        instantiate(cfg.prior_weight),
    )
    shape = (
        int(cfg.sampling.batch_size),
        flow.in_channels,
        flow.sample_size,
        flow.sample_size,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)
    snapshot_every = cfg.checkpoints.get("every_steps")
    result = sampler.sample(
        batch_size=shape[0],
        num_inference_steps=int(cfg.sampling.num_inference_steps),
        device=device,
        dtype=dtype,
        initial_noise=initial_noise,
        snapshot_every=None if snapshot_every is None else int(snapshot_every),
    )
    geometric = result.samples.detach().float().cpu()

    baseline = None
    if bool(cfg.comparison.generate_flow_baseline):
        baseline = (
            sampler.sample(
                batch_size=shape[0],
                num_inference_steps=int(cfg.sampling.num_inference_steps),
                device=device,
                dtype=dtype,
                initial_noise=initial_noise,
                flow_coefficient_override=1.0,
                prior_coefficient_override=0.0,
            )
            .samples.detach()
            .float()
            .cpu()
        )

    run_name = cfg.run.get("name") or (
        f"celeba_geometric_flow_{cfg.distribution.name}_"
        f"q{_schedule_label(cfg.flow_weight)}_p{_schedule_label(cfg.prior_weight)}_"
        f"seed{seed}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    )
    run_dir = Path(str(cfg.run.output_dir)) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config.yaml")
    torch.save(geometric, run_dir / "geometric_samples.pt")
    if baseline is not None:
        torch.save(baseline, run_dir / "flow_baseline_samples.pt")
    (run_dir / "step_diagnostics.json").write_text(
        json.dumps([asdict(step) for step in result.steps], indent=2)
    )

    grid_options = {
        "max_samples": int(cfg.image.num_preview),
        "geometric_label": "Geometric flow + filtered GSM",
        "baseline_label": "Pretrained flow baseline",
    }
    _save_comparison_grid(geometric, baseline, run_dir / "samples.png", **grid_options)
    _save_comparison_grid(
        geometric,
        baseline,
        run_dir / "samples_with_high_pass_contours.png",
        contour_cfg=cfg.image.high_pass_contours,
        **grid_options,
    )
    _save_sparsity_plot(
        geometric,
        baseline,
        run_dir / "filter_sparsity.png",
        cfg.analysis.histogram,
        geometric_label="Geometric flow + GGSM",
        baseline_label="Pretrained flow",
    )
    _save_weight_schedule(result.steps, run_dir / "field_weight_schedule.png")
    _save_snapshots(result.snapshots, geometric, run_dir, cfg)

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
            "field_weight_schedule": wandb.Image(str(run_dir / "field_weight_schedule.png")),
            "filter_sparsity": wandb.Image(str(run_dir / "filter_sparsity.png")),
        }
    )
    tracking_run.finish()
    print(f"Saved {len(geometric)} geometric flow/GGSM samples to {run_dir}")
    print(json.dumps(metrics, indent=2))
    return run_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="geometric_flow_celeba")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
