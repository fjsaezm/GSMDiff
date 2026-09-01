"""Create a four-row clean/contour comparison for two fixed-lambda runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch import Tensor

from gsmdiff.visualization import draw_high_pass_contours


def _load_samples(run_dir: Path) -> Tensor:
    path = run_dir / "geometric_samples.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing sample tensor: {path}")
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(samples, Tensor) or samples.ndim != 4:
        raise ValueError(f"Expected a BCHW tensor in {path}.")
    return samples.detach().float().cpu()


def _validate_paired_runs(baseline_run: Path, sparse_run: Path) -> None:
    baseline_noise_path = baseline_run / "initial_noise.pt"
    sparse_noise_path = sparse_run / "initial_noise.pt"
    if baseline_noise_path.is_file() and sparse_noise_path.is_file():
        baseline_noise = torch.load(baseline_noise_path, map_location="cpu", weights_only=True)
        sparse_noise = torch.load(sparse_noise_path, map_location="cpu", weights_only=True)
        if not torch.equal(baseline_noise, sparse_noise):
            raise ValueError(
                "The runs do not use identical initial noise. Re-run both commands with the "
                "same runtime.seed, runtime.device, and runtime.dtype."
            )

    keys = (
        "runtime.seed",
        "runtime.device",
        "runtime.dtype",
        "sampling.method",
        "sampling.num_inference_steps",
        "sampling.batch_size",
        "model.id",
    )
    baseline_cfg = OmegaConf.load(baseline_run / "config.yaml")
    sparse_cfg = OmegaConf.load(sparse_run / "config.yaml")
    mismatches = []
    for key in keys:
        baseline_value = OmegaConf.select(baseline_cfg, key)
        sparse_value = OmegaConf.select(sparse_cfg, key)
        if baseline_value != sparse_value:
            mismatches.append(f"{key}: {baseline_value!r} != {sparse_value!r}")
    if mismatches:
        raise ValueError("The paired run configurations differ:\n" + "\n".join(mismatches))

    baseline_lambda = float(OmegaConf.select(baseline_cfg, "blend.value"))
    sparse_lambda = float(OmegaConf.select(sparse_cfg, "blend.value"))
    if baseline_lambda != 1.0:
        raise ValueError(f"Expected baseline blend.value=1.0, got {baseline_lambda}.")
    if not sparse_lambda < 1.0:
        raise ValueError(f"Expected the sparse run blend.value < 1.0, got {sparse_lambda}.")


def _draw_image(
    axis,
    sample: Tensor,
    *,
    with_contours: bool,
    contour_quantile: float,
    contour_threshold: float | None,
    contour_color: str,
    contour_opacity: float,
) -> None:
    image = sample.clamp(-1, 1)
    if image.shape[0] == 1:
        axis.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
    elif image.shape[0] == 3:
        axis.imshow(((image.permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1))
    else:
        raise ValueError("Only one- and three-channel images are supported.")
    if with_contours:
        draw_high_pass_contours(
            axis,
            image,
            quantile=contour_quantile,
            threshold=contour_threshold,
            color=contour_color,
            opacity=contour_opacity,
        )
    axis.axis("off")


def create_comparison_figure(
    baseline_run: Path,
    sparse_run: Path,
    output: Path,
    *,
    num_samples: int = 4,
    baseline_label: str = r"$\lambda=1.00$ (diffusion)",
    sparse_label: str = r"$\lambda=0.99$ (1\% GSM)",
    contour_quantile: float = 0.9,
    contour_threshold: float | None = None,
    contour_color: str = "#ffff00",
    contour_opacity: float = 1.0,
) -> Path:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    _validate_paired_runs(baseline_run, sparse_run)
    baseline = _load_samples(baseline_run)
    sparse = _load_samples(sparse_run)
    if baseline.shape != sparse.shape:
        raise ValueError(
            f"Sample shapes differ: {tuple(baseline.shape)} != {tuple(sparse.shape)}."
        )
    count = min(num_samples, len(baseline))

    import matplotlib.pyplot as plt

    rows = (
        (f"{baseline_label}: clean", baseline, False),
        (f"{baseline_label}: high-pass contours", baseline, True),
        (f"{sparse_label}: clean", sparse, False),
        (f"{sparse_label}: high-pass contours", sparse, True),
    )
    figure, axes = plt.subplots(
        4,
        count,
        figsize=(max(6.0, 2.65 * count), 10.4),
        squeeze=False,
    )
    for row_index, (row_label, samples, with_contours) in enumerate(rows):
        for column in range(count):
            _draw_image(
                axes[row_index, column],
                samples[column],
                with_contours=with_contours,
                contour_quantile=contour_quantile,
                contour_threshold=contour_threshold,
                contour_color=contour_color,
                contour_opacity=contour_opacity,
            )
        axes[row_index, 0].set_title(row_label, loc="left", fontsize=11, pad=7)
    figure.tight_layout(pad=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--sparse-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--baseline-label", default=r"$\lambda=1.00$ (diffusion)")
    parser.add_argument("--sparse-label", default=r"$\lambda=0.99$ (1\% GSM)")
    parser.add_argument("--contour-quantile", type=float, default=0.9)
    parser.add_argument("--contour-threshold", type=float, default=None)
    parser.add_argument("--contour-color", default="#ffff00")
    parser.add_argument("--contour-opacity", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = create_comparison_figure(
        args.baseline_run,
        args.sparse_run,
        args.output,
        num_samples=args.num_samples,
        baseline_label=args.baseline_label,
        sparse_label=args.sparse_label,
        contour_quantile=args.contour_quantile,
        contour_threshold=args.contour_threshold,
        contour_color=args.contour_color,
        contour_opacity=args.contour_opacity,
    )
    print(f"Saved four-row comparison to {output}")


if __name__ == "__main__":
    main()
