"""Build a LaTeX report for the CelebA flow/GGSM coefficient and solver grid."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from gsmdiff.visualization import draw_high_pass_contours


FIELDS = (
    ("q1_p0", r"$q=1, p=0$ (flow baseline)"),
    ("q1_p0p005", r"$q=1, p=0.005$"),
    ("q1_p0p01", r"$q=1, p=0.01$"),
    ("q1_p0p05", r"$q=1, p=0.05$"),
    ("q0p95_p0p05", r"$q=0.95, p=0.05$"),
    ("q1_p_linear0_to_0p1", r"$q=1, p:0\rightarrow0.1$"),
)
REPRESENTATIVE_FIELDS = (
    ("q1_p0", r"Flow baseline"),
    ("q1_p0p01", r"$q=1, p=0.01$"),
    ("q1_p0p05", r"$q=1, p=0.05$"),
    ("q1_p_linear0_to_0p1", r"$q=1, p:0\rightarrow0.1$"),
)
CG_BUDGETS = (50, 500)


def _run_dir(
    input_root: Path,
    field: str,
    mean_field_iterations: int,
    cg_iterations: int,
    seed: int,
) -> Path:
    return input_root / (
        f"celeba_flow_sech_{field}_m{mean_field_iterations}_cg{cg_iterations}_"
        f"cold_seed{seed}"
    )


def _load_samples(
    input_root: Path,
    field: str,
    mean_field_iterations: int,
    cg_iterations: int,
    seed: int,
) -> Tensor:
    path = _run_dir(input_root, field, mean_field_iterations, cg_iterations, seed)
    path = path / "geometric_samples.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed experiment: {path}")
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(samples, Tensor) or samples.ndim != 4:
        raise ValueError(f"Expected a BCHW sample tensor at {path}.")
    return samples.detach().float().cpu()


def _load_metrics(
    input_root: Path,
    field: str,
    mean_field_iterations: int,
    cg_iterations: int,
    seed: int,
) -> dict[str, float]:
    path = _run_dir(input_root, field, mean_field_iterations, cg_iterations, seed)
    path = path / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed experiment: {path}")
    metrics = json.loads(path.read_text())
    if not isinstance(metrics, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return metrics


def _draw_sample(axis, sample: Tensor, *, contours: bool) -> None:
    image = sample.clamp(-1, 1)
    if image.shape[0] == 1:
        axis.imshow(image[0], cmap="gray", vmin=-1, vmax=1)
    elif image.shape[0] == 3:
        axis.imshow(((image.permute(1, 2, 0) + 1.0) / 2.0).clamp(0, 1))
    else:
        raise ValueError("Only one- and three-channel samples are supported.")
    if contours:
        draw_high_pass_contours(
            axis, image, quantile=0.9, color="#ffff00", opacity=1.0
        )
    axis.axis("off")


def _save_row_grid(
    rows: list[tuple[str, Tensor]],
    output: Path,
    *,
    contours: bool,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    count = min(4, *(len(samples) for _, samples in rows))
    figure, axes = plt.subplots(
        len(rows), count, figsize=(2.65 * count, 2.35 * len(rows)), squeeze=False
    )
    figure.suptitle(title, fontsize=15, y=0.998)
    for row_index, (label, samples) in enumerate(rows):
        for column in range(count):
            _draw_sample(axes[row_index, column], samples[column], contours=contours)
        axes[row_index, 0].text(
            -0.04,
            0.5,
            label,
            transform=axes[row_index, 0].transAxes,
            ha="right",
            va="center",
            rotation=90,
            fontsize=9,
        )
    figure.tight_layout(rect=(0.08, 0.0, 1.0, 0.985), h_pad=0.25, w_pad=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _paired_statistics(left: Tensor, right: Tensor) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"Cannot pair tensors with shapes {left.shape} and {right.shape}.")
    difference = left - right
    left_flat = left.flatten().double()
    right_flat = right.flatten().double()
    correlation = torch.corrcoef(torch.stack((left_flat, right_flat)))[0, 1]
    return {
        "rmse": float(difference.square().mean().sqrt()),
        "mae": float(difference.abs().mean()),
        "correlation": float(correlation),
        "max_absolute_difference": float(difference.abs().max()),
    }


def _metric_rows(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    reference_m = mean_field_values[0]
    for mean_field_iterations in mean_field_values:
        for cg_iterations in CG_BUDGETS:
            for field, label in FIELDS:
                metrics = _load_metrics(
                    input_root, field, mean_field_iterations, cg_iterations, seed
                )
                cg_pair = _paired_statistics(
                    _load_samples(input_root, field, mean_field_iterations, 50, seed),
                    _load_samples(input_root, field, mean_field_iterations, 500, seed),
                )
                m_pair = _paired_statistics(
                    _load_samples(input_root, field, reference_m, cg_iterations, seed),
                    _load_samples(
                        input_root, field, mean_field_iterations, cg_iterations, seed
                    ),
                )
                rows.append(
                    {
                        "field": field,
                        "label": label.replace("$", ""),
                        "mean_field_iterations": mean_field_iterations,
                        "cg_iterations": cg_iterations,
                        "sample_mean": metrics["samples/geometric_mean"],
                        "sample_std": metrics["samples/geometric_std"],
                        "filter_mean_absolute": metrics[
                            "filters/geometric_mean_absolute"
                        ],
                        "standardized_near_zero_fraction": metrics[
                            "filters/geometric_standardized_near_zero_fraction"
                        ],
                        "standardized_excess_kurtosis": metrics[
                            "filters/geometric_standardized_excess_kurtosis"
                        ],
                        "cg_mean_final_iterations": metrics[
                            "sampling/cg_mean_iterations"
                        ],
                        "cg_mean_total_iterations": metrics.get(
                            "sampling/cg_mean_total_iterations",
                            metrics["sampling/cg_mean_iterations"],
                        ),
                        "cg_fraction_above_tolerance": metrics[
                            "sampling/cg_fraction_above_tolerance"
                        ],
                        "cg_p95_relative_residual": metrics[
                            "sampling/cg_p95_relative_residual"
                        ],
                        "cg_max_relative_residual": metrics[
                            "sampling/cg_max_relative_residual"
                        ],
                        "weighted_prior_to_learned_rms_ratio_mean": metrics[
                            "sampling/weighted_prior_to_learned_rms_ratio_mean"
                        ],
                        "weighted_prior_to_learned_rms_ratio_p95": metrics[
                            "sampling/weighted_prior_to_learned_rms_ratio_p95"
                        ],
                        "velocity_cosine_similarity_mean": metrics[
                            "sampling/velocity_cosine_similarity_mean"
                        ],
                        "cg50_vs_cg500_rmse": cg_pair["rmse"],
                        "cg50_vs_cg500_correlation": cg_pair["correlation"],
                        "m_vs_reference_rmse": m_pair["rmse"],
                        "m_vs_reference_correlation": m_pair["correlation"],
                    }
                )
    return rows


def _save_metrics_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_field_schedules(output: Path) -> None:
    import matplotlib.pyplot as plt

    progress = np.linspace(0.0, 1.0, 1000)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    flow_values = {
        "q1_p0": 1.0,
        "q1_p0p005": 1.0,
        "q1_p0p01": 1.0,
        "q1_p0p05": 1.0,
        "q0p95_p0p05": 0.95,
        "q1_p_linear0_to_0p1": 1.0,
    }
    prior_values = {
        "q1_p0": np.zeros_like(progress),
        "q1_p0p005": np.full_like(progress, 0.005),
        "q1_p0p01": np.full_like(progress, 0.01),
        "q1_p0p05": np.full_like(progress, 0.05),
        "q0p95_p0p05": np.full_like(progress, 0.05),
        "q1_p_linear0_to_0p1": 0.1 * progress,
    }
    short_labels = ("1,0", "1,.005", "1,.01", "1,.05", ".95,.05", "1,0->.1")
    for (field, _), short_label in zip(FIELDS, short_labels):
        axes[0].plot(progress, np.full_like(progress, flow_values[field]), label=short_label)
        axes[1].plot(progress, prior_values[field], label=short_label)
    axes[0].set_title(r"Learned-flow coefficient $q$")
    axes[1].set_title(r"GGSM coefficient $p$")
    for axis in axes:
        axis.set_xlabel("Normalized forward-flow progress")
        axis.grid(alpha=0.25)
    axes[0].set_ylim(0.0, 1.08)
    axes[1].set_ylim(-0.004, 0.108)
    axes[1].legend(title="q,p", fontsize=8, ncol=2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _save_diagnostics(
    input_root: Path,
    mean_field_values: tuple[int, ...],
    seed: int,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    fields = [field for field, _ in FIELDS[1:]]
    labels = ["1,.005", "1,.01", "1,.05", ".95,.05", "linear"]
    metrics_to_plot = (
        ("sampling/cg_fraction_above_tolerance", "Fraction above CG tolerance"),
        ("sampling/cg_p95_relative_residual", "95th-percentile CG residual"),
        ("sampling/weighted_prior_to_learned_rms_ratio_mean", "Mean weighted prior/flow RMS"),
        ("sampling/velocity_cosine_similarity_mean", "Mean field cosine similarity"),
    )
    x = np.arange(len(fields))
    width = 0.36
    figure, axes = plt.subplots(
        len(mean_field_values),
        len(metrics_to_plot),
        figsize=(18.0, 3.7 * len(mean_field_values)),
        squeeze=False,
    )
    for row, mean_field_iterations in enumerate(mean_field_values):
        for column, (metric_name, title) in enumerate(metrics_to_plot):
            axis = axes[row, column]
            for offset, cg_iterations in ((-width / 2, 50), (width / 2, 500)):
                values = [
                    _load_metrics(
                        input_root, field, mean_field_iterations, cg_iterations, seed
                    )[metric_name]
                    for field in fields
                ]
                axis.bar(x + offset, values, width, label=f"CG-{cg_iterations}")
            axis.set_xticks(x, labels, rotation=25)
            axis.set_title(f"M={mean_field_iterations}: {title}", fontsize=10)
            axis.grid(axis="y", alpha=0.25)
            if row == 0 and column == 0:
                axis.legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _latex_summary_table(
    input_root: Path, mean_field_iterations: int, seed: int
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Field & Mean $|\Delta x|$ & Near zero & Kurtosis & CG fail & Prior/flow & Cosine \\",
        r"\midrule",
    ]
    for field, label in FIELDS:
        metrics = _load_metrics(input_root, field, mean_field_iterations, 500, seed)
        lines.append(
            f"{label} & "
            f"{metrics['filters/geometric_mean_absolute']:.5f} & "
            f"{metrics['filters/geometric_standardized_near_zero_fraction']:.4f} & "
            f"{metrics['filters/geometric_standardized_excess_kurtosis']:.2f} & "
            f"{metrics['sampling/cg_fraction_above_tolerance']:.3f} & "
            f"{metrics['sampling/weighted_prior_to_learned_rms_ratio_mean']:.4f} & "
            f"{metrics['sampling/velocity_cosine_similarity_mean']:.3f} \\\\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{CG-500 endpoint and trajectory statistics for "
            rf"$M={mean_field_iterations}$.}}",
            rf"\label{{tab:m{mean_field_iterations}-summary}}",
            r"\end{table}",
        )
    )
    return "\n".join(lines)


def _latex_cg_table(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{rlrrr}",
        r"\toprule",
        r"$M$ & Field & CG-50/500 RMSE & Correlation & CG-50 fail \\",
        r"\midrule",
    ]
    for mean_field_iterations in mean_field_values:
        for field, label in REPRESENTATIVE_FIELDS[1:]:
            paired = _paired_statistics(
                _load_samples(input_root, field, mean_field_iterations, 50, seed),
                _load_samples(input_root, field, mean_field_iterations, 500, seed),
            )
            metrics50 = _load_metrics(
                input_root, field, mean_field_iterations, 50, seed
            )
            lines.append(
                f"{mean_field_iterations} & {label} & {paired['rmse']:.5f} & "
                f"{paired['correlation']:.5f} & "
                f"{metrics50['sampling/cg_fraction_above_tolerance']:.3f} \\\\"
            )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired sensitivity to the CG budget. CG fail is the fraction "
            r"of final solves above tolerance in the CG-50 run.}",
            r"\label{tab:cg-paired}",
            r"\end{table}",
        )
    )
    return "\n".join(lines)


def _latex_m_table(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> str:
    reference_m = mean_field_values[0]
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Field & Comparison at CG-500 & RMSE & Correlation \\",
        r"\midrule",
    ]
    for field, label in REPRESENTATIVE_FIELDS[1:]:
        reference = _load_samples(input_root, field, reference_m, 500, seed)
        for mean_field_iterations in mean_field_values[1:]:
            paired = _paired_statistics(
                reference,
                _load_samples(
                    input_root, field, mean_field_iterations, 500, seed
                ),
            )
            lines.append(
                f"{label} & $M={reference_m}$ vs. $M={mean_field_iterations}$ & "
                f"{paired['rmse']:.5f} & {paired['correlation']:.5f} \\\\"
            )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired effect of mean-field refinement at CG-500.}",
            r"\label{tab:m-paired}",
            r"\end{table}",
        )
    )
    return "\n".join(lines)


def _data_summary(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> str:
    converged_m = mean_field_values[-1]
    baseline = _load_metrics(input_root, "q1_p0", converged_m, 500, seed)
    guided = [
        (
            field,
            label,
            _load_metrics(input_root, field, converged_m, 500, seed),
        )
        for field, label in FIELDS[1:]
    ]
    highest_near_zero = max(
        [("q1_p0", FIELDS[0][1], baseline), *guided],
        key=lambda item: item[2]["filters/geometric_standardized_near_zero_fraction"],
    )
    highest_kurtosis = max(
        [("q1_p0", FIELDS[0][1], baseline), *guided],
        key=lambda item: item[2]["filters/geometric_standardized_excess_kurtosis"],
    )
    worst_cg50 = max(
        (
            (
                field,
                label,
                _load_metrics(input_root, field, mean_field_values[0], 50, seed),
            )
            for field, label in FIELDS[1:]
        ),
        key=lambda item: item[2]["sampling/cg_fraction_above_tolerance"],
    )
    worst_cg500_m10 = _load_metrics(
        input_root, worst_cg50[0], converged_m, 500, seed
    )
    baseline_spread = max(
        _paired_statistics(
            _load_samples(input_root, "q1_p0", mean_field_values[0], 500, seed),
            _load_samples(input_root, "q1_p0", m, cg, seed),
        )["rmse"]
        for m in mean_field_values
        for cg in CG_BUDGETS
    )
    return rf"""
At CG-500 and $M={converged_m}$, the largest standardized near-zero fraction is
{highest_near_zero[2]['filters/geometric_standardized_near_zero_fraction']:.4f} for
{highest_near_zero[1]}, while the largest excess kurtosis is
{highest_kurtosis[2]['filters/geometric_standardized_excess_kurtosis']:.2f} for
{highest_kurtosis[1]}. These endpoint statistics are descriptive: visual quality and paired
changes must be inspected alongside them.

The hardest CG-50 case at $M={mean_field_values[0]}$ is {worst_cg50[1]}, with
{worst_cg50[2]['sampling/cg_fraction_above_tolerance']:.3f} of final solves above tolerance.
For the same field, CG-500 with $M={converged_m}$ reduces that fraction to
{worst_cg500_m10['sampling/cg_fraction_above_tolerance']:.3f}. This separates solver failure
from changes caused by the GGSM field itself.

The pure-flow control is numerically invariant across the nominal $M$/CG grid to a maximum
paired RMSE of {baseline_spread:.2e}, as expected because it skips the GGSM solve. Its endpoint
mean absolute first difference is
{baseline['filters/geometric_mean_absolute']:.5f} and its standardized excess kurtosis is
{baseline['filters/geometric_standardized_excess_kurtosis']:.2f}.
"""


def _write_latex_report(
    input_root: Path,
    output_root: Path,
    mean_field_values: tuple[int, ...],
    seed: int,
) -> None:
    m_list = ", ".join(str(value) for value in mean_field_values)
    result_sections = []
    for mean_field_iterations in mean_field_values:
        result_sections.append(
            rf"""
\subsection{{$M={mean_field_iterations}$, CG-500}}

\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.82\textheight,keepaspectratio]
    {{cg500_m{mean_field_iterations}_clean.png}}
  \caption{{Final samples for all field configurations at $M={mean_field_iterations}$ and
  CG-500. Rows are field choices; columns use matched initial noise.}}
  \label{{fig:m{mean_field_iterations}-clean}}
\end{{figure}}

\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.82\textheight,keepaspectratio]
    {{cg500_m{mean_field_iterations}_contours.png}}
  \caption{{The same samples with per-image high-pass contours.}}
\end{{figure}}

{_latex_summary_table(input_root, mean_field_iterations, seed)}
"""
        )

    m_sections = []
    for field, label in REPRESENTATIVE_FIELDS[1:]:
        m_sections.append(
            rf"""
\subsection{{{label}}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\textwidth,height=0.70\textheight,keepaspectratio]
    {{m_comparison_cg500_{field}_clean.png}}
  \caption{{CG-500 samples across mean-field iteration counts for {label}.}}
\end{{figure}}
"""
        )

    cg_sections = []
    for mean_field_iterations in mean_field_values:
        cg_sections.append(
            rf"""
\subsection{{$M={mean_field_iterations}$}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.84\textheight,keepaspectratio]
    {{cg_comparison_m{mean_field_iterations}_clean.png}}
  \caption{{Paired CG-50 and CG-500 samples for representative fields at
  $M={mean_field_iterations}$.}}
\end{{figure}}
"""
        )

    document = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.82in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float,microtype}}
\usepackage[hidelinks]{{hyperref}}
\graphicspath{{{{assets/}}}}

\title{{CelebA Geometric Flow--GGSM Sampling\\
\large Field coefficients, conjugate-gradient budgets, and mean-field iterations}}
\author{{}}
\date{{August 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This automatically generated report summarizes 36 paired CelebA experiments using a
pretrained optimal-transport flow and an independently weighted filtered GGSM vector field.
It compares six $(q,p)$ choices, mean-field counts $M\in\{{{m_list}\}}$, and CG budgets 50
and 500. Every grid cell contains four samples generated with 1000 Euler steps and seed
{seed}. Numerical values are loaded directly from the completed run artifacts.
\end{{abstract}}

\section{{Experimental design}}
The sampler integrates
\begin{{equation}}
  \frac{{dx_t}}{{dt}} = q_t v_\theta(x_t,t) + p_t v_{{\mathrm{{GGSM}}}}(x_t,t),
\end{{equation}}
where $q_t$ and $p_t$ are independent nonnegative coefficients and need not sum to one.
The GGSM velocity uses a moment-matched hyperbolic-secant first-difference model. Its noisy
conditional estimate is refined for $M$ mean-field iterations, with each linear system solved
using a cold-started conjugate-gradient method capped at either 50 or 500 iterations.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\textwidth]{{field_schedules.png}}
  \caption{{The six learned-flow and GGSM coefficient schedules.}}
\end{{figure}}

\section{{Quick findings}}
{_data_summary(input_root, mean_field_values, seed)}

\section{{Final samples and endpoint statistics}}
Rows in the image grids are field configurations and columns are paired samples. Displayed
images are clipped to $[-1,1]$. High-pass contours use each image's 90th percentile and show
the locations, rather than the absolute number, of large first differences.

{"".join(result_sections)}

\clearpage
\section{{Mean-field sensitivity}}
These comparisons hold the CG budget at 500. Differences between rows therefore measure the
effect of the variational refinement and its accumulated effect along the flow trajectory.

{"".join(m_sections)}

{_latex_m_table(input_root, mean_field_values, seed)}

\clearpage
\section{{CG-budget sensitivity}}
The diagnostic figure compares convergence and field interactions. A high fraction above
tolerance or a large residual indicates that CG-50 is not a faithful approximation to the
corresponding CG-500 trajectory.

\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.99\textwidth,height=0.90\textheight,keepaspectratio]
    {{flow_diagnostics.png}}
  \caption{{Solver convergence, relative field magnitude, and cosine alignment across the grid.}}
\end{{figure}}

{"".join(cg_sections)}

{_latex_cg_table(input_root, mean_field_values, seed)}

\section{{Reading the results}}
\begin{{enumerate}}
\item Use the $q=1,p=0$ row as the exact paired flow control. It intentionally performs no
GGSM solve, so $M$ and the CG budget cannot affect it.
\item Compare CG-50 and CG-500 before interpreting a visual change as an effect of the prior.
Poor residuals mean that solver truncation is itself changing the trajectory.
\item The prior/flow RMS ratio measures the actual weighted perturbation, while cosine
similarity distinguishes reinforcement from opposition between vector fields.
\item Near-zero concentration and excess kurtosis describe first-difference sparsity, but
neither alone measures perceptual image quality. Inspect the paired image grids as well.
\item The complete 36-row numerical export is available as \texttt{{metrics.csv}}.
\end{{enumerate}}

\end{{document}}
"""
    (output_root / "report.tex").write_text(document)


def _compile_report(output_root: Path) -> None:
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("--compile requested, but pdflatex is not available.")
    for _ in range(2):
        subprocess.run(
            [executable, "-interaction=nonstopmode", "-halt-on-error", "report.tex"],
            cwd=output_root,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )


def build_report(
    input_root: Path,
    output_root: Path,
    *,
    mean_field_values: tuple[int, ...],
    seed: int,
    compile_pdf: bool,
) -> None:
    if not mean_field_values:
        raise ValueError("At least one mean-field iteration value is required.")
    if len(set(mean_field_values)) != len(mean_field_values):
        raise ValueError("Mean-field iteration values must be unique.")
    output_root.mkdir(parents=True, exist_ok=True)
    assets = output_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # Validate the complete requested grid before producing a partial report.
    for mean_field_iterations in mean_field_values:
        for cg_iterations in CG_BUDGETS:
            for field, _ in FIELDS:
                _load_metrics(
                    input_root, field, mean_field_iterations, cg_iterations, seed
                )
                _load_samples(
                    input_root, field, mean_field_iterations, cg_iterations, seed
                )

    for mean_field_iterations in mean_field_values:
        rows = [
            (
                label,
                _load_samples(input_root, field, mean_field_iterations, 500, seed),
            )
            for field, label in FIELDS
        ]
        for contours, suffix in ((False, "clean"), (True, "contours")):
            _save_row_grid(
                rows,
                assets / f"cg500_m{mean_field_iterations}_{suffix}.png",
                contours=contours,
                title=f"CG-500, M={mean_field_iterations}: flow/GGSM fields",
            )

    for field, label in REPRESENTATIVE_FIELDS[1:]:
        rows = [
            (
                f"M={mean_field_iterations}",
                _load_samples(input_root, field, mean_field_iterations, 500, seed),
            )
            for mean_field_iterations in mean_field_values
        ]
        _save_row_grid(
            rows,
            assets / f"m_comparison_cg500_{field}_clean.png",
            contours=False,
            title=f"{label}: mean-field comparison at CG-500",
        )

    for mean_field_iterations in mean_field_values:
        rows = []
        for field, label in REPRESENTATIVE_FIELDS:
            for cg_iterations in CG_BUDGETS:
                rows.append(
                    (
                        f"{label}, CG-{cg_iterations}",
                        _load_samples(
                            input_root,
                            field,
                            mean_field_iterations,
                            cg_iterations,
                            seed,
                        ),
                    )
                )
        _save_row_grid(
            rows,
            assets / f"cg_comparison_m{mean_field_iterations}_clean.png",
            contours=False,
            title=f"M={mean_field_iterations}: paired CG-50 versus CG-500",
        )

    _save_field_schedules(assets / "field_schedules.png")
    _save_diagnostics(
        input_root, mean_field_values, seed, assets / "flow_diagnostics.png"
    )
    rows = _metric_rows(input_root, mean_field_values, seed)
    _save_metrics_csv(rows, output_root / "metrics.csv")
    _write_latex_report(input_root, output_root, mean_field_values, seed)
    if compile_pdf:
        _compile_report(output_root)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--mean-field-iterations", type=int, nargs="+", default=(1, 5, 10)
    )
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_report(
        args.input_root,
        args.output_root,
        mean_field_values=tuple(args.mean_field_iterations),
        seed=args.seed,
        compile_pdf=args.compile,
    )
    print(f"Saved metrics CSV to {args.output_root / 'metrics.csv'}")
    print(f"Saved LaTeX report to {args.output_root / 'report.tex'}")
    if args.compile:
        print(f"Saved compiled report to {args.output_root / 'report.pdf'}")


if __name__ == "__main__":
    main()
