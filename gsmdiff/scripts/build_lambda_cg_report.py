"""Build a LaTeX report for the lambda, CG, and mean-field iteration grid."""

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


SCHEDULES = (
    ("lambda100", r"Constant $\lambda=1.00$"),
    ("lambda099", r"Constant $\lambda=0.99$"),
    ("lambda090", r"Constant $\lambda=0.90$"),
    ("lambda080", r"Constant $\lambda=0.80$"),
    ("power5_1_to_0", r"$1-(t/(N-1))^5$"),
    ("power10_1_to_0", r"$1-(t/(N-1))^{10}$"),
)
REPRESENTATIVE_SCHEDULES = (
    ("lambda080", r"Constant $\lambda=0.80$"),
    ("power5_1_to_0", r"Fifth-power $1\rightarrow0$"),
    ("power10_1_to_0", r"Tenth-power $1\rightarrow0$"),
)
CG_BUDGETS = (500, 50)


def _run_dir(
    input_root: Path,
    schedule: str,
    mean_field_iterations: int,
    cg_iterations: int,
    seed: int,
) -> Path:
    return input_root / (
        f"celebahq_sech_{schedule}_m{mean_field_iterations}_cg{cg_iterations}_"
        f"cold_seed{seed}"
    )


def _load_samples(
    input_root: Path,
    schedule: str,
    mean_field_iterations: int,
    cg_iterations: int,
    seed: int,
) -> Tensor:
    path = _run_dir(
        input_root, schedule, mean_field_iterations, cg_iterations, seed
    ) / "geometric_samples.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed experiment: {path}")
    samples = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(samples, Tensor) or samples.ndim != 4:
        raise ValueError(f"Expected a BCHW sample tensor at {path}.")
    return samples.detach().float().cpu()


def _load_metrics(
    input_root: Path,
    schedule: str,
    mean_field_iterations: int,
    cg_iterations: int,
    seed: int,
) -> dict[str, float]:
    path = _run_dir(
        input_root, schedule, mean_field_iterations, cg_iterations, seed
    ) / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed experiment: {path}")
    return json.loads(path.read_text())


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
        len(rows), count, figsize=(2.7 * count, 2.5 * len(rows)), squeeze=False
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
            fontsize=10,
        )
    figure.tight_layout(rect=(0.07, 0.0, 1.0, 0.985), h_pad=0.35, w_pad=0.15)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _paired_statistics(left: Tensor, right: Tensor) -> dict[str, float]:
    difference = left - right
    correlation = torch.corrcoef(torch.stack((left.flatten(), right.flatten())))[0, 1]
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
            for schedule, label in SCHEDULES:
                metrics = _load_metrics(
                    input_root,
                    schedule,
                    mean_field_iterations,
                    cg_iterations,
                    seed,
                )
                cg_pair = _paired_statistics(
                    _load_samples(input_root, schedule, mean_field_iterations, 50, seed),
                    _load_samples(input_root, schedule, mean_field_iterations, 500, seed),
                )
                m_pair = _paired_statistics(
                    _load_samples(input_root, schedule, reference_m, cg_iterations, seed),
                    _load_samples(
                        input_root,
                        schedule,
                        mean_field_iterations,
                        cg_iterations,
                        seed,
                    ),
                )
                rows.append(
                    {
                        "schedule": schedule,
                        "label": label.replace("$", ""),
                        "mean_field_iterations": mean_field_iterations,
                        "cg_iterations": cg_iterations,
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
                        "cg_max_relative_residual": metrics[
                            "sampling/cg_max_relative_residual"
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


def _save_cg_diagnostics(
    input_root: Path,
    mean_field_values: tuple[int, ...],
    seed: int,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    schedules = [schedule for schedule, _ in SCHEDULES[1:]]
    labels = ["0.99", "0.90", "0.80", "power-5", "power-10"]
    fields = (
        ("sampling/cg_fraction_above_tolerance", "Fraction above tolerance"),
        ("sampling/cg_max_relative_residual", "Maximum relative residual"),
        ("sampling/cg_mean_total_iterations", "Mean total CG iterations"),
    )
    x = np.arange(len(schedules))
    width = 0.36
    figure, axes = plt.subplots(
        len(mean_field_values), len(fields), figsize=(15.0, 4.0 * len(mean_field_values)),
        squeeze=False,
    )
    for row_index, mean_field_iterations in enumerate(mean_field_values):
        for column, (field, title) in enumerate(fields):
            axis = axes[row_index, column]
            for offset, cg_iterations in ((-width / 2, 50), (width / 2, 500)):
                values = []
                for schedule in schedules:
                    metrics = _load_metrics(
                        input_root,
                        schedule,
                        mean_field_iterations,
                        cg_iterations,
                        seed,
                    )
                    fallback = metrics.get("sampling/cg_mean_iterations", 0.0)
                    values.append(metrics.get(field, fallback))
                axis.bar(
                    x + offset,
                    values,
                    width,
                    label=f"CG-{cg_iterations}",
                )
            axis.set_xticks(x, labels, rotation=25)
            axis.set_title(f"M={mean_field_iterations}: {title}")
            axis.grid(axis="y", alpha=0.25)
            if row_index == 0 and column == 0:
                axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _latex_summary_table(
    input_root: Path, mean_field_iterations: int, seed: int
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Schedule & Mean $|\Delta x|$ & Near zero & Kurtosis & CG fail & Max residual \\",
        r"\midrule",
    ]
    for schedule, label in SCHEDULES:
        metrics = _load_metrics(input_root, schedule, mean_field_iterations, 500, seed)
        lines.append(
            f"{label} & "
            f"{metrics['filters/geometric_mean_absolute']:.5f} & "
            f"{metrics['filters/geometric_standardized_near_zero_fraction']:.5f} & "
            f"{metrics['filters/geometric_standardized_excess_kurtosis']:.2f} & "
            f"{metrics['sampling/cg_fraction_above_tolerance']:.3f} & "
            f"{metrics['sampling/cg_max_relative_residual']:.5f} \\\\"
        )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{CG-500 statistics for $M={mean_field_iterations}$.}}",
            rf"\label{{tab:m{mean_field_iterations}-summary}}",
            r"\end{table}",
        )
    )
    return "\n".join(lines)


def _latex_m_comparison_table(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> str:
    reference_m = mean_field_values[0]
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Schedule & Comparison & RMSE & Correlation \\",
        r"\midrule",
    ]
    for schedule, label in REPRESENTATIVE_SCHEDULES:
        reference = _load_samples(input_root, schedule, reference_m, 500, seed)
        for mean_field_iterations in mean_field_values[1:]:
            candidate = _load_samples(
                input_root, schedule, mean_field_iterations, 500, seed
            )
            paired = _paired_statistics(reference, candidate)
            lines.append(
                f"{label} & $M={reference_m}$ vs. $M={mean_field_iterations}$ & "
                f"{paired['rmse']:.5f} & {paired['correlation']:.6f} \\\\"
            )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired CG-500 sample differences caused by increasing the number of mean-field iterations.}",
            r"\label{tab:m-paired}",
            r"\end{table}",
        )
    )
    return "\n".join(lines)


def _latex_cg_comparison_table(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\begin{tabular}{rlrr}",
        r"\toprule",
        r"$M$ & Schedule & CG-50 vs. CG-500 RMSE & Correlation \\",
        r"\midrule",
    ]
    for mean_field_iterations in mean_field_values:
        for schedule, label in REPRESENTATIVE_SCHEDULES:
            paired = _paired_statistics(
                _load_samples(input_root, schedule, mean_field_iterations, 50, seed),
                _load_samples(input_root, schedule, mean_field_iterations, 500, seed),
            )
            lines.append(
                f"{mean_field_iterations} & {label} & {paired['rmse']:.5f} & "
                f"{paired['correlation']:.6f} \\\\"
            )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Paired sensitivity to the CG budget for representative schedules at every $M$.}",
            r"\label{tab:cg-paired}",
            r"\end{table}",
        )
    )
    return "\n".join(lines)


def _latex_failure_analysis(
    input_root: Path, mean_field_values: tuple[int, ...], seed: int
) -> str:
    reference_m = mean_field_values[0]
    converged_m = mean_field_values[-1]
    baseline = _load_metrics(input_root, "lambda100", converged_m, 500, seed)
    schedules = (
        ("lambda099", r"Constant $\lambda=0.99$"),
        ("lambda090", r"Constant $\lambda=0.90$"),
        ("lambda080", r"Constant $\lambda=0.80$"),
        ("power10_1_to_0", "Power 10"),
        ("power5_1_to_0", "Power 5"),
    )

    statistic_lines = []
    for schedule, label in schedules:
        metrics = _load_metrics(input_root, schedule, converged_m, 500, seed)
        statistic_lines.append(
            f"{label} & {metrics['filters/geometric_mean_absolute']:.4f} & "
            f"{metrics['filters/geometric_standardized_near_zero_fraction']:.4f} & "
            f"{metrics['filters/geometric_standardized_excess_kurtosis']:.2f} \\\\"
        )

    m_differences = {}
    for schedule in ("lambda080", "power5_1_to_0", "power10_1_to_0"):
        m_differences[schedule] = _paired_statistics(
            _load_samples(input_root, schedule, reference_m, 500, seed),
            _load_samples(input_root, schedule, converged_m, 500, seed),
        )["rmse"]

    cg_differences = {}
    for schedule in ("power5_1_to_0", "power10_1_to_0"):
        cg_differences[schedule] = _paired_statistics(
            _load_samples(input_root, schedule, converged_m, 50, seed),
            _load_samples(input_root, schedule, converged_m, 500, seed),
        )["rmse"]

    power10_samples = _load_samples(
        input_root, "power10_1_to_0", converged_m, 500, seed
    )
    power5_samples = _load_samples(
        input_root, "power5_1_to_0", converged_m, 500, seed
    )
    baseline_samples = _load_samples(input_root, "lambda100", converged_m, 500, seed)
    def outside(value: Tensor) -> float:
        return float(((value < -1) | (value > 1)).float().mean())

    diagnostics_path = _run_dir(
        input_root, "power10_1_to_0", converged_m, 500, seed
    ) / "step_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text())
    by_timestep = {int(step["timestep"]): step for step in diagnostics}
    late_rows = []
    for timestep in (49, 9, 0):
        step = by_timestep[timestep]
        late_rows.append(
            f"{timestep} & {step['blend']:.5f} & "
            f"{step['diffusion_score_norm']:.2f} & "
            f"{step['prior_score_norm']:.2f} \\\\"
        )

    return rf"""
\clearpage
\section{{Why the intended sparsification is not occurring}}

The combined evidence indicates that the main failure is not an insufficient linear solve or
too few mean-field iterations. Instead, the experiment progressively replaces a highly
structured CelebA score with a broad first-difference model whose scale and probability path
are not matched to the diffusion model.

\subsection{{The numerical solvers are not the primary limitation}}

At CG-500, changing $M={reference_m}$ to $M={converged_m}$ changes the final samples by paired
RMSE {m_differences['lambda080']:.5f} for constant $\lambda=0.80$,
{m_differences['power5_1_to_0']:.5f} for power 5, and
{m_differences['power10_1_to_0']:.5f} for power 10. Moreover, the $M=5$ and $M=10$
statistics and images are nearly indistinguishable. The mean-field fixed point has therefore
essentially stabilized by $M=5$.

For $M={converged_m}$, the CG-500 final solves reach the $10^{{-4}}$ tolerance, but the
artifacts remain. Power 5 and power 10 are especially diagnostic: their CG-50 versus CG-500
paired RMSE values are only {cg_differences['power5_1_to_0']:.2e} and
{cg_differences['power10_1_to_0']:.2e}, respectively. The difficult CG systems occur mainly
at early noisy steps, where these power schedules assign negligible weight to the GGSM score.
Increasing either numerical budget further is therefore unlikely to correct the qualitative
failure.

\subsection{{The convex blend removes the learned denoising score}}

The implemented score is
\begin{{equation}}
  s_r(x_t,t)=\lambda_t s_q(x_t,t)+(1-\lambda_t)s_p(x_t,t).
\end{{equation}}
Reducing $\lambda_t$ does not merely add regularization: it removes part of the learned
CelebA score. This becomes most damaging near the clean endpoint, where the diffusion score
must remove the remaining fine-scale noise. For power 10 and $M={converged_m}$:

\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{rrrr}}
\toprule
Timestep & $\lambda_t$ & RMS $\lVert s_q\rVert$ & RMS $\lVert s_p\rVert$ \\
\midrule
{"".join(late_rows)}
\bottomrule
\end{{tabular}}
\caption{{Late reverse-process score magnitudes for power 10.}}
\end{{table}}

The learned score grows rapidly near $t=0$, while the GGSM score remains of order one and
$\lambda_t$ tends to zero. Thus the sampler suppresses precisely the score needed to eliminate
late high-frequency noise. The colored texture in the samples is consistent with incomplete
learned denoising, not with successful sparse regularization.

\subsection{{A schedule ending at $\lambda=0$ has the wrong endpoint target}}

For either power schedule, the formal endpoint is
\begin{{equation}}
 r_0(x)\propto q_0(x)^0p_0(x)^1=p_0(x).
\end{{equation}}
The first-difference GGSM contains no model of faces, identity, pose, coherent color, or
background structure. It cannot replace the learned CelebA distribution while preserving
realistic faces. Power 10 looks better than power 5 only because it delays this replacement;
it does not remove the endpoint incompatibility.

\subsection{{The GGSM coefficient scale is mismatched}}

The hyperbolic-secant prior uses scale one, while the diffusion baseline has mean absolute
first difference {baseline['filters/geometric_mean_absolute']:.4f}. For coefficients much
smaller than one,
\begin{{equation}}
 -\log\operatorname{{sech}}(d)\simeq\frac{{d^2}}2,
 \qquad \tanh(d)\simeq d.
\end{{equation}}
Consequently, over most clean CelebA coefficients the chosen prior behaves like a broad
quadratic model rather than a sharply peaked sparse model. Its scale should be estimated from
clean CelebA filter responses and its strength calibrated separately from the diffusion score.

The empirical trend confirms the mismatch:
\begin{{table}}[H]
\centering\small
\begin{{tabular}}{{lrrr}}
\toprule
Schedule & Mean $|\Delta x|$ & Near-zero fraction & Excess kurtosis \\
\midrule
Diffusion $\lambda=1$ & {baseline['filters/geometric_mean_absolute']:.4f} &
{baseline['filters/geometric_standardized_near_zero_fraction']:.4f} &
{baseline['filters/geometric_standardized_excess_kurtosis']:.2f} \\
{"".join(statistic_lines)}
\bottomrule
\end{{tabular}}
\caption{{CG-500, $M={converged_m}$: strengthening the GGSM moves the filter statistics away
from a peaked, heavy-tailed response law.}}
\end{{table}}

The baseline already has the largest near-zero concentration and heaviest tails. Greater GGSM
weight increases the mean difference while collapsing both sparsity indicators. This is the
opposite of the intended effect.

The raw tensors also increasingly leave the model's nominal $[-1,1]$ image range: the fraction
is {outside(baseline_samples):.3f} for $\lambda=1$, {outside(power10_samples):.3f} for power 10,
and {outside(power5_samples):.3f} for power 5. Display clipping therefore hides part of the
trajectory's amplitude growth.

\subsection{{Mean-field convergence does not remove modeling approximations}}

The exact noisy GGSM conditional mean contains
\begin{{equation}}
 \mathbb E\!\left[(\alpha_t^2I+\sigma_t^2P(\Omega))^{{-1}}\mid x_t\right].
\end{{equation}}
The implementation replaces this expectation of an inverse by an inverse evaluated at plug-in
precisions. Iterating $M$ converges that plug-in fixed point; it does not make the approximation
equal to the exact posterior expectation.

More fundamentally,
\begin{{equation}}
 r_t(x_t)\propto q_t(x_t)^{{\lambda_t}}p_t(x_t)^{{1-\lambda_t}}
\end{{equation}}
is generally not the forward-diffusion marginal of the corresponding clean product. A
time-dependent $\lambda_t$ increases this path inconsistency. Inserting its instantaneous score
into the original VP reverse transition is therefore a guided dynamics, not an exact reverse
sampler for a consistent clean target.

\subsection{{Most informative next experiment}}

The next test should preserve the complete learned score and add independently weighted prior
guidance:
\begin{{equation}}
 s_{{\mathrm{{total}}}}=s_q+\gamma_t s_p,
 \qquad
 r_t(x_t)\propto q_t(x_t)p_t(x_t)^{{\gamma_t}}.
\end{{equation}}
This makes $\gamma_t=0$ the exact diffusion baseline without attenuating learned denoising.
The hyperbolic-secant scale should first be fitted to clean CelebA first differences, followed
by a small sweep over $\gamma_t$. A late-time schedule may increase $\gamma_t$, but it should
not turn off $s_q$. Logging the cosine similarity and the separately weighted norms of $s_q$
and $s_p$ would then reveal whether the prior reinforces or conflicts with learned denoising.
"""


def _write_latex_report(
    input_root: Path,
    output_root: Path,
    mean_field_values: tuple[int, ...],
    seed: int,
) -> None:
    m_list = ", ".join(str(value) for value in mean_field_values)
    sections: list[str] = []
    for mean_field_iterations in mean_field_values:
        sections.append(
            rf"""
\subsection{{$M={mean_field_iterations}$}}

Figures~\ref{{fig:m{mean_field_iterations}-clean}} and
\ref{{fig:m{mean_field_iterations}-contours}} compare all lambda schedules using CG-500.

\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.82\textheight,keepaspectratio]{{cg500_m{mean_field_iterations}_clean.png}}
  \caption{{Clean CG-500 samples for $M={mean_field_iterations}$. Rows are lambda schedules and columns share the fixed random seed.}}
  \label{{fig:m{mean_field_iterations}-clean}}
\end{{figure}}

\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.82\textheight,keepaspectratio]{{cg500_m{mean_field_iterations}_contours.png}}
  \caption{{CG-500 samples with high-pass contours for $M={mean_field_iterations}$.}}
  \label{{fig:m{mean_field_iterations}-contours}}
\end{{figure}}

{_latex_summary_table(input_root, mean_field_iterations, seed)}
"""
        )

    representative_sections: list[str] = []
    for schedule, label in REPRESENTATIVE_SCHEDULES:
        representative_sections.append(
            rf"""
\subsection{{{label}}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\textwidth,height=0.68\textheight,keepaspectratio]{{m_comparison_cg500_{schedule}_clean.png}}
  \caption{{Clean CG-500 comparison across $M$ for {label}.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.92\textwidth,height=0.68\textheight,keepaspectratio]{{m_comparison_cg500_{schedule}_contours.png}}
  \caption{{High-pass contour comparison across $M$ for {label}.}}
\end{{figure}}
"""
        )

    cg_visual_sections: list[str] = []
    for mean_field_iterations in mean_field_values:
        cg_visual_sections.append(
            rf"""
\subsection{{$M={mean_field_iterations}$}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.82\textheight,keepaspectratio]{{cg_comparison_m{mean_field_iterations}_clean.png}}
  \caption{{Representative clean samples for CG-50 and CG-500 at $M={mean_field_iterations}$.}}
\end{{figure}}
\begin{{figure}}[p]
  \centering
  \includegraphics[width=0.94\textwidth,height=0.82\textheight,keepaspectratio]{{cg_comparison_m{mean_field_iterations}_contours.png}}
  \caption{{Representative high-pass contour comparison for CG-50 and CG-500 at $M={mean_field_iterations}$.}}
\end{{figure}}
"""
        )

    document = rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=0.82in]{{geometry}}
\usepackage{{amsmath,amssymb,booktabs,graphicx,float,microtype}}
\usepackage[hidelinks]{{hyperref}}
\graphicspath{{{{assets/}}}}

\title{{CelebA-HQ Geometric Diffusion--GGSM Sampling\\
\large Lambda, conjugate-gradient, and mean-field iteration study}}
\author{{}}
\date{{August 2026}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report compares Algorithm~2 of \texttt{{Sampling\_GSM\_models-2}} using variational
mean-field iteration counts $M\in\{{{m_list}\}}$. For every $M$, six lambda schedules and
CG budgets 50 and 500 are evaluated. All paired runs use four samples, 1000 reverse steps,
cold starts between reverse timesteps, and seed {seed}. The report is generated directly
from the completed run tensors and JSON diagnostics; no numerical table is hard-coded.
\end{{abstract}}

\section{{Algorithm and experimental design}}
At reverse timestep $t$, the diffusion model initializes
\begin{{equation}}
\widehat x_0^{{(0)}}=\alpha_t^{{-1}}(x_t+\sigma_t^2s_q(x_t,t)).
\end{{equation}}
For $m=1,\ldots,M$, Algorithm~2 alternates
\begin{{align}}
\widehat\Omega_t^{{(m)}}
 &=\operatorname{{Diag}}\!\left(\mathbb E[\omega_i\mid
 (F\widehat x_0^{{(m-1)}})_i]\right),\\
(\alpha_t^2I+\sigma_t^2F^\top\widehat\Omega_t^{{(m)}}F)v_t^{{(m)}}&=x_t,\\
\widehat x_0^{{(m)}}&=\alpha_tv_t^{{(m)}}.
\end{{align}}
The final GGSM score is combined with the diffusion score as
\begin{{equation}}
s_r=\lambda_ts_q+(1-\lambda_t)s_p,
\qquad
s_p=-\sigma_t^{{-2}}(x_t-\alpha_t\widehat x_0^{{(M)}}).
\end{{equation}}
The stochastic reverse transition uses the DDPM posterior standard deviation
$\sigma_{{t\to t-1}}$. CG-50 and CG-500 denote the maximum iterations of each inner
linear solve, not the number of mean-field iterations.

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.88\textwidth]{{lambda_schedules.png}}
  \caption{{The six lambda schedules shared by every $M$ and CG configuration.}}
\end{{figure}}

\section{{CG-500 results within each mean-field setting}}
The contour threshold is the per-image 90th percentile, so contour figures compare where
large first differences occur rather than their absolute count. Near-zero fractions and
kurtosis in the tables are computed from standardized pooled first differences.

{"".join(sections)}

\clearpage
\section{{Direct comparison across mean-field iterations}}
The following figures place $M={mean_field_values[0]}$, $M={mean_field_values[1]}$, and
$M={mean_field_values[2]}$ in adjacent rows for representative schedules. Because all runs
share the initial and reverse random streams, changes are paired consequences of the
mean-field refinement and its interaction with the reverse trajectory.

{"".join(representative_sections)}

{_latex_m_comparison_table(input_root, mean_field_values, seed)}

\section{{CG accuracy across $M$}}
\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.98\textwidth,height=0.82\textheight,keepaspectratio]{{cg_diagnostics_by_m.png}}
  \caption{{CG-50 versus CG-500 diagnostics for each $M$. Total CG iterations sum all $M$
  inner solves at a reverse step.}}
\end{{figure}}

The next paired figures retain the representative visual CG comparison from the original
report, now repeated independently for every mean-field setting.

{"".join(cg_visual_sections)}

{_latex_cg_comparison_table(input_root, mean_field_values, seed)}

Increasing $M$ changes the precision fixed point, whereas increasing the CG budget improves
the numerical solution for each fixed precision. These controls must therefore be interpreted
separately. Visual changes should be assessed together with paired RMSE, correlation, filter
statistics, and final-solve residuals. The complete machine-readable results are stored in
\texttt{{metrics.csv}}.

{_latex_failure_analysis(input_root, mean_field_values, seed)}

\section{{Interpretation checklist}}
\begin{{enumerate}}
\item Compare $M$ values at fixed lambda and fixed CG budget before attributing changes to
the mean-field refinement.
\item Use CG-500 to assess the effect of $M$ with less linear-solve error; use CG-50 to test
whether that conclusion is solver-sensitive.
\item A sparse super-Gaussian response should increase near-zero concentration while retaining
heavy tails. Lower image quality or larger first differences alone do not demonstrate sparsity.
\item Runs with $\lambda=1$ skip the GGSM calculation, so their outputs should be identical
across $M$ for the fixed seed. They provide an internal reproducibility check.
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
    if len(mean_field_values) != 3:
        raise ValueError("Exactly three mean-field iteration values are required.")
    assets = output_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    for mean_field_iterations in mean_field_values:
        rows = [
            (
                label,
                _load_samples(
                    input_root, schedule, mean_field_iterations, 500, seed
                ),
            )
            for schedule, label in SCHEDULES
        ]
        for contours, suffix in ((False, "clean"), (True, "contours")):
            _save_row_grid(
                rows,
                assets / f"cg500_m{mean_field_iterations}_{suffix}.png",
                contours=contours,
                title=f"CG-500, M={mean_field_iterations}: lambda schedules",
            )

    for schedule, label in REPRESENTATIVE_SCHEDULES:
        rows = [
            (
                f"M={mean_field_iterations}",
                _load_samples(
                    input_root, schedule, mean_field_iterations, 500, seed
                ),
            )
            for mean_field_iterations in mean_field_values
        ]
        for contours, suffix in ((False, "clean"), (True, "contours")):
            _save_row_grid(
                rows,
                assets / f"m_comparison_cg500_{schedule}_{suffix}.png",
                contours=contours,
                title=f"{label}: mean-field comparison at CG-500",
            )

    for mean_field_iterations in mean_field_values:
        rows = []
        for schedule, label in REPRESENTATIVE_SCHEDULES:
            for cg_iterations in (50, 500):
                rows.append(
                    (
                        f"{label}, CG-{cg_iterations}",
                        _load_samples(
                            input_root,
                            schedule,
                            mean_field_iterations,
                            cg_iterations,
                            seed,
                        ),
                    )
                )
        for contours, suffix in ((False, "clean"), (True, "contours")):
            _save_row_grid(
                rows,
                assets / f"cg_comparison_m{mean_field_iterations}_{suffix}.png",
                contours=contours,
                title=f"M={mean_field_iterations}: representative CG-50 versus CG-500",
            )

    _save_cg_diagnostics(
        input_root, mean_field_values, seed, assets / "cg_diagnostics_by_m.png"
    )
    rows = _metric_rows(input_root, mean_field_values, seed)
    _save_metrics_csv(rows, output_root / "metrics.csv")
    shutil.copy2(input_root / "lambda_schedules.png", assets / "lambda_schedules.png")
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
    mean_field_values = tuple(args.mean_field_iterations)
    build_report(
        args.input_root,
        args.output_root,
        mean_field_values=mean_field_values,
        seed=args.seed,
        compile_pdf=args.compile,
    )
    print(f"Saved LaTeX report to {args.output_root / 'report.tex'}")
    if args.compile:
        print(f"Saved compiled report to {args.output_root / 'report.pdf'}")


if __name__ == "__main__":
    main()
