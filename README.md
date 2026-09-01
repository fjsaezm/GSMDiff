# GSMDiff

This implementation samples image-shaped tensors from the analytic super-Gaussian Gaussian
scale mixtures in Sections 3 and 4 of
[the accompanying report](Sampling_GSM_models-1.pdf). It follows the
package/config/entrypoint separation of PnP-Flow and the Hydra, OmegaConf, testing, output,
and W&B conventions of `flow_ood`.

The implementation deliberately separates three concerns:

- `gsmdiff.distributions`: analytic potential, score, and `E[w | x]` for each GSM family;
- `gsmdiff.samplers`: reusable ULA and MALA transition kernels and chain orchestration;
- `gsmdiff.scripts`: Hydra-composed experiments, persistence, previews, and W&B tracking.

The image target applies the selected GSM density directly to horizontal and vertical
first-difference responses. Its score maps the analytic coefficient score back through the
adjoint filters, as in Section 4 of `notes/filtered_histogram_similarity.tex`. A weak Gaussian
factor on each channel's spatial mean resolves the filters' constant-image null space; its
scale is configured by `prior.mean_scale`.

## Installation and tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
```

W&B is installed as a first-class dependency but defaults to disabled mode, so local runs do
not require credentials.

## Sampling

The default is MALA with the hyperbolic-secant (`-logsech`) distribution:

```bash
python -m gsmdiff.scripts.sample
```

Choose any legacy GSM family and sampler through Hydra:

```bash
python -m gsmdiff.scripts.sample distribution=laplace sampler=ula
python -m gsmdiff.scripts.sample distribution=generalized_gaussian sampler=mala
python -m gsmdiff.scripts.sample distribution=gaussian sampler=mala
python -m gsmdiff.scripts.sample distribution=log sampler=mala wandb.mode=online
```

Any setting can be overridden without changing code:

```bash
python -m gsmdiff.scripts.sample \
  distribution=hyperbolic_secant \
  sampler=mala sampler.step_size=1e-2 \
  image.shape='[3,64,64]' sampling.num_chains=32 \
  sampling.burn_in=5000 sampling.draws_per_chain=4 sampling.thinning=50 \
  runtime.device=cuda run.name=sech_rgb_seed0
```

For example, rerun the four-image filtered-prior comparison with an explicit output name:

```bash
python -m gsmdiff.scripts.sample \
  distribution=hyperbolic_secant sampler=mala \
  sampling.num_chains=4 sampling.draws_per_chain=1 \
  run.name=filtered_sech_comparison
```

Each run writes the fully resolved config, lossless tensor samples, preview, and diagnostics:

```text
outputs/<distribution>_<sampler>_<timestamp>/
├── config.yaml
├── filtered_coefficients_histogram.png
├── gaussian_baseline_samples.pt
├── gsm_vs_gaussian.png
├── gsm_vs_gaussian_with_high_pass_contours.png
├── metrics.json
├── samples.png
├── samples_with_high_pass_contours.png
└── samples.pt
```

The second preview overlays thresholded horizontal and vertical first-difference responses in
yellow, matching the high-pass filters used by the original `GSMLoss`. The overlay defaults to
the strongest ten percent of responses per image and is configured under
`image.high_pass_contours`. Set an absolute response cutoff with
`image.high_pass_contours.threshold=<value>` when comparisons need a shared threshold.

The comparison images place exact i.i.d. Gaussian reference samples above the selected GSM
samples. The Gaussian images are variance-matched to the GSM images by default. The contour
comparison uses one shared threshold estimated from the Gaussian responses, rather than
selecting the same percentage of pixels from both groups. `filtered_coefficients_histogram.png`
pools the signed horizontal and vertical filter coefficients from each group. By default each
group is standardized independently and the histogram has both linear-density and log-density
panels, making differences in central concentration and tail mass visible without confounding
them with marginal scale. These choices are configurable under `comparison`.

MALA uses exact unnormalized log-density differences by default. To exercise Equations 11--12
using Gauss-Legendre integration of only the score, set
`sampler.density_ratio=path_integral`.

## Mapping from the legacy GSM loss

| Legacy name | Class | Precision expectation |
|---|---|---|
| `l2` | `Gaussian` | `1 / scale^2` |
| `l1` | `Laplace` | inverse coefficient magnitude (smoothed if configured) |
| `l0.8` | `GeneralizedGaussian` | smoothed coefficient magnitude raised to `shape - 2` |
| `-logsech` / `logsech` | `HyperbolicSecant` | `tanh(x/scale) / (scale*x)` with its exact limit at zero |
| `log` | `SmoothedPowerLaw` | `tail_exponent / (x^2 + scale^2)` |

The report's literal `log(epsilon + abs(x))` with unit exponent is not integrable over the real
line and therefore is not a probability distribution. `SmoothedPowerLaw` makes this model
proper by requiring `tail_exponent > 1`; exponent two retains the inverse-square reweighting
used by the legacy loss.

The legacy `bu3` branch would imply generalized-Gaussian shape `-1`. That produces a
non-normalizable function rather than a probability density, so it is intentionally rejected
instead of being exposed as a misleading sampler target.

## Geometric diffusion/GSM sampling on CelebA-HQ

The repository implements Algorithm 2 of `Sampling_GSM_models-2` for a geometric mixture of a
pretrained Diffusers score and the filtered GSM prior from Equation 5 of *Deep Sparse Prior*.
The noisy filtered-prior score uses a matrix-free weighted-Laplacian conjugate-gradient solve;
it does not make the incorrect independent-pixel approximation. Precision estimation and the
linear solve are alternated for a configurable number of variational mean-field iterations, and
the stochastic transition uses the DDPM posterior variance from the algorithm.

Install and run it with:

```bash
python -m pip install -e ".[diffusion,dev]"
python -m gsmdiff.scripts.sample_geometric_celebahq
```

The default checkpoint is `google/ddpm-celebahq-256`. Constant, linear, and cosine lambda
schedules and stochastic reverse-SDE or deterministic DDIM updates are selectable through
Hydra. Runs include a matched diffusion-only baseline, yellow high-pass overlays, filter-response
histograms, sparsity curves and metrics, and solver diagnostics.

See [the geometric filtered-GSM sampling guide](docs/geometric_filtered_gsm_sampling.md) for
the derivation, algorithm, configuration examples, and approximation caveats.

## Geometric pretrained-flow/GSM sampling on CelebA

Algorithm 5 of the accompanying report is implemented for the public PnP-Flow OT Flow
Matching checkpoint. The learned and GGSM vector fields have separate Hydra schedules, so
their coefficients need not sum to one:

```bash
python -m pip install -e ".[flow,dev]"
python -m gsmdiff.scripts.sample_geometric_flow_celeba \
  flow_weight.value=1.0 prior_weight.value=0.05
```

The first run downloads PnP-Flow's released 128x128 CelebA weights. It produces the same class
of tensor samples, matched base-model baseline, previews, contours, filter statistics,
snapshots, and solver diagnostics as the diffusion experiment. See
[the flow/GGSM guide](docs/geometric_flow_gsm_sampling.md) for the exact Algorithm 5 mapping,
checkpoint details, and configuration examples.

Run the flow coefficient/CG/fixed-point grid with:

```bash
bash bash_scripts/run_celeba_flow_cg_m_grid.sh
```

The report Section 5.6 follow-up is implemented by additive guidance
`s_total = s_q + gamma*s_p`, dataset-specific first-difference scale calibration, and the
four-GPU launcher `bash_scripts/run_celebahq_gamma_cg_m_grid.sh`. The original lambda launcher
is retained for reproducibility.

Generated runs record the calibrated GGSM filter energy. Dataset-backed FID and KID can be
aggregated across paired seed batches with `bash_scripts/evaluate_celebahq_gamma_metrics.sh`;
install the `metrics` optional dependencies and provide a real CelebA-HQ image set.
