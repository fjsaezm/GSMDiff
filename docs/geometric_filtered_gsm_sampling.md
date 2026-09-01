# Geometric diffusion sampling with a filtered GSM prior

## 1. Objective

This implementation combines an unconditional diffusion density `q(x)` with the sparse
filtered-image density `p(x)` from Equation 5 of *Deep Sparse Prior*:

\[
r_0(x) \propto q_0(x)^\lambda p_0(x)^{1-\lambda}, \qquad 0\leq\lambda\leq1,
\]

\[
p_0(x) \propto
\prod_{n=1}^{N}\prod_i f\!\left((F^{(n)}x)_i\right).
\]

The experiment uses the horizontal and vertical first-difference filters. The scalar density
`f` is one of the Gaussian scale-mixture (GSM) families already provided by this repository.
The model is sampled in the normalized image coordinates used by the pretrained diffusion
model, normally `[-1, 1]`.

The original geometric score is the weighted sum

\[
s_r(x_t,t)=\lambda(t)s_q(x_t,t)+(1-\lambda(t))s_p(x_t,t).
\]

A constant `lambda(t)` implements the fixed geometric mixture in the definition of `r_0`.
Linear and cosine schedules are also implemented as useful annealing experiments, but a
time-varying lambda should not be described as exact sampling from one fixed Equation 25
density.

For the follow-up experiment motivated by Section 5.6 of the lambda/CG/M report, use
`sampling.score_combination=additive` instead:

\[
s_{\mathrm{total}}(x_t,t)=s_q(x_t,t)+\gamma(t)s_p(x_t,t).
\]

This preserves the complete learned score. `guidance=constant` selects a fixed non-negative
`guidance.value`; `guidance=power_growth` increases gamma from zero to `guidance.maximum`
near clean time. Gamma zero is exactly the diffusion baseline and skips the prior solve.

## 2. Diffusion score

The forward diffusion convention is

\[
x_t=\alpha_t x_0+\sigma_t z, \qquad z\sim\mathcal N(0,I),
\]

where

\[
\alpha_t=\sqrt{\bar\alpha_t}, \qquad
\sigma_t=\sqrt{1-\bar\alpha_t}.
\]

For a Diffusers checkpoint that predicts noise `epsilon_theta`, the score and clean estimate are

\[
s_q(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sigma_t},
\qquad
\hat x_{0,q}=\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}.
\]

`DiffusersScoreModel` also converts checkpoints using clean-sample or velocity prediction.
This conversion is kept outside the sampler so the geometric code always consumes a score
with one unambiguous convention.

## 3. Clean filtered-GSM score

Let `u^(n) = F^(n)x`. For an elementwise GSM,

\[
f(u_i)=\int \mathcal N(u_i\mid0,\omega_i^{-1})\,\pi(\omega_i)\,d\omega_i.
\]

Its conditional mean precision satisfies

\[
\bar\omega_i(u_i)=E[\omega_i\mid u_i]
=-\frac{\partial\log f(u_i)/\partial u_i}{u_i}.
\]

Writing `Omega^(n)(x) = Diag(bar_omega^(n)(F^(n)x))`, the clean score is

\[
\nabla_x\log p_0(x)
=-\sum_n F^{(n)\top}\Omega^{(n)}(x)F^{(n)}x.
\]

This is the score implemented by `FirstDifferenceFilterProduct`. The weak Gaussian factor on
each channel's spatial mean completes the otherwise unconstrained constant-image direction.
It makes `p_0` a proper standalone density and is included in every precision operation.

## 4. Noisy filtered-GSM score

The scalar expression in Equation 37 of *Sampling GSM Models* assumes a pixelwise diagonal
precision and is not valid for the filter product. Conditioned on all filter precisions, the clean
image instead has precision

\[
P(\omega)=\sum_n F^{(n)\top}\Omega^{(n)}F^{(n)}+P_{\mathrm{mean}}.
\]

Gaussian conditioning then gives

\[
E[x_0\mid x_t,\omega]
=\alpha_t(\alpha_t^2I+\sigma_t^2P(\omega))^{-1}x_t.
\]

The posterior over `omega` given the noisy image is intractable. Following Algorithm 2's
variational mean-field approximation, the implementation initializes the clean estimate with
the diffusion denoised image and then alternates precision updates and conditional-mean solves:

\[
\hat\omega_i^{(n)}
=E\!\left[\omega_i^{(n)}\mid
             (F^{(n)}\hat x_0^{(m-1)})_i\right],
\]

\[
\hat P_t^{(m)}=\sum_nF^{(n)\top}\operatorname{Diag}(\hat\omega^{(n)})F^{(n)}
          +P_{\mathrm{mean}}.
\]

It computes

\[
\hat x_0^{(m)}=\alpha_t(\alpha_t^2I+\sigma_t^2\hat P_t^{(m)})^{-1}x_t.
\]

After `prior.mean_field_iterations` rounds, it applies Tweedie's identity:

\[
s_p(x_t,t)=-\frac{x_t-\alpha_t\hat x_0^{(M)}}{\sigma_t^2}.
\]

The linear system is solved by batched preconditioned conjugate gradients. `P_hat` is applied
as a spatially varying weighted Laplacian; no dense `CHW x CHW` matrix is constructed. The
Jacobi preconditioner is the exact diagonal of the filter precision plus the diffusion term.
Positive GSM smoothing is required for families whose expected precision is singular at zero.

## 5. Sampling algorithms

At every reverse step the sampler performs:

1. Evaluate the pretrained diffusion model and convert its output into `s_q` and `x0_hat_q`.
2. Evaluate the GSM precisions on horizontal and vertical differences of the current clean
   estimate.
3. Solve the filtered Gaussian conditioning system, update the clean estimate, and repeat steps
   2--3 for the configured number of mean-field iterations `M`.
4. Obtain `s_p` from the final conditional mean.
5. Evaluate the configured lambda schedule and construct `s_total`.
6. Advance with the selected reverse integrator.

The `reverse_sde` method implements Algorithm 2 of `Sampling_GSM_models-2`. When inference
timesteps are skipped, it uses
the effective transition

\[
a=\frac{\bar\alpha_t}{\bar\alpha_{t^-}},\qquad b=1-a,
\]

and updates

\[
x_{t^-}=\frac{x_t+b\,s_r(x_t,t)}{\sqrt a}
 +\sqrt{\frac{1-\bar\alpha_{t^-}}{1-\bar\alpha_t}b}\,z.
\]

The stochastic coefficient is the DDPM posterior standard deviation
`sigma_(t->t-)`, and noise is omitted on the final step. The `ddim` method implements the
deterministic Algorithm 3 update:

\[
\hat x_{0,r}=\frac{x_t+\sigma_t^2s_r(x_t,t)}{\alpha_t},
\]

\[
x_{t^-}=\alpha_{t^-}\hat x_{0,r}
          -\sigma_{t^-}\sigma_t s_r(x_t,t).
\]

Both are approximations in the same sense as the source algorithm: the filtered GSM latent
precisions use a plug-in posterior estimate. More generally, the product of separately noised
marginals `q_t^lambda p_t^(1-lambda)` is not guaranteed to be the forward marginal of the
clean geometric product. Results should therefore be reported as geometric score sampling,
not as an exact finite-step draw from a known normalized density.

## 6. CelebA-HQ experiment

Install the diffusion dependencies:

```bash
python -m pip install -e ".[diffusion,dev]"
```

Run the default stochastic experiment with `google/ddpm-celebahq-256`:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq
```

Useful variants are:

```bash
# Fixed geometric density weight
python -m gsmdiff.scripts.sample_geometric_celebahq blend.value=0.8

# Anneal from the diffusion model toward a stronger sparse prior
python -m gsmdiff.scripts.sample_geometric_celebahq \
  blend=cosine blend.start=1.0 blend.end=0.6

# Deterministic Algorithm 3 and a different GSM
python -m gsmdiff.scripts.sample_geometric_celebahq \
  sampling.method=ddim sampling.num_inference_steps=50 \
  distribution=generalized_gaussian
```

The run saves:

- `geometric_samples.pt` and, by default, a matched-initial-noise lambda-one baseline;
- a clean comparison grid;
- `samples_with_high_pass_contours.png`, using the same first differences as the prior;
- `lambda_schedule.png`, showing the diffusion-model weight over reverse-process time;
- `filter_sparsity.png`, containing a log-density histogram and near-zero concentration curve;
- scalar sparsity metrics, including near-zero fraction and excess kurtosis;
- GGSM filter energy, both as the requested per-image sum
  $\sum_i\rho((Fx)_i)$ and normalized per filter coefficient;
- per-timestep score norms, lambda values, CG iterations, and residuals.
- per-timestep prior weights, weighted prior-score norms, and cosine similarity between the
  diffusion and prior scores.

### Calibrating the prior scale for each dataset

The configured scale of a hyperbolic-secant density is not its standard deviation. For
`p(d) = sech(d / scale) / (pi * scale)`, `std(d) = pi * scale / 2`. The calibration command
computes the pooled population standard deviation (`correction=0`) of signed horizontal and
vertical first differences in clean NCHW image tensors, then uses `scale = 2 * std / pi`:

```bash
python -m gsmdiff.scripts.calibrate_filter_scale \
  /path/to/clean_dataset_batches.pt \
  --distribution hyperbolic_secant \
  --output outputs/celebahq_prior_scale.json
```

Run this once per dataset and image normalization. Images must use the same coordinates as the
diffusion model (CelebA-HQ is normally `[-1, 1]`). Dataset samples are preferred; a sufficiently
large diffusion-only sample batch is a model-matched fallback.

The lambda-one baseline skips the filtered-prior linear solve. With `ddim` it shares the entire
deterministic trajectory's initial noise; with `reverse_sde` it also uses the same random seed for
the reverse noise increments.

## 7. Interpreting the diagnostics

The yellow overlay marks the strongest configured quantile of combined horizontal/vertical
first-difference magnitudes. It visualizes where the prior sees large, edge-like coefficients; it
is not itself a binary sparsity measure.

The histogram standardizes each sample group independently, separating distribution shape
from global contrast. A sparse super-Gaussian response distribution should generally have both
a larger concentration close to zero and heavier tails than a Gaussian-shaped response
distribution. The near-zero CDF panel and excess kurtosis metric quantify those two effects.

CG residuals should remain close to or below the configured tolerance. Repeatedly reaching the
maximum iteration count with large residuals indicates that the noisy-prior score is inaccurate;
increase `prior.cg.max_iterations`, relax the GSM singularity with smoothing, or use float32.

## 8. Command reference

Run all commands from the repository root.

### Installation

```bash
python -m pip install -e ".[diffusion,dev]"
```

### Tests and lint

```bash
pytest -q
ruff check .
```

### Default CelebA-HQ experiment

This uses the stochastic reverse-SDE sampler, 100 inference steps, four samples, the
hyperbolic-secant filtered GSM, fixed `lambda=0.95`, and a matched diffusion-only baseline:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq
```

### Paired lambda=1 and lambda=0.99 experiment with 200-step checkpoints

The following two commands use the hyperbolic-secant GSM, a constant blend, 1000 reverse
steps, and exactly the same initial-noise and reverse-noise seeds. The sampler saves a
checkpoint after 200, 400, 600, 800, and 1000 completed reverse steps. Set the same visible
GPU for both runs.

```bash
CUDA_VISIBLE_DEVICES=0 python -m gsmdiff.scripts.sample_geometric_celebahq \
  distribution=hyperbolic_secant blend=constant blend.value=1.0 \
  sampling.num_inference_steps=1000 sampling.batch_size=4 \
  checkpoints.every_steps=200 comparison.generate_diffusion_baseline=false \
  runtime.seed=20260811 runtime.device=cuda runtime.deterministic=true \
  run.name=celebahq_sech_lambda100_seed20260811

CUDA_VISIBLE_DEVICES=0 python -m gsmdiff.scripts.sample_geometric_celebahq \
  distribution=hyperbolic_secant blend=constant blend.value=0.99 \
  sampling.num_inference_steps=1000 sampling.batch_size=4 \
  checkpoints.every_steps=200 comparison.generate_diffusion_baseline=false \
  runtime.seed=20260811 runtime.device=cuda runtime.deterministic=true \
  run.name=celebahq_sech_lambda099_seed20260811
```

Each `checkpoints/step_NNNN` directory contains:

- `samples.png`, using the guided clean estimate (or final reverse state at the last step);
- `samples_with_high_pass_contours.png`, showing the same images with filter contours.

Intermediate tensors and metadata are intentionally not saved. The final run directory retains
`geometric_samples.pt` for quantitative analysis. Initial noise is regenerated deterministically
from `runtime.seed` and is not written to disk.

Create the final four-row figure (lambda=1 clean/contours, then lambda=0.99 clean/contours):

```bash
python -m gsmdiff.scripts.compare_celebahq_geometric_runs \
  --baseline-run outputs/celebahq_sech_lambda100_seed20260811 \
  --sparse-run outputs/celebahq_sech_lambda099_seed20260811 \
  --output outputs/celebahq_lambda100_vs_lambda099_4row.png \
  --num-samples 4
```

The comparison script verifies the relevant seed, model, device, dtype, and sampling settings
before plotting. A mismatch raises an error rather than producing an unpaired comparison.

### Four-GPU lambda, CG, and mean-field grid

Launch the complete 36-run grid on GPUs 0, 3, 4, and 7:

```bash
bash bash_scripts/run_celebahq_lambda_cg_grid.sh
```

The launcher maintains exactly four active processes. Whenever a GPU finishes, the next queued
experiment is assigned to it. It runs six lambda schedules with CG budgets 500 and 50 for each
of $M\in\{1,5,10\}$ variational mean-field iterations:

- constants 1.0, 0.99, 0.9, and 0.8;
- the fifth-power decay $1-(t/(N-1))^5$;
- the tenth-power decay $1-(t/(N-1))^{10}$, which retains the diffusion score longer and
  reaches exactly zero on the last step.

To run only one schedule while retaining automatic report generation, set `ONLY_SCHEDULE`.
For example, after the other five schedules are complete:

```bash
ONLY_SCHEDULE=power10_1_to_0 bash bash_scripts/run_celebahq_lambda_cg_grid.sh
```

Results are written under `outputs/lambda_cg_m_grid_seed20260811`. The run name records both
`mM` and `cgK`. The combined schedule plot is `lambda_schedules.png`, and one log per job is
stored in `launcher_logs`. After all jobs finish, the launcher generates and compiles
`reports/lambda_cg_m_grid_seed20260811/report.tex`. Environment variables can override the
main settings, for example:

```bash
PYTHON_BIN=/path/to/python OUTPUT_DIR=/path/to/results REPORT_DIR=/path/to/report BATCH_SIZE=8 \
  bash bash_scripts/run_celebahq_lambda_cg_grid.sh
```

### Additive-guidance gamma, CG, and mean-field grid (report Section 5.6)

The follow-up launcher preserves the full learned score and moment-matches the sech prior.
Prefer supplying clean, normalized dataset tensors:

```bash
PYTHON_BIN=/path/to/python \
CALIBRATION_SAMPLES=/path/to/clean_celebahq_nchw.pt \
bash bash_scripts/run_celebahq_gamma_cg_m_grid.sh
```

Without `CALIBRATION_SAMPLES`, the launcher first generates an eight-image diffusion-only
calibration batch. The default candidate set is now restricted to gamma 0, constant gamma
0.005 and 0.01, and the late power-10 schedule ending at 0.1. The destructive constant
gamma 0.1 configuration is not launched. `SEED_COUNT` creates consecutive paired seeds and
allows enough saved batches to be accumulated for distributional metrics.
Use a cheap pilot before the complete grid:

```bash
PYTHON_BIN=/path/to/python NUM_STEPS=100 BATCH_SIZE=2 GPU_IDS="0" \
MEAN_FIELD_VALUES="5" CG_VALUES="500" ONLY_GUIDANCE=gamma0p01 \
bash bash_scripts/run_celebahq_gamma_cg_m_grid.sh
```

The calibration JSON is saved beside the results. Every run config records the fitted
`distribution.scale`; diagnostics make it possible to compare `gamma * RMS(s_p)` with
`RMS(s_q)` and to detect conflicting guidance through negative cosine similarity.
`GPU_IDS`, `MEAN_FIELD_VALUES`, and `CG_VALUES` accept space-separated subsets, so the first
full-length gamma sweep can sensibly use `MEAN_FIELD_VALUES="5" CG_VALUES="500"` before
spending compute on numerical-sensitivity controls.

### FID, KID, and aggregate GGSM energy

Install the optional metric dependencies:

```bash
python -m pip install -e ".[diffusion,metrics,dev]"
```

FID and KID require real CelebA-HQ reference images in the same data domain. They cannot be
computed meaningfully from the two-image pilot. The evaluator accepts a recursively searched
image directory or one or more NCHW tensor files. Generated tensors are converted from
`[-1,1]` to uint8 RGB; directory images are converted to RGB and resized to 256 square before
the common Inception preprocessing.

The following screening run produces 1,000 generated images per candidate as 250 batches of
four. All candidates use identical seeds:

```bash
PYTHON_BIN=/path/to/python \
OUTPUT_DIR=outputs/gamma_metrics_seed20260811 \
PRIOR_SCALE=0.03226316360572 \
NUM_STEPS=100 BATCH_SIZE=4 SEED_COUNT=250 \
GPU_IDS="0 3 4 7" MEAN_FIELD_VALUES="5" CG_VALUES="500" \
CHECKPOINT_EVERY=null CONTOUR_QUANTILE=0.98 \
bash bash_scripts/run_celebahq_gamma_cg_m_grid.sh
```

Use a scale calibrated from a large real CelebA-HQ subset instead of the shown pilot scale
when available. For final FID, increase to at least 5,000 generated images per candidate
(`SEED_COUNT=1250` with batch size four). Set `NUM_STEPS=1000` if that is the sampler being
reported; scores obtained with 100 and 1000 steps are different experiments and must not be
mixed.

After generation, evaluate every retained candidate against one fixed real reference set:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/path/to/python \
OUTPUT_DIR=outputs/gamma_metrics_seed20260811 \
REAL_IMAGES=/path/to/real/celebahq/images \
METRIC_DEVICE=cuda \
bash bash_scripts/evaluate_celebahq_gamma_metrics.sh
```

This writes one FID/KID JSON and one aggregate GGSM-energy JSON per candidate under
`distribution_metrics/`. The default guard refuses fewer than 1,000 real or generated images.
FID uses 2048-dimensional Inception-v3 features and float64 accumulation. KID reports its mean
and standard deviation over 100 subsets of size 1,000. GGSM energy is the exact configured
sum of hyperbolic-secant potentials; lower energy means stronger agreement with this prior but
can also reward blur, so it must be interpreted together with FID, KID, and the filter-shape
statistics.

Contour previews now default to the per-image 98th percentile, showing only the strongest two
percent of responses. Override it with `CONTOUR_QUANTILE=0.99`. A per-image quantile always
marks the same fraction and therefore compares edge locations, not edge counts. For comparable
edge density, set one shared absolute cutoff for every run with
`CONTOUR_THRESHOLD=<value>` (typically estimated from the gamma-zero baseline).

Give the run a stable output name:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  run.name=celebahq_geometric_sech_lambda095_seed0
```

### Deterministic DDIM experiment

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  sampling.method=ddim \
  sampling.num_inference_steps=50 \
  run.name=celebahq_geometric_ddim50
```

### Fixed lambda sweep

Hydra can launch one run per fixed geometric-mixture weight:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq --multirun \
  blend.value=1.0,0.98,0.95,0.9,0.8 \
  runtime.seed=0 \
  run.name=null
```

`lambda=1` is the diffusion-only endpoint. Lower values increase the filtered-GSM contribution.
Values much below `0.9` can overpower the denoiser and should be treated as exploratory.

### Scheduled lambda experiments

Linear schedule from the diffusion model at high noise to `lambda=0.9` near the clean image:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  blend=linear blend.start=1.0 blend.end=0.9 \
  run.name=celebahq_linear_lambda_100_to_090
```

Cosine schedule with the same endpoints:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  blend=cosine blend.start=1.0 blend.end=0.9 \
  run.name=celebahq_cosine_lambda_100_to_090
```

Compare several final scheduled weights:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq --multirun \
  blend=cosine blend.start=1.0 blend.end=0.98,0.95,0.9,0.85
```

### GSM-family comparison

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq --multirun \
  distribution=hyperbolic_secant,laplace,generalized_gaussian,log \
  blend.value=0.95
```

The Laplace and generalized-Gaussian configurations use positive smoothing. Keep smoothing
positive in diffusion experiments because an infinite expected precision at a zero filter
coefficient makes the noisy-prior solve undefined.

### More samples and seeds

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  sampling.batch_size=8 image.num_preview=8 runtime.seed=1 \
  run.name=celebahq_geometric_seed1
```

Seed sweep:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq --multirun \
  runtime.seed=0,1,2,3,4 \
  sampling.batch_size=4
```

### CG accuracy experiments

The default has three Algorithm 2 mean-field iterations per reverse step and a maximum of 200
CG iterations per inner solve. To compare cold starts and warm starts with the same seed,
reverse noise, lambda, and checkpoint schedule, run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m gsmdiff.scripts.sample_geometric_celebahq \
  distribution=hyperbolic_secant blend=constant blend.value=0.8 \
  sampling.num_inference_steps=1000 sampling.batch_size=4 \
  checkpoints.every_steps=200 comparison.generate_diffusion_baseline=false \
  prior.cg.max_iterations=200 prior.cg.warm_start=false \
  runtime.seed=20260811 runtime.device=cuda runtime.deterministic=true \
  run.name=celebahq_sech_lambda080_cg200_cold_seed20260811

CUDA_VISIBLE_DEVICES=0 python -m gsmdiff.scripts.sample_geometric_celebahq \
  distribution=hyperbolic_secant blend=constant blend.value=0.8 \
  sampling.num_inference_steps=1000 sampling.batch_size=4 \
  checkpoints.every_steps=200 comparison.generate_diffusion_baseline=false \
  prior.cg.max_iterations=200 prior.cg.warm_start=true \
  runtime.seed=20260811 runtime.device=cuda runtime.deterministic=true \
  run.name=celebahq_sech_lambda080_cg200_warm_seed20260811
```

Warm starting reuses the preceding timestep's linear solution as the next initial guess. The
cache is local to one call of the sampler and is reset for every new trajectory. Within one
timestep, the solution from mean-field iteration `m` initializes iteration `m+1`. Override the
outer iteration count with, for example, `prior.mean_field_iterations=1` or
`prior.mean_field_iterations=5`. Compare `sampling/cg_mean_total_iterations` and
`sampling/cg_max_relative_residual` in `metrics.json`, and inspect the full per-step diagnostics.

Use a cheaper exploratory solve only for smoke tests:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  prior.cg.max_iterations=20 \
  prior.cg.tolerance=1e-2 \
  run.name=celebahq_geometric_fast_cg
```

Always inspect `step_diagnostics.json` when changing these values. A small iteration budget is
not reliable if the recorded relative residual remains large.

### Fast smoke test

This checks checkpoint loading and the complete execution path but is not long enough to assess
image quality:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  sampling.batch_size=1 \
  sampling.num_inference_steps=2 \
  prior.cg.max_iterations=2 \
  prior.cg.tolerance=0.2 \
  comparison.generate_diffusion_baseline=false \
  image.num_preview=1 \
  run.name=celebahq_api_smoke
```

### Disable the comparison baseline

This avoids the second diffusion trajectory and approximately halves UNet inference work:

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  comparison.generate_diffusion_baseline=false
```

### Select a GPU

```bash
CUDA_VISIBLE_DEVICES=0 \
python -m gsmdiff.scripts.sample_geometric_celebahq runtime.device=cuda
```

Keep `runtime.dtype=float32` for the conjugate-gradient solve. Lower-precision CG may become
unstable for the poorly conditioned systems encountered at high noise levels.

### W&B logging

```bash
wandb login
python -m gsmdiff.scripts.sample_geometric_celebahq \
  wandb.mode=online \
  wandb.project=gsmdiff \
  wandb.name=celebahq_geometric_lambda095
```

### Change the output directory

```bash
python -m gsmdiff.scripts.sample_geometric_celebahq \
  run.output_dir=/path/to/experiment_outputs \
  run.name=celebahq_geometric_run
```
