# Geometric pretrained-flow/GGSM sampling

This entry point implements Algorithm 5 of `Sampling_GSM_models-2.pdf` with the public
optimal-transport Flow Matching U-Net released by
[PnP-Flow](https://github.com/annegnx/PnP-Flow). It follows the report's forward path

\[
x_t=(1-t)x_0+t x_1, \qquad x_0\sim\mathcal N(0,I),
\]

and uses explicit Euler integration from `t=0` to `t=1`.

## Independent vector-field coefficients

The implementation deliberately generalizes Equations 40--41 to

\[
v_r(x_t,t)=\lambda_q(t)v_\theta(x_t,t)+\lambda_p(t)v_p(x_t,t).
\]

`lambda_q` and `lambda_p` are configured independently as `flow_weight` and
`prior_weight`. They are required to be non-negative, but they do **not** have to sum to one.
The default is `lambda_q=1` and `lambda_p=0.05`. This is an independently weighted composite
vector field; when the coefficients are not complementary, it should not be described as the
report's normalized geometric-density interpolation without an additional derivation.

For each Euler step, the sampler evaluates the learned field once and runs the Algorithm 5
fixed-point loop. Each GGSM refinement solves

\[
[t^2 I+(1-t)^2P(\widehat\Omega)]w_p=x_t
\]

with the repository's matrix-free, preconditioned conjugate-gradient implementation, computes

\[
v_p=t w_p-(1-t)P(\widehat\Omega)w_p,
\]

and updates the composite clean estimate

\[
\widehat x_{1,r}=x_t+(1-t)v_r.
\]

The outer state is then advanced by the report's Euler step.

## Installation and checkpoint

```bash
python -m pip install -e ".[flow,dev]"
```

The default configuration points to PnP-Flow's published CelebA checkpoint. On the first run,
`gdown` stores it at `models/pnp_flow/celeba_ot_128.pt`; later runs reuse the local file. To use
an existing download, set `model.download=false` and override `model.checkpoint`. The configured
SHA-256 checksum is verified before deserialization; set `model.sha256=null` only for a different
trusted checkpoint.

The released PnP-Flow OT checkpoint is trained on center-cropped **CelebA at 128x128**, whereas
this repository's diffusion example uses CelebA-HQ at 256x256. The experiment pipeline and
GGSM prior are the same, but the two pretrained base models and resolutions are not a
like-for-like FID comparison.

## Running

```bash
python -m gsmdiff.scripts.sample_geometric_flow_celeba
```

Choose arbitrary independent constant coefficients:

```bash
python -m gsmdiff.scripts.sample_geometric_flow_celeba \
  flow_weight.value=0.8 prior_weight.value=0.2

python -m gsmdiff.scripts.sample_geometric_flow_celeba \
  flow_weight.value=1.0 prior_weight.value=0.15
```

The second example demonstrates that the coefficients need not sum to one. They can also use
independent linear schedules:

```bash
python -m gsmdiff.scripts.sample_geometric_flow_celeba \
  flow_weight=linear flow_weight.start=1.0 flow_weight.end=0.9 \
  prior_weight=linear prior_weight.start=0.0 prior_weight.end=0.1
```

Increase the integration and fixed-point accuracy with:

```bash
python -m gsmdiff.scripts.sample_geometric_flow_celeba \
  sampling.num_inference_steps=1000 prior.mean_field_iterations=3 \
  prior.cg.max_iterations=500 checkpoints.every_steps=200
```

Each run saves the resolved configuration, `geometric_samples.pt`, an optional matched
`flow_baseline_samples.pt`, previews, high-pass contours, filter statistics, field-weight plot,
CG/velocity diagnostics, and optional intermediate checkpoints. The baseline uses the same
initial Gaussian samples with `lambda_q=1` and `lambda_p=0`.

## Multi-GPU coefficient/solver grid

The flow counterpart of the diffusion launchers runs six field configurations across the
requested mean-field and CG budgets:

```bash
bash bash_scripts/run_celeba_flow_cg_m_grid.sh
```

It tests `q=1` with prior coefficients `0`, `0.005`, `0.01`, and `0.05`; the complementary
pair `q=0.95,p=0.05`; and `q=1` with a prior coefficient growing linearly from `0` to `0.1`.
By default it uses GPUs `0 3 4 7`, mean-field counts `1 5 10`, CG limits `500 50`, 1,000 Euler
steps, and a shared seed. Before launching workers it generates a flow-only calibration batch,
downloads/verifies the model if necessary, and moment-matches the hyperbolic-secant scale.

For a small one-GPU check:

```bash
GPU_IDS="0" MEAN_FIELD_VALUES="1" CG_VALUES="50" \
NUM_STEPS=20 BATCH_SIZE=1 CALIBRATION_BATCH_SIZE=2 CHECKPOINT_EVERY=10 \
ONLY_FIELD=q1_p0p01 \
bash bash_scripts/run_celeba_flow_cg_m_grid.sh
```

Useful environment overrides include `SEED`, `SEED_COUNT`, `OUTPUT_DIR`, `GPU_IDS`,
`MEAN_FIELD_VALUES`, `CG_VALUES`, `ONLY_FIELD`, `PRIOR_SCALE`, `CALIBRATION_SAMPLES`, and
`SKIP_COMPLETED`. `MODEL_CHECKPOINT`, `MODEL_DOWNLOAD`, `MODEL_SHA256`, and `RUNTIME_DEVICE`
are available for custom/local checkpoints and CPU smoke tests. Completed runs are reused by
default.

After the grid finishes, generate the metrics CSV, comparison figures, LaTeX source, and PDF
with the same Python environment used for sampling:

```bash
python -m gsmdiff.scripts.build_flow_cg_report \
  --input-root outputs/flow_cg_m_grid_seed20260811 \
  --output-root reports/flow_cg_m_grid_seed20260811 \
  --seed 20260811 \
  --mean-field-iterations 1 5 10 \
  --compile
```

The generated report compares all six field configurations at CG-500, paired mean-field and
CG-budget changes, solver convergence, weighted prior-to-flow magnitude, and vector-field
alignment. Omit `--compile` when only `report.tex` and the report assets are needed.

## Custom compatible checkpoints

The bundled architecture intentionally retains the module names of `pnpflow.models.UNet`, so
the public state dictionary loads strictly. Architecture settings live under `model` in
`configs/geometric_flow_celeba.yaml`; override all of them when using a differently trained
PnP-Flow U-Net. A different flow architecture needs a small adapter exposing:

- `sample_size` and `in_channels`;
- `velocity(x_t, t)` returning a tensor with the same shape as `x_t`.
