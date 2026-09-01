from pathlib import Path

import torch
from omegaconf import OmegaConf

from gsmdiff.scripts.compare_celebahq_geometric_runs import create_comparison_figure
from gsmdiff.scripts.plot_lambda_schedules import save_lambda_schedule_comparison


def _write_run(path: Path, samples: torch.Tensor, blend: float):
    path.mkdir()
    torch.save(samples, path / "geometric_samples.pt")
    config = OmegaConf.create(
        {
            "runtime": {"seed": 1234, "device": "cpu", "dtype": "float32"},
            "sampling": {
                "method": "reverse_sde",
                "num_inference_steps": 1000,
                "batch_size": 2,
            },
            "model": {"id": "test/celebahq"},
            "blend": {"value": blend},
        }
    )
    OmegaConf.save(config, path / "config.yaml")


def test_four_row_comparison_figure_is_created_for_paired_runs(tmp_path):
    initial_noise = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(1234))
    baseline_samples = initial_noise.tanh()
    sparse_samples = (initial_noise * 0.9).tanh()
    baseline_run = tmp_path / "baseline"
    sparse_run = tmp_path / "sparse"
    _write_run(baseline_run, baseline_samples, 1.0)
    _write_run(sparse_run, sparse_samples, 0.99)

    output = tmp_path / "comparison.png"
    assert create_comparison_figure(
        baseline_run,
        sparse_run,
        output,
        num_samples=2,
    ) == output
    assert output.is_file()
    assert output.stat().st_size > 0


def test_lambda_schedule_comparison_plot_is_created(tmp_path):
    output = tmp_path / "lambda_schedules.png"
    assert save_lambda_schedule_comparison(output, num_steps=50) == output
    assert output.is_file()
    assert output.stat().st_size > 0
