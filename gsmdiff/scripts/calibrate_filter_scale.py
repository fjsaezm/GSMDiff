"""Fit a first-difference density scale from clean image tensor batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor

from gsmdiff.geometric.calibration import (
    coefficient_standard_deviation,
    distribution_scale_from_standard_deviation,
)


def _load_tensor(path: Path) -> Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("samples", "images", "geometric_samples"):
            if key in value and isinstance(value[key], Tensor):
                value = value[key]
                break
    if not isinstance(value, Tensor):
        raise TypeError(f"{path} does not contain a tensor or a recognized tensor mapping.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Moment-match a coefficient prior to clean dataset image tensors."
    )
    parser.add_argument("samples", nargs="+", type=Path, help="One or more clean NCHW .pt files")
    parser.add_argument("--distribution", default="hyperbolic_secant")
    parser.add_argument("--shape", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--print-scale-only", action="store_true")
    args = parser.parse_args()

    standard_deviation, coefficient_count = coefficient_standard_deviation(
        _load_tensor(path) for path in args.samples
    )
    scale = distribution_scale_from_standard_deviation(
        standard_deviation, args.distribution, shape=args.shape
    )
    result = {
        "distribution": args.distribution,
        "sample_files": [str(path) for path in args.samples],
        "coefficient_count": coefficient_count,
        "coefficient_standard_deviation": standard_deviation,
        "distribution_scale": scale,
        "standard_deviation_correction": 0,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{scale:.17g}" if args.print_scale_only else json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
