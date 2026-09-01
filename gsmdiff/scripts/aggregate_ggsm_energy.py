"""Aggregate per-run GGSM filter-energy moments across generated batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GGSM energy from run metrics JSON.")
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    count = 0
    first_moment = 0.0
    second_moment = 0.0
    per_coefficient = 0.0
    total_energy = 0.0
    for path in args.metrics:
        values = json.loads(path.read_text())
        batch_count = int(values["samples/count"])
        mean = float(values["ggsm_energy/geometric_filter_sum_mean"])
        std = float(values["ggsm_energy/geometric_filter_sum_std"])
        count += batch_count
        first_moment += batch_count * mean
        second_moment += batch_count * (std * std + mean * mean)
        per_coefficient += batch_count * float(
            values["ggsm_energy/geometric_filter_mean_per_coefficient"]
        )
        total_energy += batch_count * float(values["ggsm_energy/geometric_total_mean"])
    mean = first_moment / count
    variance = max(0.0, second_moment / count - mean * mean)
    result = {
        "sample_count": count,
        "ggsm_filter_energy_sum_mean": mean,
        "ggsm_filter_energy_sum_std": variance**0.5,
        "ggsm_filter_energy_mean_per_coefficient": per_coefficient / count,
        "ggsm_total_energy_mean": total_energy / count,
        "source_metrics": [str(path) for path in args.metrics],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
