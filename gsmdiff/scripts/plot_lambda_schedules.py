"""Plot the six lambda schedules used by the queued CelebA-HQ experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gsmdiff.geometric import ConstantLambda, PowerDecayLambda


def save_lambda_schedule_comparison(output: Path, *, num_steps: int = 1000) -> Path:
    if num_steps <= 1:
        raise ValueError("num_steps must be greater than one.")
    schedules = (
        (r"Constant $\lambda=1.00$", ConstantLambda(1.0)),
        (r"Constant $\lambda=0.99$", ConstantLambda(0.99)),
        (r"Constant $\lambda=0.90$", ConstantLambda(0.9)),
        (r"Constant $\lambda=0.80$", ConstantLambda(0.8)),
        (r"$1-(t/(N-1))^5$", PowerDecayLambda(power=5.0)),
        (r"$1-(t/(N-1))^{10}$", PowerDecayLambda(power=10.0)),
    )
    progress = np.linspace(0.0, 1.0, num_steps)

    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.4, 5.2))
    for label, schedule in schedules:
        values = [schedule(index, num_steps) for index in range(num_steps)]
        axis.plot(progress, values, linewidth=2.0, label=label)
    axis.set_xlabel("Normalized reverse-process progress")
    axis.set_ylabel(r"Diffusion-model weight $\lambda$")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower left")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = save_lambda_schedule_comparison(args.output, num_steps=args.num_steps)
    print(f"Saved lambda schedule comparison to {output}")


if __name__ == "__main__":
    main()
