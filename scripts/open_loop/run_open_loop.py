"""Run one open-loop config and write a log plus a plot. No viewer."""

from __future__ import annotations

import argparse
from pathlib import Path

from controllers.constant import ConstantForce
from experiments.open_loop.config import load_open_loop
from experiments.plot import plot
from experiments.runner import run, save
from sim.plant import Plant


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an open-loop cartpole episode.")
    parser.add_argument("--config", type=Path, default=Path("configs/open_loop/fall.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    experiment, force = load_open_loop(args.config)
    trajectory = run(Plant(), ConstantForce(force), experiment)
    stem = args.out / args.config.parent.name / experiment.name
    save(trajectory, stem.with_suffix(".npz"))
    plot(trajectory, stem.with_suffix(".png"))
    print(f"wrote {stem.with_suffix('.npz')} and {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
