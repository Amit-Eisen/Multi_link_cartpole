"""Balance the upright plant with full-state PID."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

from controllers.pid import PID
from controllers.place import pole_placement
from experiments.metrics import converged, evaluate
from experiments.pid.config import load_pid
from experiments.plot import plot
from experiments.runner import Experiment, run, save
from sim.angles import absolute_link_angles
from sim.linearize import linearize
from sim.plant import Plant


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one PID balance episode.")
    parser.add_argument("--config", type=Path, default=Path("configs/pid/balance.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--view", action="store_true", help="open the MuJoCo window")
    args = parser.parse_args()

    experiment, poles, ki, n_links = load_pid(args.config)
    plant = Plant(n_links=n_links)
    model = linearize(plant)
    gain = pole_placement(model.A, model.B, poles)
    policy = PID(gain, ki=ki)
    trajectory = run(plant, policy, experiment)
    stem = args.out / args.config.parent.name / experiment.name
    save(trajectory, stem.with_suffix(".npz"))
    plot(trajectory, stem.with_suffix(".png"))
    metrics = evaluate(trajectory)
    pole_row = " ".join(f"{value:.4g}" for value in poles)
    gain_row = " ".join(f"{value:+.4f}" for value in np.ravel(gain))
    settling = "not settled" if metrics.settling_time is None else f"{metrics.settling_time:.3f} s"
    print(f"poles = {pole_row}")
    print(f"K = {gain_row}")
    print(f"ki = {ki}")
    print(f"settling time {settling}")
    print("final sample in band" if converged(trajectory) else "final sample outside the band")
    print(f"wrote {stem.with_suffix('.npz')} and {stem.with_suffix('.png')}")

    if args.view:
        play(plant, policy, experiment)


def play(plant: Plant, policy: PID, experiment: Experiment) -> None:
    """Keep balancing in the viewer until the window is closed."""
    plant.reset(qpos=np.asarray(experiment.qpos), qvel=np.asarray(experiment.qvel))
    frame = 0.01
    substeps = max(1, int(round(frame / plant.timestep)))
    next_print = 0.0

    with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            state = plant.state()

            for _ in range(substeps):
                state = plant.step(policy(state))

            if state.time >= next_print:
                degrees = np.degrees(absolute_link_angles(state.qpos))
                links = "  ".join(f"link{i + 1}={angle:+6.1f} deg" for i, angle in enumerate(degrees))
                print(f"t={state.time:5.2f} s  x={state.qpos[0]:+.3f} m  {links}")
                next_print += 0.5

            viewer.sync()
            remain = frame - (time.perf_counter() - start)

            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
