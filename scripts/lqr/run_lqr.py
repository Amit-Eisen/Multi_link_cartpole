"""Balance the upright plant with LQR."""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import mujoco.viewer
import numpy as np
from controllers.lqr import LQR, lqr_gain
from experiments.lqr.config import load_lqr
from experiments.plot import plot
from experiments.runner import Experiment, run, save
from sim.linearize import linearize
from sim.plant import Plant


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one LQR balance episode.")
    parser.add_argument("--config", type=Path, default=Path("configs/lqr/balance.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--view", action="store_true", help="open the MuJoCo window")
    args = parser.parse_args()

    experiment, state_weight, input_weight = load_lqr(args.config)
    plant = Plant()
    model = linearize(plant)
    policy = LQR(lqr_gain(model.A, model.B, state_weight, input_weight).K)
    trajectory = run(plant, policy, experiment)
    stem = args.out / args.config.parent.name / experiment.name
    save(trajectory, stem.with_suffix(".npz"))
    plot(trajectory, stem.with_suffix(".png"))
    row = " ".join(f"{value:+.4f}" for value in np.ravel(policy.K))
    print(f"K = {row}")
    print(f"wrote {stem.with_suffix('.npz')} and {stem.with_suffix('.png')}")

    if args.view:
        play(plant, policy, experiment)


def play(plant: Plant, policy: LQR, experiment: Experiment) -> None:
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
                degrees = np.degrees(state.qpos[1:])
                print(
                    f"t={state.time:5.2f} s  x={state.qpos[0]:+.3f} m  "
                    f"link1={degrees[0]:+6.1f} deg  link2={degrees[1]:+6.1f} deg"
                )
                next_print += 0.5

            viewer.sync()
            remain = frame - (time.perf_counter() - start)

            if remain > 0:
                time.sleep(remain)


if __name__ == "__main__":
    main()
