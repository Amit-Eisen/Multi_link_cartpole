"""Watch the 2-link plant. No controller.

Angles on the command line are radians. The printed angles are degrees.
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from sim.plant import Plant


def main() -> None:
    parser = argparse.ArgumentParser(description="View the upright 2-link cartpole.")
    parser.add_argument("--angle", type=float, default=0.1, help="initial hinge_1 angle, rad")
    parser.add_argument("--force", type=float, default=0.0, help="constant cart force, N")
    args = parser.parse_args()

    plant = Plant()
    plant.reset(qpos=[0.0, args.angle, 0.0])
    frame = 0.01
    substeps = max(1, int(round(frame / plant.timestep)))
    next_print = 0.0

    with mujoco.viewer.launch_passive(plant.model, plant.data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            state = plant.state()
            for _ in range(substeps):
                state = plant.step(args.force)
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
