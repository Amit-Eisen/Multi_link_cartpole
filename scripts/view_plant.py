"""Watch the plant. No controller.

``--angle`` is the first hinge, in radians. The printout is absolute degrees.
"""

from __future__ import annotations

import argparse
import time

import mujoco.viewer
import numpy as np

from sim.angles import absolute_link_angles
from sim.plant import Plant


def main() -> None:
    parser = argparse.ArgumentParser(description="View the upright cartpole.")
    parser.add_argument("--links", type=int, default=2, help="number of links, 1 to 5")
    parser.add_argument("--angle", type=float, default=0.1, help="initial hinge_1 angle, rad")
    parser.add_argument("--force", type=float, default=0.0, help="constant cart force, N")
    args = parser.parse_args()

    plant = Plant(n_links=args.links)
    qpos = np.zeros(plant.n_links + 1)
    qpos[1] = args.angle
    plant.reset(qpos=qpos)
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
