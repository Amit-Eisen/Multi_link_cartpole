"""Save a trajectory figure. Angles are plotted in degrees."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from sim.angles import absolute_link_angles

from .runner import Trajectory


def plot(trajectory: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    degrees = np.degrees(absolute_link_angles(trajectory.qpos))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    axes[0].plot(trajectory.time, trajectory.qpos[:, 0])
    axes[0].set_ylabel("cart x (m)")

    for index in range(degrees.shape[1]):
        axes[1].plot(trajectory.time, degrees[:, index], label=f"link {index + 1}")

    axes[1].set_ylabel("absolute angle (deg)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)

    plt.close(fig)
