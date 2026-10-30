"""Settle check. Angles are absolute link orientations, in radians."""

from __future__ import annotations

import numpy as np

from experiments.runner import Trajectory
from sim.angles import absolute_link_angles, absolute_link_rates

CART_POSITION = 0.02
CART_VELOCITY = 0.05
LINK_ANGLE = np.deg2rad(2.0)
LINK_RATE = np.deg2rad(5.0)


def converged(trajectory: Trajectory) -> bool:
    """Final cart pose, every absolute link angle, and every rate are in band."""
    position = trajectory.qpos[-1]
    velocity = trajectory.qvel[-1]
    angles = absolute_link_angles(position)
    rates = absolute_link_rates(velocity)

    return bool(
        abs(position[0]) <= CART_POSITION
        and abs(velocity[0]) <= CART_VELOCITY
        and np.all(np.abs(angles) <= LINK_ANGLE)
        and np.all(np.abs(velocity[1:]) <= LINK_RATE)
        and np.all(np.abs(rates) <= LINK_RATE)
    )
