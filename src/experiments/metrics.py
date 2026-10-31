"""Episode metrics shared by every controller.

Settle band: cart within 2 cm and 0.05 m/s, every absolute link angle within
2 deg, every joint rate and every absolute link rate within 5 deg/s. Settling
time is the start of a continuous hold (default 2 s) inside that band. Leaving
the band resets the window. The time is ``None`` when the hold never finishes.

Control effort is the integral of applied force squared. The applied force is
constant on each physics step. Saturation is the fraction of steps whose
command lies outside the force limit. Deviation is the peak absolute cart
position and the peak absolute angle of each link, over the whole episode.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.runner import Trajectory
from sim.angles import absolute_link_angles, absolute_link_rates

CART_POSITION = 0.02
CART_VELOCITY = 0.05
LINK_ANGLE = np.deg2rad(2.0)
LINK_RATE = np.deg2rad(5.0)
HOLD = 2.0


@dataclass(frozen=True)
class Metrics:
    """``settling_time`` is ``None`` when the hold never completes."""

    settling_time: float | None
    peak_applied_force: float
    control_effort: float
    saturation: float
    max_cart: float
    max_link: np.ndarray


def inside_band(trajectory: Trajectory) -> np.ndarray:
    """One boolean per state sample."""
    position = trajectory.qpos
    velocity = trajectory.qvel
    angles = absolute_link_angles(position)
    rates = absolute_link_rates(velocity)
    cart = (np.abs(position[:, 0]) <= CART_POSITION) & (np.abs(velocity[:, 0]) <= CART_VELOCITY)
    links = np.all(np.abs(angles) <= LINK_ANGLE, axis=1)
    joint_rates = np.all(np.abs(velocity[:, 1:]) <= LINK_RATE, axis=1)
    link_rates = np.all(np.abs(rates) <= LINK_RATE, axis=1)

    return cart & links & joint_rates & link_rates


def settling_time(trajectory: Trajectory, hold: float = HOLD) -> float | None:
    """Start of the first window that stays inside the band for ``hold`` seconds."""
    if hold <= 0.0:
        raise ValueError("hold must be positive")

    mask = inside_band(trajectory)
    start: float | None = None

    for index, inside in enumerate(mask):
        if not inside:
            start = None
            continue

        if start is None:
            start = float(trajectory.time[index])

        if float(trajectory.time[index]) - start + 1e-9 >= hold:
            return start

    return None


def evaluate(trajectory: Trajectory, hold: float = HOLD) -> Metrics:
    """Score one episode. The scorer does not know which controller ran."""
    _check_shape(trajectory)
    dt = np.diff(trajectory.time)
    saturated = np.abs(trajectory.command) > trajectory.force_limit
    angles = absolute_link_angles(trajectory.qpos)

    return Metrics(
        settling_time=settling_time(trajectory, hold=hold),
        peak_applied_force=float(np.max(np.abs(trajectory.force))),
        control_effort=float(np.sum(np.square(trajectory.force) * dt)),
        saturation=float(np.mean(saturated)),
        max_cart=float(np.max(np.abs(trajectory.qpos[:, 0]))),
        max_link=np.max(np.abs(angles), axis=0),
    )


def converged(trajectory: Trajectory) -> bool:
    """The final sample is inside the settle band. This does not check the hold."""
    return bool(inside_band(trajectory)[-1])


def _check_shape(trajectory: Trajectory) -> None:
    samples = trajectory.time.shape[0]

    if trajectory.force.shape != (samples - 1,) or trajectory.command.shape != (samples - 1,):
        raise ValueError("command and force must have one sample fewer than time")

    if trajectory.qpos.shape[0] != samples or trajectory.qvel.shape[0] != samples:
        raise ValueError("qpos and qvel must have one row per time sample")
