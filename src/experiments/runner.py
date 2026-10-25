"""Run one episode on a plant and keep the trajectory.

A policy is ``state -> force``. The runner does not know which controller it is.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sim.plant import Plant, State

Policy = Callable[[State], float]


@dataclass(frozen=True)
class Experiment:
    """Initial state and duration. Numbers come from config, not from the runner."""

    name: str
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    duration: float


@dataclass(frozen=True)
class Trajectory:
    """``force[k]`` is the input that produced ``qpos[k + 1]``."""

    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    force: np.ndarray


def run(plant: Plant, policy: Policy, experiment: Experiment) -> Trajectory:
    steps = int(round(experiment.duration / plant.timestep))

    if steps <= 0:
        raise ValueError("duration must cover at least one physics step")

    state = plant.reset(qpos=np.asarray(experiment.qpos), qvel=np.asarray(experiment.qvel))
    time = np.empty(steps + 1)
    qpos = np.empty((steps + 1, plant.model.nq))
    qvel = np.empty((steps + 1, plant.model.nv))
    force = np.empty(steps)
    time[0] = state.time
    qpos[0] = state.qpos
    qvel[0] = state.qvel

    for k in range(steps):
        command = float(policy(state))
        force[k] = command
        state = plant.step(command)
        time[k + 1] = state.time
        qpos[k + 1] = state.qpos
        qvel[k + 1] = state.qvel

    return Trajectory(time=time, qpos=qpos, qvel=qvel, force=force)


def save(trajectory: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        time=trajectory.time,
        qpos=trajectory.qpos,
        qvel=trajectory.qvel,
        force=trajectory.force,
        qpos0=trajectory.qpos[0],
    )
