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
    """``command[k]`` is the controller output. ``force[k]`` is what the plant applied."""

    time: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    command: np.ndarray
    force: np.ndarray
    force_limit: float


def run(plant: Plant, policy: Policy, experiment: Experiment) -> Trajectory:
    steps = int(round(experiment.duration / plant.timestep))

    if steps <= 0:
        raise ValueError("duration must cover at least one physics step")

    state = plant.reset(qpos=np.asarray(experiment.qpos), qvel=np.asarray(experiment.qvel))
    time = np.empty(steps + 1)
    qpos = np.empty((steps + 1, plant.model.nq))
    qvel = np.empty((steps + 1, plant.model.nv))
    command = np.empty(steps)
    force = np.empty(steps)
    time[0] = state.time
    qpos[0] = state.qpos
    qvel[0] = state.qvel

    for k in range(steps):
        state = plant.step(float(policy(state)))
        command[k] = plant.command
        force[k] = plant.applied
        time[k + 1] = state.time
        qpos[k + 1] = state.qpos
        qvel[k + 1] = state.qvel

    return Trajectory(
        time=time,
        qpos=qpos,
        qvel=qvel,
        command=command,
        force=force,
        force_limit=plant.force_limit,
    )


def save(trajectory: Trajectory, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        time=trajectory.time,
        qpos=trajectory.qpos,
        qvel=trajectory.qvel,
        command=trajectory.command,
        force=trajectory.force,
        force_limit=np.array(trajectory.force_limit),
        qpos0=trajectory.qpos[0],
    )
