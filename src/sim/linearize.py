"""Continuous linearization of a plant about the upright equilibrium.

State order is ``[qpos, qvel]``. The derivative is ``[qvel, qacc]``.
``lqr_gain`` consumes this ``A`` and ``B`` directly.
"""

from __future__ import annotations
from dataclasses import dataclass
import mujoco
import numpy as np
from sim.plant import Plant


@dataclass(frozen=True)
class LinearModel:
    A: np.ndarray
    B: np.ndarray


def linearize(plant: Plant, eps: float = 1e-6) -> LinearModel:
    """Finite-difference ``A`` and ``B`` at ``q = 0``, ``v = 0``, ``u = 0``."""

    model, data = plant.model, plant.data
    plant.reset(qpos=np.zeros(model.nq), qvel=np.zeros(model.nv))
    data.ctrl[:] = 0.0
    n = model.nq + model.nv
    A = np.zeros((n, n))
    B = np.zeros((n, model.nu))
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()
    ctrl0 = data.ctrl.copy()

    def xdot() -> np.ndarray:
        mujoco.mj_forward(model, data)

        return np.concatenate((data.qvel, data.qacc))

    def column(assign) -> np.ndarray:
        assign(eps)
        plus = xdot()
        assign(-eps)
        minus = xdot()
        assign(0.0)

        return (plus - minus) / (2.0 * eps)

    for index in range(model.nq):
        A[:, index] = column(lambda step, i=index: data.qpos.__setitem__(i, qpos0[i] + step))

    for index in range(model.nv):
        A[:, model.nq + index] = column(lambda step, i=index: data.qvel.__setitem__(i, qvel0[i] + step))

    for index in range(model.nu):
        B[:, index] = column(lambda step, i=index: data.ctrl.__setitem__(i, ctrl0[i] + step))

    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    data.ctrl[:] = ctrl0
    mujoco.mj_forward(model, data)

    return LinearModel(A=A, B=B)


def is_controllable(A: np.ndarray, B: np.ndarray) -> bool:
    """Rank of ``[B, AB, ..., A^{n-1}B]`` equals the state size."""

    blocks = [np.asarray(B, dtype=float)]
    current = blocks[0]

    for _ in range(A.shape[0] - 1):
        current = A @ current
        blocks.append(current)

    return int(np.linalg.matrix_rank(np.hstack(blocks))) == A.shape[0]
