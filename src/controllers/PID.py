"""Full-state PD plus an integral on cart position.

``k`` weights ``[qpos, qvel]``. ``ki`` weights the running integral of cart
position. ``ki`` is 0 until a constant force bias exists. This file does not
use the LQR gain.
"""

from __future__ import annotations

import numpy as np

from sim.plant import State


class PID:
    """One force: ``u = -(k [qpos, qvel] + ki * integral of cart position)``."""

    def __init__(self, k: np.ndarray, ki: float = 0.0) -> None:
        self.k = np.atleast_2d(np.asarray(k, dtype=float))
        self.ki = float(ki)
        self._integral = 0.0
        self._time: float | None = None

        if self.k.shape[0] != 1:
            raise ValueError("k must have one row")

    def __call__(self, state: State) -> float:
        x = np.concatenate((state.qpos, state.qvel))

        if x.shape != (self.k.shape[1],):
            raise ValueError(f"state length {x.shape[0]} does not match k columns {self.k.shape[1]}")

        self._advance(state)

        return float(-(self.k @ x + self.ki * self._integral).item())

    def _advance(self, state: State) -> None:
        if self._time is None or state.time < self._time:
            self._integral = 0.0
        else:
            self._integral += float(state.qpos[0]) * (state.time - self._time)

        self._time = float(state.time)
