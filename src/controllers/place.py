"""Pole placement for ``x_dot = A x + B u``.

``pole_placement`` returns the ``K`` for which the eigenvalues of ``A - B K``
are the requested poles. The policy is ``u = -K x``. Pass the poles in. This
module does not keep a gain.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import place_poles

from sim.plant import State


def pole_placement(A: np.ndarray, B: np.ndarray, poles: np.ndarray) -> np.ndarray:
    """Return ``K`` with shape ``(inputs, states)`` so that ``u = -K x``."""
    state_matrix = np.atleast_2d(np.asarray(A, dtype=float))
    input_matrix = np.atleast_2d(np.asarray(B, dtype=float))
    desired = np.asarray(poles, dtype=complex).reshape(-1)
    state_size = state_matrix.shape[0]

    if state_matrix.shape != (state_size, state_size):
        raise ValueError("A must be square")

    if input_matrix.shape[0] != state_size:
        raise ValueError("B must have one row per state")

    if desired.shape != (state_size,):
        raise ValueError(f"need {state_size} poles, got {desired.shape[0]}")

    if not np.all(np.isfinite(desired)):
        raise ValueError("poles must be finite")

    result = place_poles(state_matrix, input_matrix, desired)
    gain = np.real_if_close(np.asarray(result.gain_matrix), tol=1e-6)

    if np.iscomplexobj(gain):
        raise ValueError("pole placement returned a complex gain")

    return np.asarray(gain, dtype=float)


class PD:
    """One force from full-state feedback: ``u = -K [qpos, qvel]``."""

    def __init__(self, K: np.ndarray) -> None:
        self.K = np.atleast_2d(np.asarray(K, dtype=float))

        if self.K.shape[0] != 1:
            raise ValueError("K must have one row")

    def __call__(self, state: State) -> float:
        x = np.concatenate((state.qpos, state.qvel))

        if x.shape != (self.K.shape[1],):
            raise ValueError(f"state length {x.shape[0]} does not match K columns {self.K.shape[1]}")

        return float((-self.K @ x).item())
