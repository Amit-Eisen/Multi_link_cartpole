"""Continuous infinite-horizon LQR gain.

The plant is not here. Pass the linear model and the weights. The control law
that uses this gain is ``u = -K x``, applied later by the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are


@dataclass(frozen=True)
class LQRGain:
    """``K`` is the feedback gain. ``P`` is the solution of the Riccati equation."""

    K: np.ndarray
    P: np.ndarray


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> LQRGain:
    """Solve the continuous LQR problem for ``x_dot = A x + B u``.

    The cost is the integral of ``x' Q x + u' R u``. The gain minimizes it:

        A' P + P A - P B R^{-1} B' P + Q = 0
        K = R^{-1} B' P
        u = -K x
    """
    state_matrix = np.atleast_2d(np.asarray(A, dtype=float))
    input_matrix = np.atleast_2d(np.asarray(B, dtype=float))
    state_weight = _symmetric(Q)
    input_weight = _symmetric(R)
    state_size = state_matrix.shape[0]
    input_size = input_matrix.shape[1]

    if state_matrix.shape != (state_size, state_size):
        raise ValueError("A must be square")

    if input_matrix.shape != (state_size, input_size):
        raise ValueError("B must have one row per state")

    if state_weight.shape != (state_size, state_size):
        raise ValueError("Q must match the state size")

    if input_weight.shape != (input_size, input_size):
        raise ValueError("R must match the input size")

    riccati = solve_continuous_are(state_matrix, input_matrix, state_weight, input_weight)
    gain = np.linalg.solve(input_weight, input_matrix.T @ riccati)

    return LQRGain(K=gain, P=riccati)

def _symmetric(matrix: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(np.asarray(matrix, dtype=float))

    return 0.5 * (values + values.T)
