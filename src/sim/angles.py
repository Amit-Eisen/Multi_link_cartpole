"""Relative hinge angles versus absolute link orientations.

``qpos[0]`` is the cart. ``qpos[i]`` for ``i >= 1`` is the hinge relative to
its parent. The absolute angle of link ``k`` from upright is the sum of the
hinges from the cart out to that link. LQR keeps using the relative coordinates.
"""

from __future__ import annotations

import numpy as np


def absolute_link_angles(qpos: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=float)

    if qpos.ndim == 1:
        return np.cumsum(qpos[1:])

    return np.cumsum(qpos[:, 1:], axis=1)


def absolute_link_rates(qvel: np.ndarray) -> np.ndarray:
    qvel = np.asarray(qvel, dtype=float)

    if qvel.ndim == 1:
        return np.cumsum(qvel[1:])

    return np.cumsum(qvel[:, 1:], axis=1)
