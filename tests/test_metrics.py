"""Settle hold, effort, saturation, and peak deviation on a made-up log."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.metrics import evaluate
from experiments.runner import Trajectory


def _log(time: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, command: np.ndarray, force: np.ndarray) -> Trajectory:
    return Trajectory(
        time=np.asarray(time, dtype=float),
        qpos=np.asarray(qpos, dtype=float),
        qvel=np.asarray(qvel, dtype=float),
        command=np.asarray(command, dtype=float),
        force=np.asarray(force, dtype=float),
        force_limit=50.0,
    )


def _quiet(samples: int, end: float) -> Trajectory:
    time = np.linspace(0.0, end, samples)
    zeros = np.zeros((samples, 3))
    steps = np.zeros(samples - 1)

    return _log(time, zeros, zeros, steps, steps)


class MetricsTest(unittest.TestCase):
    def test_hold_from_the_start(self) -> None:
        metrics = evaluate(_quiet(5, 2.0))

        self.assertAlmostEqual(metrics.settling_time, 0.0)

    def test_short_hold_does_not_count(self) -> None:
        metrics = evaluate(_quiet(5, 1.0))

        self.assertIsNone(metrics.settling_time)

    def test_an_exit_restarts_the_hold(self) -> None:
        time = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        qpos = np.zeros((9, 3))
        qpos[3, 0] = 0.1
        qvel = np.zeros((9, 3))
        steps = np.zeros(8)
        metrics = evaluate(_log(time, qpos, qvel, steps, steps), hold=1.5)

        self.assertAlmostEqual(metrics.settling_time, 2.0)

    def test_effort_uses_applied_force_and_saturation_uses_command(self) -> None:
        time = np.array([0.0, 1.0, 2.0])
        qpos = np.array([[0.0, 0.1, -0.2], [0.2, 0.0, 0.4], [-0.05, 0.0, 0.0]])
        qvel = np.zeros((3, 3))
        metrics = evaluate(_log(time, qpos, qvel, command=np.array([100.0, 0.0]), force=np.array([3.0, -4.0])))

        self.assertAlmostEqual(metrics.control_effort, 25.0)
        self.assertAlmostEqual(metrics.peak_applied_force, 4.0)
        self.assertAlmostEqual(metrics.saturation, 0.5)
        self.assertAlmostEqual(metrics.max_cart, 0.2)
        np.testing.assert_allclose(metrics.max_link, [0.1, 0.4])


if __name__ == "__main__":
    unittest.main()
