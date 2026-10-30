"""Full-state PID force, with the cart integral at zero and nonzero gain."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from controllers.pid import PID
from experiments.metrics import converged
from experiments.pid.config import load_pid
from experiments.runner import run
from sim.plant import Plant, State

ROOT = Path(__file__).resolve().parents[1]


class PIDTest(unittest.TestCase):
    def test_zero_integral_is_full_state_pd(self) -> None:
        pid = PID([[2.0, 0.0, 0.0, 3.0, 0.0, 0.0]])
        first = State(time=0.0, qpos=np.array([0.1, 0.0, 0.0]), qvel=np.zeros(3))
        later = State(time=0.2, qpos=np.array([0.1, 0.2, 0.0]), qvel=np.array([0.0, 0.0, 0.0]))

        self.assertAlmostEqual(pid(first), -0.2)
        self.assertAlmostEqual(pid(later), -0.2)
        self.assertEqual(pid.ki, 0.0)

    def test_cart_integral_adds_to_the_force(self) -> None:
        pid = PID(np.zeros((1, 6)), ki=4.0)
        first = State(time=0.0, qpos=np.array([0.1, 0.0, 0.0]), qvel=np.zeros(3))
        later = State(time=0.1, qpos=np.array([0.1, 0.0, 0.0]), qvel=np.zeros(3))

        self.assertAlmostEqual(pid(first), 0.0)
        self.assertAlmostEqual(pid(later), -0.04)

    def test_rewound_time_clears_the_integral(self) -> None:
        pid = PID(np.zeros((1, 6)), ki=4.0)
        pid(State(time=0.0, qpos=np.array([0.1, 0.0, 0.0]), qvel=np.zeros(3)))
        pid(State(time=0.1, qpos=np.array([0.1, 0.0, 0.0]), qvel=np.zeros(3)))
        restarted = State(time=0.0, qpos=np.array([0.5, 0.0, 0.0]), qvel=np.zeros(3))

        self.assertAlmostEqual(pid(restarted), 0.0)

    def test_two_link_balance_settles(self) -> None:
        experiment, gain, ki, n_links = load_pid(ROOT / "configs" / "pid" / "balance.yaml")
        plant = Plant(n_links=n_links)
        trajectory = run(plant, PID(gain, ki=ki), experiment)
        self.assertEqual(n_links, 2)
        self.assertEqual(ki, 0.0)
        self.assertTrue(converged(trajectory))
        self.assertLessEqual(float(np.max(np.abs(trajectory.force))), plant.force_limit)


if __name__ == "__main__":
    unittest.main()
