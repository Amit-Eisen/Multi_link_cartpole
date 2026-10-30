"""Full-state PID force, with the cart integral at zero and nonzero gain."""

from __future__ import annotations

import unittest

import numpy as np

from controllers.pid import PID
from sim.plant import State


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


if __name__ == "__main__":
    unittest.main()
