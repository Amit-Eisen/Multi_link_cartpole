"""Pole placement: known gain, closed-loop poles, and the cartpole poles."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from controllers.place import PD, pole_placement
from experiments.pid.config import load_pid
from sim.linearize import linearize
from sim.plant import Plant, State

ROOT = Path(__file__).resolve().parents[1]


def _sorted(poles: np.ndarray) -> np.ndarray:
    values = np.asarray(poles, dtype=complex).reshape(-1)
    order = np.lexsort((values.imag, values.real))

    return values[order]


class PolePlacementTest(unittest.TestCase):
    def test_double_integrator_gain(self) -> None:
        # Poles -1 and -2. Then K = [2, 3], because A - B K = [[0, 1], [-2, -3]].
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        gain = pole_placement(A, B, [-1.0, -2.0])

        np.testing.assert_allclose(gain, [[2.0, 3.0]], atol=1e-8)
        closed = A - B @ gain
        np.testing.assert_allclose(_sorted(np.linalg.eigvals(closed)), _sorted([-1.0, -2.0]), atol=1e-8)

    def test_complex_pair_stays_real(self) -> None:
        A = np.array([[0.0, 1.0], [0.0, 0.0]])
        B = np.array([[0.0], [1.0]])
        poles = [-1.0 + 1.0j, -1.0 - 1.0j]
        gain = pole_placement(A, B, poles)
        closed = A - B @ gain

        self.assertFalse(np.iscomplexobj(gain))
        np.testing.assert_allclose(_sorted(np.linalg.eigvals(closed)), _sorted(poles), atol=1e-8)

    def test_pole_count_must_match_the_state(self) -> None:
        with self.assertRaises(ValueError):
            pole_placement([[0.0, 1.0], [0.0, 0.0]], [[0.0], [1.0]], [-1.0])

    def test_cartpole_poles_match_the_request(self) -> None:
        _experiment, poles, _ki, n_links = load_pid(ROOT / "configs" / "pid" / "balance.yaml")
        plant = Plant(n_links=n_links)
        model = linearize(plant)
        gain = pole_placement(model.A, model.B, poles)
        closed = model.A - model.B @ gain

        np.testing.assert_allclose(_sorted(np.linalg.eigvals(closed)), _sorted(poles), atol=1e-6)

    def test_policy_returns_one_force(self) -> None:
        state = State(time=0.0, qpos=np.array([0.1, 0.0, 0.0]), qvel=np.zeros(3))
        force = PD([[2.0, 0.0, 0.0, 3.0, 0.0, 0.0]])(state)

        self.assertAlmostEqual(force, -0.2)


if __name__ == "__main__":
    unittest.main()
