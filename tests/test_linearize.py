"""Upright linearization and LQR convergence for several link counts."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from controllers.lqr import LQR, lqr_gain
from experiments.lqr.config import load_lqr
from experiments.metrics import converged
from experiments.runner import Experiment, run
from sim.linearize import is_controllable, linearize
from sim.plant import Plant

ROOT = Path(__file__).resolve().parents[1]
LINK_COUNTS = (1, 2, 3, 5)


def _weights(n_links: int) -> tuple[np.ndarray, np.ndarray]:
    diagonal = [5.0] + [50.0] * n_links + [1.0] * (n_links + 1)

    return np.diag(diagonal), np.atleast_2d([0.1])


def _episode(n_links: int, duration: float) -> Experiment:
    qpos = np.zeros(n_links + 1)
    qvel = np.zeros(n_links + 1)
    qpos[1] = 0.05

    return Experiment("balance", tuple(qpos), tuple(qvel), duration)


class LinearizeTest(unittest.TestCase):
    def test_upright_is_equilibrium(self) -> None:
        for n_links in LINK_COUNTS:
            plant = Plant(n_links=n_links)
            plant.reset(qpos=np.zeros(plant.model.nq), qvel=np.zeros(plant.model.nv))
            plant.data.ctrl[:] = 0.0
            mujoco.mj_forward(plant.model, plant.data)
            self.assertLess(float(np.max(np.abs(plant.data.qacc))), 1e-8)

    def test_structure_and_controllability(self) -> None:
        for n_links in LINK_COUNTS:
            plant = Plant(n_links=n_links)
            size = plant.model.nq
            model = linearize(plant)
            self.assertEqual(model.A.shape, (2 * size, 2 * size))
            self.assertEqual(model.B.shape, (2 * size, 1))
            np.testing.assert_allclose(model.A[:size, :size], 0.0, atol=1e-6)
            np.testing.assert_allclose(model.A[:size, size:], np.eye(size), atol=1e-6)
            self.assertTrue(is_controllable(model.A, model.B))
            self.assertTrue(np.any(np.real(np.linalg.eigvals(model.A)) > 0.1))

    def test_lqr_converges(self) -> None:
        for n_links in LINK_COUNTS:
            with self.subTest(n_links=n_links):
                plant = Plant(n_links=n_links)
                model = linearize(plant)
                state_weight, input_weight = _weights(n_links)
                gain = lqr_gain(model.A, model.B, state_weight, input_weight)
                trajectory = run(plant, LQR(gain.K), _episode(n_links, duration=5.0))
                self.assertTrue(np.all(np.isfinite(trajectory.qpos)))
                self.assertTrue(converged(trajectory), f"{n_links} links did not settle")

    def test_configured_two_and_three_link_balance(self) -> None:
        for name in ("balance.yaml", "balance_3.yaml"):
            loaded, state_weight, input_weight, n_links = load_lqr(ROOT / "configs" / "lqr" / name)
            plant = Plant(n_links=n_links)
            model = linearize(plant)
            gain = lqr_gain(model.A, model.B, state_weight, input_weight)
            episode = Experiment(loaded.name, loaded.qpos, loaded.qvel, 4.0)
            trajectory = run(plant, LQR(gain.K), episode)
            self.assertTrue(converged(trajectory), name)


if __name__ == "__main__":
    unittest.main()
