"""Upright linearization of the 2-link plant."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco
import numpy as np

from controllers.lqr import LQR, lqr_gain
from experiments.lqr.config import load_lqr
from experiments.runner import Experiment, run
from sim.linearize import is_controllable, linearize
from sim.plant import Plant

ROOT = Path(__file__).resolve().parents[1]
BALANCE = ROOT / "configs" / "lqr" / "balance.yaml"


class LinearizeTest(unittest.TestCase):
    def test_upright_is_equilibrium(self) -> None:
        plant = Plant()
        plant.reset(qpos=np.zeros(3), qvel=np.zeros(3))
        plant.data.ctrl[:] = 0.0
        mujoco.mj_forward(plant.model, plant.data)
        self.assertLess(float(np.max(np.abs(plant.data.qacc))), 1e-8)

    def test_structure_and_controllability(self) -> None:
        model = linearize(Plant())
        self.assertEqual(model.A.shape, (6, 6))
        self.assertEqual(model.B.shape, (6, 1))
        np.testing.assert_allclose(model.A[:3, :3], 0.0, atol=1e-6)
        np.testing.assert_allclose(model.A[:3, 3:], np.eye(3), atol=1e-6)
        self.assertTrue(is_controllable(model.A, model.B))
        self.assertTrue(np.any(np.real(np.linalg.eigvals(model.A)) > 0.1))

    def test_balance_does_not_fall(self) -> None:
        loaded, state_weight, input_weight = load_lqr(BALANCE)
        episode = Experiment(
            name=loaded.name,
            qpos=loaded.qpos,
            qvel=loaded.qvel,
            duration=2.0,
        )
        plant = Plant()
        model = linearize(plant)
        gain = lqr_gain(model.A, model.B, state_weight, input_weight)
        trajectory = run(plant, LQR(gain.K), episode)
        self.assertTrue(np.all(np.isfinite(trajectory.qpos)))
        self.assertLess(float(np.max(np.abs(trajectory.qpos[:, 0]))), 1.0)
        self.assertLess(float(np.max(np.abs(trajectory.qpos[:, 1:]))), np.deg2rad(20.0))
        self.assertLess(float(np.max(np.abs(trajectory.qpos[-1, 1:]))), np.deg2rad(5.0))


if __name__ == "__main__":
    unittest.main()
