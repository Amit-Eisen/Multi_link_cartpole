"""Open-loop checks for the upright 2-link plant. No controller."""

from __future__ import annotations

import unittest

import numpy as np

from sim.plant import Plant


def _roll(plant: Plant, force: float, duration: float):
    steps = int(round(duration / plant.timestep))
    state = plant.state()
    for _ in range(steps):
        state = plant.step(force)
    return state


class PlantTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plant = Plant()

    def test_model_contract(self) -> None:
        plant = self.plant
        self.assertEqual(plant.model.nq, 3)
        self.assertEqual(plant.model.nv, 3)
        self.assertEqual(plant.model.nu, 1)
        self.assertAlmostEqual(plant.timestep, 0.001)
        self.assertEqual(plant.qpos_index("slide"), 0)
        self.assertEqual(plant.qpos_index("hinge_1"), 1)
        self.assertEqual(plant.qpos_index("hinge_2"), 2)
        self.assertAlmostEqual(plant.body_mass("cart"), 1.0)
        self.assertAlmostEqual(plant.body_mass("link_1"), 0.1)
        self.assertAlmostEqual(plant.body_mass("link_2"), 0.1)
        np.testing.assert_allclose(plant.model.opt.gravity, [0.0, 0.0, -9.81])

    def test_upright_holds_with_zero_force(self) -> None:
        self.plant.reset()
        state = _roll(self.plant, force=0.0, duration=0.1)
        np.testing.assert_allclose(state.qpos, 0.0, atol=1e-8)
        np.testing.assert_allclose(state.qvel, 0.0, atol=1e-8)

    def test_small_angle_falls(self) -> None:
        angle = 0.05
        self.plant.reset(qpos=[0.0, angle, 0.0])
        state = _roll(self.plant, force=0.0, duration=0.3)
        self.assertGreater(abs(state.qpos[1]), abs(angle) + 0.02)

    def test_force_moves_cart(self) -> None:
        self.plant.reset()
        state = _roll(self.plant, force=2.0, duration=0.3)
        self.assertGreater(state.qpos[0], 0.01)
        self.assertLess(state.qpos[0], 1.0)

    def test_free_response_stays_finite(self) -> None:
        self.plant.reset(qpos=[0.0, 0.2, 0.0])
        state = _roll(self.plant, force=0.0, duration=2.0)
        self.assertTrue(np.all(np.isfinite(state.qpos)))
        self.assertTrue(np.all(np.isfinite(state.qvel)))
        self.assertLessEqual(abs(state.qpos[0]), 1.0 + 1e-6)
        self.assertLess(float(np.max(np.abs(state.qvel))), 50.0)


if __name__ == "__main__":
    unittest.main()
