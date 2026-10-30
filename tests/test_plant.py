"""Open-loop checks for the upright 2-link plant. No controller."""

from __future__ import annotations

import unittest

import numpy as np

from sim.angles import absolute_link_angles
from sim.plant import DEFAULT_MODEL, Plant, State


def _roll(plant: Plant, force: float, duration: float) -> State:
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
        self.assertEqual(plant.n_links, 2)
        self.assertEqual(plant.model.nq, plant.n_links + 1)
        self.assertEqual(plant.model.nv, plant.n_links + 1)
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

    def test_link_counts(self) -> None:
        for n_links in (1, 2, 3, 5):
            plant = Plant(n_links=n_links)
            self.assertEqual(plant.n_links, n_links)
            self.assertEqual(plant.model.nq, n_links + 1)
            self.assertEqual(plant.model.nv, n_links + 1)
            self.assertEqual(plant.model.nu, 1)
            self.assertAlmostEqual(plant.body_mass("cart"), 1.0)

            for index in range(1, n_links + 1):
                self.assertEqual(plant.qpos_index(f"hinge_{index}"), index)
                self.assertAlmostEqual(plant.body_mass(f"link_{index}"), 0.1)

    def test_absolute_angle_is_the_hinge_sum(self) -> None:
        np.testing.assert_allclose(absolute_link_angles(np.array([0.2, 0.3, -0.1, 0.4])), [0.3, 0.2, 0.6])

    def test_generated_two_link_matches_stored_model(self) -> None:
        generated = Plant(n_links=2)
        stored = Plant(model_path=DEFAULT_MODEL)
        qpos = np.array([0.0, 0.05, 0.0])
        generated.reset(qpos=qpos)
        stored.reset(qpos=qpos)

        for _ in range(200):
            generated_state = generated.step(0.0)
            stored_state = stored.step(0.0)

        np.testing.assert_allclose(generated_state.qpos, stored_state.qpos, atol=1e-5)
        np.testing.assert_allclose(generated_state.qvel, stored_state.qvel, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
