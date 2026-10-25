"""The shared episode loop, checked without a viewer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from controllers.constant import ConstantForce
from experiments.open_loop.config import load_open_loop
from experiments.plot import plot
from experiments.runner import Experiment, run, save
from sim.plant import Plant

ROOT = Path(__file__).resolve().parents[1]


class RunnerTest(unittest.TestCase):
    def test_zero_force_holds_and_log_shape_matches(self) -> None:
        experiment = Experiment(name="upright", qpos=(0.0, 0.0, 0.0), qvel=(0.0, 0.0, 0.0), duration=0.1)
        trajectory = run(Plant(), ConstantForce(0.0), experiment)
        steps = int(round(experiment.duration / Plant().timestep))
        self.assertEqual(len(trajectory.time), steps + 1)
        self.assertEqual(len(trajectory.force), steps)
        self.assertEqual(trajectory.qpos.shape, (steps + 1, 3))
        np.testing.assert_allclose(trajectory.qpos, 0.0, atol=1e-8)
        np.testing.assert_allclose(trajectory.qvel, 0.0, atol=1e-8)
        np.testing.assert_allclose(trajectory.force, 0.0)

    def test_force_step_moves_the_cart(self) -> None:
        experiment = Experiment(name="push", qpos=(0.0, 0.0, 0.0), qvel=(0.0, 0.0, 0.0), duration=0.3)
        trajectory = run(Plant(), ConstantForce(2.0), experiment)
        self.assertGreater(trajectory.qpos[-1, 0], 0.01)
        self.assertLess(trajectory.qpos[-1, 0], 1.0)
        np.testing.assert_allclose(trajectory.force, 2.0)

    def test_same_experiment_matches(self) -> None:
        experiment = Experiment(name="fall", qpos=(0.0, 0.05, 0.0), qvel=(0.0, 0.0, 0.0), duration=0.2)
        policy = ConstantForce(0.0)
        first = run(Plant(), policy, experiment)
        second = run(Plant(), policy, experiment)
        np.testing.assert_allclose(first.qpos, second.qpos)
        np.testing.assert_allclose(first.qvel, second.qvel)
        np.testing.assert_allclose(first.force, second.force)

    def test_configs_load(self) -> None:
        fall, fall_force = load_open_loop(ROOT / "configs" / "open_loop" / "fall.yaml")
        push, push_force = load_open_loop(ROOT / "configs" / "open_loop" / "force_step.yaml")
        self.assertEqual(fall.qpos, (0.0, 0.05, 0.0))
        self.assertEqual(fall_force, 0.0)
        self.assertEqual(push.qpos, (0.0, 0.0, 0.0))
        self.assertEqual(push_force, 2.0)

    def test_save_and_plot(self) -> None:
        experiment = Experiment(name="push", qpos=(0.0, 0.0, 0.0), qvel=(0.0, 0.0, 0.0), duration=0.05)
        trajectory = run(Plant(), ConstantForce(2.0), experiment)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "push.npz"
            figure_path = root / "push.png"
            save(trajectory, log_path)
            plot(trajectory, figure_path)
            with np.load(log_path) as loaded:
                np.testing.assert_allclose(loaded["force"], trajectory.force)
                np.testing.assert_allclose(loaded["qpos0"], trajectory.qpos[0])
            self.assertGreater(figure_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
