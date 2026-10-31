"""The 2-link comparison protocol, and the summary table."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from controllers.lqr import lqr_gain
from controllers.place import pole_placement
from experiments.benchmark.config import load_benchmark
from experiments.benchmark.report import Record, format_table
from experiments.metrics import Metrics
from sim.linearize import linearize
from sim.plant import Plant

ROOT = Path(__file__).resolve().parents[1]


def _sorted(poles: np.ndarray) -> np.ndarray:
    values = np.asarray(poles, dtype=complex).reshape(-1)
    order = np.lexsort((values.imag, values.real))

    return values[order]


class BenchmarkTest(unittest.TestCase):
    def test_cases_share_duration_and_both_gains_match_their_design(self) -> None:
        spec = load_benchmark(ROOT / "configs" / "benchmark" / "two_link.yaml")
        plant = Plant(n_links=spec.n_links, force_limit=spec.force_limit)
        model = linearize(plant)
        k_lqr = lqr_gain(model.A, model.B, spec.state_weight, spec.input_weight).K
        k_pd = pole_placement(model.A, model.B, spec.poles)
        closed = model.A - model.B @ k_pd

        self.assertGreaterEqual(len(spec.cases), 3)
        self.assertEqual({case.duration for case in spec.cases}, {15.0})
        self.assertEqual({len(case.qpos) for case in spec.cases}, {spec.n_links + 1})
        self.assertEqual(k_lqr.shape, (1, 6))
        self.assertEqual(k_pd.shape, (1, 6))
        np.testing.assert_allclose(_sorted(np.linalg.eigvals(closed)), _sorted(spec.poles), atol=1e-6)

    def test_missed_hold_is_written_as_not_settled(self) -> None:
        metrics = Metrics(
            settling_time=None,
            peak_applied_force=50.0,
            control_effort=10.0,
            saturation=0.5,
            max_cart=0.2,
            max_link=np.array([0.1, 0.3]),
        )
        text = format_table([Record("hard", "PD", metrics)])

        self.assertIn("not settled", text)
        self.assertIn("50.00", text)


if __name__ == "__main__":
    unittest.main()
