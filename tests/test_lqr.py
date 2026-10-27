"""Continuous LQR gain against two systems with a known answer."""

from __future__ import annotations
import unittest
import numpy as np
from controllers.LQR import lqr_gain


class LQRGainTest(unittest.TestCase):
    def test_scalar_unstable_pole(self) -> None:
        # x_dot = x + u, Q = R = 1. Then P = 1 + sqrt(2) and K = P.
        gain = lqr_gain(A=[[1.0]], B=[[1.0]], Q=[[1.0]], R=[[1.0]])
        expected = 1.0 + np.sqrt(2.0)
        np.testing.assert_allclose(gain.K, [[expected]])
        np.testing.assert_allclose(gain.P, [[expected]])

    def test_double_integrator(self) -> None:
        # q = 1, R = 1. K = [1, sqrt(3)].
        gain = lqr_gain(
            A=[[0.0, 1.0], [0.0, 0.0]],
            B=[[0.0], [1.0]],
            Q=np.eye(2),
            R=[[1.0]],
        )

        np.testing.assert_allclose(gain.K, [[1.0, np.sqrt(3.0)]], atol=1e-10)
        self.assertEqual(gain.K.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
