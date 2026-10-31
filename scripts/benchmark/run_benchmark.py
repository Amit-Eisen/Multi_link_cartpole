"""LQR versus PD on the same initial states, force limit, and duration.

PD's gain is pole placement on the upright linear model. LQR's gain is the
Riccati solution for Q and R in the same config. Neither gain is stored.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from controllers.lqr import LQR, lqr_gain
from controllers.place import PD, pole_placement
from experiments.benchmark.config import Benchmark, load_benchmark
from experiments.benchmark.report import Record, format_report
from experiments.metrics import evaluate
from experiments.plot import plot_comparison
from experiments.runner import run, save
from sim.linearize import linearize
from sim.plant import Plant


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LQR and pole-placement PD.")
    parser.add_argument("--config", type=Path, default=Path("configs/benchmark/two_link.yaml"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    spec = load_benchmark(args.config)
    plant = Plant(n_links=spec.n_links, force_limit=spec.force_limit)
    model = linearize(plant)
    k_lqr = lqr_gain(model.A, model.B, spec.state_weight, spec.input_weight).K
    k_pd = pole_placement(model.A, model.B, spec.poles)
    achieved = np.linalg.eigvals(model.A - model.B @ k_pd)
    folder = args.out / args.config.parent.name / spec.name
    records: list[Record] = []

    for case in spec.cases:
        series = []

        for controller, gain in (("LQR", k_lqr), ("PD", k_pd)):
            policy = LQR(gain) if controller == "LQR" else PD(gain)
            trajectory = run(plant, policy, case)
            save(trajectory, folder / f"{case.name}_{controller.lower()}.npz")
            series.append((controller, trajectory))
            records.append(Record(case.name, controller, evaluate(trajectory)))

        plot_comparison(series, folder / f"{case.name}.png")

    text = format_report(_preamble(spec, k_lqr, k_pd, achieved), records)
    summary = folder / "summary.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(text, encoding="utf-8")
    print(text, end="")


def _preamble(spec: Benchmark, k_lqr: np.ndarray, k_pd: np.ndarray, achieved: np.ndarray) -> str:
    weights = " ".join(f"{value:.4g}" for value in np.diag(spec.state_weight))
    control_weight = " ".join(f"{value:.4g}" for value in np.ravel(spec.input_weight))

    return "\n".join(
        (
            f"{spec.n_links} links, duration {spec.cases[0].duration:.1f} s, force limit {spec.force_limit:.1f} N",
            f"Q diag = {weights}",
            f"R = {control_weight}",
            f"PD poles requested = {_poles(spec.poles)}",
            f"PD poles achieved  = {_poles(achieved)}",
            f"K_pd  = {_row(k_pd)}",
            f"K_lqr = {_row(k_lqr)}",
            "settling_s is the start of a 2 s stay inside the band. not settled means that hold never finished.",
        )
    )


def _poles(poles: np.ndarray) -> str:
    values = np.asarray(poles, dtype=complex).reshape(-1)
    order = np.lexsort((values.imag, values.real))
    parts = []

    for value in values[order]:
        if abs(value.imag) < 1e-6:
            parts.append(f"{value.real:.4g}")
        else:
            parts.append(f"{value.real:.4g}{value.imag:+.4g}j")

    return " ".join(parts)


def _row(gain: np.ndarray) -> str:
    return " ".join(f"{value:+.4f}" for value in np.ravel(gain))


if __name__ == "__main__":
    main()
