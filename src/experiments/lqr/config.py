"""LQR balance episode: weights plus the initial state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from experiments.runner import Experiment


def load_lqr(path: Path) -> tuple[Experiment, np.ndarray, np.ndarray, int]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    qpos = tuple(float(value) for value in data["qpos"])
    n_links = _n_links(data, qpos)
    qvel_raw = data.get("qvel")
    qvel = tuple(0.0 for _ in qpos) if qvel_raw is None else tuple(float(value) for value in qvel_raw)
    experiment = Experiment(
        name=str(data["name"]),
        qpos=qpos,
        qvel=qvel,
        duration=float(data["duration"]),
    )
    state_weight = np.diag(np.asarray(data["Q"], dtype=float))

    if state_weight.shape != (2 * (n_links + 1), 2 * (n_links + 1)):
        raise ValueError(f"Q length must be {2 * (n_links + 1)} for {n_links} links")

    input_weight = np.atleast_2d(np.asarray(data["R"], dtype=float))

    return experiment, state_weight, input_weight, n_links


def _n_links(data: dict, qpos: tuple[float, ...]) -> int:
    inferred = len(qpos) - 1
    n_links = int(data.get("n_links", inferred))

    if n_links != inferred:
        raise ValueError(f"n_links={n_links} but qpos has {inferred} hinges")

    return n_links
