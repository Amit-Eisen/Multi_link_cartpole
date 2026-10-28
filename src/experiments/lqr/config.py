"""LQR balance episode: weights plus the initial state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from experiments.runner import Experiment


def load_lqr(path: Path) -> tuple[Experiment, np.ndarray, np.ndarray]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    qpos = tuple(float(value) for value in data["qpos"])
    qvel_raw = data.get("qvel")
    qvel = tuple(0.0 for _ in qpos) if qvel_raw is None else tuple(float(value) for value in qvel_raw)
    experiment = Experiment(
        name=str(data["name"]),
        qpos=qpos,
        qvel=qvel,
        duration=float(data["duration"]),
    )
    state_weight = np.diag(np.asarray(data["Q"], dtype=float))
    input_weight = np.atleast_2d(np.asarray(data["R"], dtype=float))
    return experiment, state_weight, input_weight
