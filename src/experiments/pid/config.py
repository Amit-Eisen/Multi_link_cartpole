"""PID balance episode: gains plus the initial state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from experiments.runner import Experiment


def load_pid(path: Path) -> tuple[Experiment, np.ndarray, float, int]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    qpos = tuple(float(value) for value in data["qpos"])
    inferred = len(qpos) - 1
    n_links = int(data.get("n_links", inferred))

    if n_links != inferred:
        raise ValueError(f"n_links={n_links} but qpos has {inferred} hinges")

    qvel_raw = data.get("qvel")
    qvel = tuple(0.0 for _ in qpos) if qvel_raw is None else tuple(float(value) for value in qvel_raw)
    experiment = Experiment(
        name=str(data["name"]),
        qpos=qpos,
        qvel=qvel,
        duration=float(data["duration"]),
    )
    gain = np.atleast_2d(np.asarray(data["k"], dtype=float))

    if gain.shape != (1, 2 * (n_links + 1)):
        raise ValueError(f"k length must be {2 * (n_links + 1)} for {n_links} links")

    return experiment, gain, float(data.get("ki", 0.0)), n_links
