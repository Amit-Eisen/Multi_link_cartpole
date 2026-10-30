"""Load an open-loop episode from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..runner import Experiment


def load_open_loop(path: Path) -> tuple[Experiment, float, int]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    qpos = tuple(float(value) for value in data["qpos"])
    inferred = len(qpos) - 1
    n_links = int(data.get("n_links", inferred))

    if n_links != inferred:
        raise ValueError(f"n_links={n_links} but qpos has {inferred} hinges")

    qvel_raw = data.get("qvel")

    if qvel_raw is None:
        qvel = tuple(0.0 for _ in qpos)
    else:
        qvel = tuple(float(value) for value in qvel_raw)

    experiment = Experiment(
        name=str(data["name"]),
        qpos=qpos,
        qvel=qvel,
        duration=float(data["duration"]),
    )

    return experiment, float(data["force"]), n_links
