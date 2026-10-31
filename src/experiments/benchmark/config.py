"""One comparison: shared limits and duration, several initial states."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from experiments.runner import Experiment


@dataclass(frozen=True)
class Benchmark:
    """LQR weights, PD poles, and the episodes both controllers run."""

    name: str
    n_links: int
    force_limit: float
    state_weight: np.ndarray
    input_weight: np.ndarray
    poles: np.ndarray
    cases: tuple[Experiment, ...]


def load_benchmark(path: Path) -> Benchmark:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_cases = data.get("cases")

    if not raw_cases:
        raise ValueError("benchmark needs at least one case")

    first = tuple(float(value) for value in raw_cases[0]["qpos"])
    inferred = len(first) - 1
    n_links = int(data.get("n_links", inferred))

    if n_links != inferred:
        raise ValueError(f"n_links={n_links} but qpos has {inferred} hinges")

    state_size = 2 * (n_links + 1)
    state_weight = np.diag(np.asarray(data["Q"], dtype=float))
    input_weight = np.atleast_2d(np.asarray(data["R"], dtype=float))
    poles = np.asarray(data["poles"], dtype=float).reshape(-1)
    duration = float(data["duration"])
    cases = tuple(_case(raw, n_links, duration) for raw in raw_cases)

    if state_weight.shape != (state_size, state_size):
        raise ValueError(f"Q length must be {state_size} for {n_links} links")

    if input_weight.shape != (1, 1):
        raise ValueError("R must be a single weight")

    if poles.shape != (state_size,):
        raise ValueError(f"need {state_size} poles for {n_links} links, got {poles.shape[0]}")

    return Benchmark(
        name=str(data["name"]),
        n_links=n_links,
        force_limit=float(data["force_limit"]),
        state_weight=state_weight,
        input_weight=input_weight,
        poles=poles,
        cases=cases,
    )


def _case(raw: dict, n_links: int, duration: float) -> Experiment:
    qpos = tuple(float(value) for value in raw["qpos"])

    if len(qpos) != n_links + 1:
        raise ValueError(f"{raw['name']} qpos length {len(qpos)} != {n_links + 1}")

    qvel_raw = raw.get("qvel")
    qvel = tuple(0.0 for _ in qpos) if qvel_raw is None else tuple(float(value) for value in qvel_raw)

    if len(qvel) != len(qpos):
        raise ValueError(f"{raw['name']} qvel length {len(qvel)} != {len(qpos)}")

    return Experiment(name=str(raw["name"]), qpos=qpos, qvel=qvel, duration=duration)
