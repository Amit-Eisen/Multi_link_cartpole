# multi-link cartpole

Upright multi-link cartpole in MuJoCo. One plant and one metric suite, first for classical control, later for RL.

Upright is joint angle 0. MuJoCo owns the physics. A controller maps state to cart force and nothing else.

Plan, metrics, and locked decisions: [docs/PROGRESS.md](docs/PROGRESS.md). Update that file when the work moves. This page is only how to run the repo.

## Setup

Python 3.12.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
```

Tests:

```
python -m unittest discover -s tests
```

## Scripts

Viewer. `--angle` is radians. The printout is degrees.

```
python scripts/view_plant.py --angle 0.1 --force 0
```

One open-loop run. Writes a log and a plot under `results/`.

```
python scripts/open_loop/run_open_loop.py --config configs/open_loop/fall.yaml
```

## Layout

```
src/sim/            plant and MJCF
src/controllers/    state -> force
src/experiments/    shared episode loop
configs/            numbers for a run
scripts/            entry points
results/            logs and plots
docs/PROGRESS.md    plan and decisions
```
