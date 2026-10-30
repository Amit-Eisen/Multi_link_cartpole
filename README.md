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

One open-loop run. Writes a log and a plot under `results/`. A missing `n_links` means 2. `qpos` length must be `n_links + 1`.

```
python scripts/open_loop/run_open_loop.py --config configs/open_loop/fall.yaml
```

LQR balance. The 2-link case is the default. `configs/lqr/balance_3.yaml` is the 3-link case. Plots show absolute link angles, the sum of the hinges out to that link. The controller still uses the relative joint coordinates.

```
python scripts/lqr/run_lqr.py --config configs/lqr/balance.yaml --view
python scripts/lqr/run_lqr.py --config configs/lqr/balance_3.yaml --view
```

Viewer for any link count from 1 to 5:

```
python scripts/view_plant.py --links 3 --angle 0.1
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
