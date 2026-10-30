# multi-link cartpole

Upright cartpole in MuJoCo. Link count is a parameter, 1 to 5. Controllers so far: constant force, LQR, full-state PID.

Upright is joint angle 0. MuJoCo owns the physics. A controller maps state to cart force. The motor limit is 50 N.

## Setup

Python 3.12.

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
```

```
python -m unittest discover -s tests
```

## Run

Command-line angles are radians. Plots and the viewer print absolute link angles, the sum of the hinges out to that link. Controllers still use the relative joint coordinates.

A missing `n_links` in a config means 2. `qpos` length is `n_links + 1`.

Viewer:

```
python scripts/view_plant.py --links 3 --angle 0.1
```

Open loop. Writes a log and a plot under `results/`.

```
python scripts/open_loop/run_open_loop.py --config configs/open_loop/fall.yaml
```

LQR. `--view` opens the window after the log is written. `balance.yaml` is 2 links, `balance_3.yaml` is 3.

```
python scripts/lqr/run_lqr.py --config configs/lqr/balance.yaml --view
python scripts/lqr/run_lqr.py --config configs/lqr/balance_3.yaml --view
```

PID on the same 2-link initial state. Gains are pole placement, not the LQR gain. The log stores the commanded force and the force the plant applied.

```
python scripts/pid/run_pid.py --config configs/pid/balance.yaml --view
```

## Layout

```
src/sim/            plant, generated MJCF, upright linearization
src/controllers/    state -> force
src/experiments/    episode loop, configs, settle check, plots
configs/            numbers for a run
scripts/            entry points
results/            logs and plots, not committed
```
