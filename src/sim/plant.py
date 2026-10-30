"""Load a cartpole MJCF, apply a cart force, and read the true state.

Angles stay in radians. ``qpos`` and ``qvel`` are the ideal joint readings.
There is no sensor model here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from sim.model import MAX_LINKS, MIN_LINKS, cartpole_xml

ACTUATOR_NAME = "cart_force"
DEFAULT_MODEL = Path(__file__).with_name("cartpole_2.xml")
FORCE_LIMIT = 50.0


@dataclass(frozen=True)
class State:
    """Copy of the generalized state. Arrays are not views into MjData."""

    time: float
    qpos: np.ndarray
    qvel: np.ndarray


class Plant:
    """One compiled MuJoCo model and its live data.

    Controllers use :meth:`reset` and :meth:`step`. ``model`` and ``data`` are
    exposed for the viewer and for later linearization.
    """

    def __init__(
        self,
        n_links: int | None = None,
        model_path: Path | None = None,
        force_limit: float = FORCE_LIMIT,
    ) -> None:
        if model_path is not None:
            self.model_path = Path(model_path)
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
            self.n_links = self._link_count()

            if n_links is not None and n_links != self.n_links:
                raise ValueError(f"n_links={n_links} but {self.model_path.name} has {self.n_links}")
        else:
            self.n_links = 2 if n_links is None else n_links

            if self.n_links < MIN_LINKS or self.n_links > MAX_LINKS:
                raise ValueError(f"n_links must be {MIN_LINKS}..{MAX_LINKS}, got {self.n_links}")

            self.model_path = Path(f"cartpole_{self.n_links}.xml")
            print(f"Loading model from {self.model_path}")
            self.model = mujoco.MjModel.from_xml_string(cartpole_xml(self.n_links))

        self.data = mujoco.MjData(self.model)
        actuator_id = self._object_id(mujoco.mjtObj.mjOBJ_ACTUATOR, ACTUATOR_NAME)
        self._ctrl_index = int(self.model.actuator_ctrladr[actuator_id])
        self.force_limit = float(force_limit)
        self.command = 0.0
        self.applied = 0.0

        if self.model.nu != 1:
            raise ValueError(f"{self.model_path.name} must have one actuator, got {self.model.nu}")

        if self.force_limit <= 0.0 or not np.isfinite(self.force_limit):
            raise ValueError("force_limit must be a positive finite number")

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def body_mass(self, name: str) -> float:
        body_id = self._object_id(mujoco.mjtObj.mjOBJ_BODY, name)

        return float(self.model.body_mass[body_id])

    def qpos_index(self, joint: str) -> int:
        joint_id = self._object_id(mujoco.mjtObj.mjOBJ_JOINT, joint)

        return int(self.model.jnt_qposadr[joint_id])

    def reset(
        self,
        qpos: np.ndarray | None = None,
        qvel: np.ndarray | None = None,
    ) -> State:
        mujoco.mj_resetData(self.model, self.data)

        if qpos is not None:
            qpos = np.asarray(qpos, dtype=float)

            if qpos.shape != (self.model.nq,):
                raise ValueError(f"qpos shape {qpos.shape} != ({self.model.nq},)")

            self.data.qpos[:] = qpos

        if qvel is not None:
            qvel = np.asarray(qvel, dtype=float)

            if qvel.shape != (self.model.nv,):
                raise ValueError(f"qvel shape {qvel.shape} != ({self.model.nv},)")

            self.data.qvel[:] = qvel

        mujoco.mj_forward(self.model, self.data)

        return self.state()

    def step(self, force: float) -> State:
        """Advance one physics step. ``force`` is the commanded cart force, in newtons.

        The applied force is clipped to ``±force_limit``. ``command`` keeps the
        unclipped value.
        """
        command = float(force)

        if not np.isfinite(command):
            raise ValueError("force must be finite")

        self.command = command
        self.applied = float(np.clip(command, -self.force_limit, self.force_limit))
        self.data.ctrl[self._ctrl_index] = self.applied
        mujoco.mj_step(self.model, self.data)

        return self.state()

    def state(self) -> State:
        return State(
            time=float(self.data.time),
            qpos=self.data.qpos.copy(),
            qvel=self.data.qvel.copy(),
        )

    def _link_count(self) -> int:
        count = 0

        for joint_id in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or ""

            if name.startswith("hinge_"):
                count += 1

        return count

    def _object_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)

        if obj_id < 0:
            raise KeyError(f"no {mujoco.mjtObj(obj_type).name} named {name!r} in {self.model_path.name}")

        return int(obj_id)
