"""Open-loop force. Later controllers use the same call shape."""

from __future__ import annotations

from sim.plant import State


class ConstantForce:
    """Return one cart force in newtons, ignoring the state."""

    def __init__(self, force: float) -> None:
        self.force = float(force)

    def __call__(self, state: State) -> float:
        del state

        return self.force
