"""Build an upright cartpole MJCF for 1 to 5 links.

Joint angle 0 is upright. Hinge angles are relative to the parent link.
The 2-link numbers match ``cartpole_2.xml``.
"""

from __future__ import annotations

MIN_LINKS = 1
MAX_LINKS = 5

_LINK_COLORS = (
    "0.85 0.35 0.15 1",
    "0.2 0.4 0.75 1",
    "0.2 0.65 0.35 1",
    "0.65 0.25 0.55 1",
    "0.85 0.7 0.2 1",
)


def cartpole_xml(n_links: int) -> str:
    """Return MJCF text. One cart force, no contacts, rail ±1 m."""
    if n_links < MIN_LINKS or n_links > MAX_LINKS:
        raise ValueError(f"n_links must be {MIN_LINKS}..{MAX_LINKS}, got {n_links}")

    links = []

    for index in range(1, n_links + 1):
        pos = "0 0 0" if index == 1 else "0 0 0.5"
        links.append(
            f"""
      <body name="link_{index}" pos="{pos}">
        <joint name="hinge_{index}" type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0.25" mass="0.1" diaginertia="0.002083333 0.002083333 0.00001"/>
        <geom name="link_{index}" type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" rgba="{_LINK_COLORS[index - 1]}"/>"""
        )

    zeros = " ".join(["0"] * (n_links + 1))
    closing = "</body>\n" * (n_links + 1)

    return f"""<mujoco model="cartpole_{n_links}">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="Euler"/>
  <visual>
    <global azimuth="90" elevation="-15"/>
  </visual>
  <default>
    <joint damping="0"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <light pos="0 -1 2" dir="0 0.4 -1" directional="true"/>
    <geom name="floor" type="plane" size="2 0.6 0.02" pos="0 0 -0.1" rgba="0.92 0.92 0.92 1"/>
    <geom name="rail" type="capsule" fromto="-1.15 0 0 1.15 0 0" size="0.008" rgba="0.45 0.45 0.45 1"/>
    <body name="cart" pos="0 0 0">
      <joint name="slide" type="slide" axis="1 0 0" limited="true" range="-1 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.00296667 0.00416667 0.00546667"/>
      <geom name="cart" type="box" size="0.1 0.08 0.05" rgba="0.25 0.25 0.28 1"/>
      {''.join(links)}
      {closing}
  </worldbody>
  <actuator>
    <motor name="cart_force" joint="slide" gear="1"/>
  </actuator>
  <keyframe>
    <key name="upright" qpos="{zeros}" qvel="{zeros}"/>
  </keyframe>
</mujoco>
"""
