"""The readings of mjlab's contact sensors on all three LocoMuJoCo backends.

mjlab's G1 velocity task adds MuJoCo contact sensors to the model: ``feet_ground_contact`` (per foot: the subtree of
the ankle roll link against the terrain body, net force) and ``self_collision`` (the robot's subtree against itself,
the first matched contact). MuJoCo (CPU) and MuJoCo Warp compute them natively, into ``sensordata``. MJX's JAX
implementation does not support subtree or body matching, so for MJX the sensors are removed from the model (see
``strip_contact_sensors``) and computed here from MJX's contact list, with MuJoCo's semantics: a contact counts if
it is active (``dist < includemargin``), ``found`` is the number of matching contacts, the net force is the sum of
the contact forces in the world frame, signed so that it acts on the primary (the feet), and the self-collision
force is that of the first matching contact, in its contact frame.
"""
from __future__ import annotations

import mujoco
import numpy as np

FEET_BODIES = ("robot/left_ankle_roll_link", "robot/right_ankle_roll_link")
TERRAIN_BODY = "terrain"
SELF_BODY = "robot/pelvis"


def sensor_slice(model: mujoco.MjModel, name: str) -> np.ndarray:
    i = model.sensor(name).id
    return np.arange(model.sensor_adr[i], model.sensor_adr[i] + model.sensor_dim[i])


def strip_contact_sensors(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Removes the contact sensors, which MJX's JAX implementation cannot compute."""
    for s in list(spec.sensors):
        if s.type == mujoco.mjtSensor.mjSENS_CONTACT:
            spec.delete(s)
    return spec


def _subtree_geoms(model: mujoco.MjModel, body: str) -> np.ndarray:
    """Geoms of ``body`` and of all its descendants."""
    root = model.body(body).id
    inside = np.zeros(model.nbody, dtype=bool)
    for b in range(model.nbody):
        p = b
        while p > 0 and p != root:
            p = model.body_parentid[p]
        inside[b] = p == root
    return inside[model.geom_bodyid]


class ContactReadings:
    """``found`` and ``force`` of the feet and self-collision sensors, from sensordata or from MJX's contact list."""

    def __init__(self, model: mujoco.MjModel, native: bool):
        self.native = native
        if native:
            self.feet_found = np.concatenate([sensor_slice(model, f"feet_ground_contact_{s}_ankle_roll_link_found")
                                              for s in ("left", "right")])
            self.feet_force = np.stack([sensor_slice(model, f"feet_ground_contact_{s}_ankle_roll_link_force")
                                        for s in ("left", "right")])
            self.self_found = sensor_slice(model, "self_collision_pelvis_found")
            self.self_force = sensor_slice(model, "self_collision_pelvis_force")
            return
        from mujoco import mjx
        # the geoms of MJX's fixed contact list are filled in by the collision pass
        d = mjx.forward(mjx.put_model(model), mjx.put_data(model, mujoco.MjData(model)))
        g1, g2 = np.asarray(d._impl.contact.geom1), np.asarray(d._impl.contact.geom2)
        self.dims = sorted(set(np.asarray(d._impl.contact.dim).tolist()))
        terrain = model.geom_bodyid == model.body(TERRAIN_BODY).id
        match, sign = [], []
        for foot in FEET_BODIES:
            sub = _subtree_geoms(model, foot)
            regular, reverse = sub[g1] & terrain[g2], sub[g2] & terrain[g1]
            match.append(regular | reverse)
            sign.append(np.where(reverse & ~regular, -1.0, 1.0))
        self.feet_match, self.feet_sign = np.stack(match), np.stack(sign)
        sub = _subtree_geoms(model, SELF_BODY)
        self.self_match = sub[g1] & sub[g2]

    def __call__(self, model, data, backend):
        """(feet found (2,), feet net force (2, 3), self-collision found (), self-collision force (3,))."""
        if self.native:
            sd = data.sensordata
            return sd[self.feet_found], sd[self.feet_force], sd[self.self_found][0], sd[self.self_force]
        import jax.numpy as jnp
        from mujoco.mjx._src import support
        c = data._impl.contact
        force = jnp.zeros((c.dist.shape[0], 6))
        for dim in self.dims:
            f, ids = support.contact_force_dim(model, data, dim)
            force = force.at[ids].set(f)
        active = (c.dist - c.includemargin) < 0
        world = jnp.einsum("cji,cj->ci", c.frame, force[:, :3])  # the rows of the frame are the contact axes
        feet = active[None] & self.feet_match
        own = active & self.self_match
        self_found = own.sum().astype(float)
        self_force = jnp.where(self_found > 0, force[jnp.argmax(own), :3], 0.0)
        return (feet.sum(axis=1).astype(float), jnp.einsum("fc,ci->fi", feet * self.feet_sign, world),
                self_found, self_force)
