"""Default-pose lookup.

The standing pose is stored as a MuJoCo **keyframe** (named ``init_state``), and
``qpos0`` stays at the MJCF's own zero configuration. They are not the same thing:

* ``qpos0`` is derived from each joint's ``ref``, and joint ``ref`` *shifts the
  kinematics* -- it defines the ``qpos`` value at which the joint sits at the pose
  implied by the body frames. Encoding a default pose there silently changes forward
  kinematics, which breaks any comparison against a reference trajectory whose body
  positions were computed on the unmodified model.
* the keyframe is inert: it records a pose without touching kinematics.

So the "default joint position" that the position controller and the ``joint_pos_rel``
observation are relative to is the **keyframe**, and that is what :func:`default_qpos`
returns.
"""

from __future__ import annotations

import mujoco
import numpy as np

DEFAULT_KEYFRAME_NAME = "init_state"


def default_qpos(model, keyframe_name: str = DEFAULT_KEYFRAME_NAME) -> np.ndarray:
    """Default pose for the model: the named keyframe if present, else ``qpos0``.

    Args:
        model: A compiled ``MjModel``.
        keyframe_name: Keyframe to look for.

    Returns:
        ``qpos``-shaped array of the default pose.
    """
    if getattr(model, "nkey", 0) > 0:
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
        if key_id >= 0:
            return np.asarray(model.key_qpos[key_id]).copy()
    return np.asarray(model.qpos0).copy()
