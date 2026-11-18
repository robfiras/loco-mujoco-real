"""The MJCFs load and keep the conventions the rest of the stack relies on."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

MODELS = Path(__file__).resolve().parents[1] / "lmj_real" / "models"
ROBOT_XMLS = ["g1_real/g1_real.xml", "x2_ultra_real/x2_ultra_real.xml"]


@pytest.mark.parametrize("xml", ROBOT_XMLS + ["x2_ultra_real/x2_ultra_real_lab.xml"])
def test_model_loads(xml):
    model = mujoco.MjModel.from_xml_path(str(MODELS / xml))
    assert model.nu == 29 and model.nq == 7 + 29


@pytest.mark.parametrize("xml", ROBOT_XMLS)
def test_conventions(xml):
    model = mujoco.MjModel.from_xml_path(str(MODELS / xml))
    # The free joint must be named `root`: trajectories are matched to the model by joint name.
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    assert root == 0 and model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE
    # The default pose is the `init_state` keyframe, never joint `ref` (which shifts kinematics).
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state") >= 0
    assert np.all(model.qpos0[7:] == 0.0)
    # One actuator per joint, and every target is clipped to a real range or not at all.
    assert set(model.actuator_trnid[:, 0]) == set(range(1, model.njnt))
    limited = model.actuator_ctrllimited.astype(bool)
    assert np.all(model.actuator_ctrlrange[limited, 0] < model.actuator_ctrlrange[limited, 1])

