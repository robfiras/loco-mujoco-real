"""Loads the model of mjlab's G1 velocity task from the files export_models.py writes.

``scene.xml`` (mjlab's ``Scene.write``) rounds numbers to six significant digits, so the actuator gains, joint
armature and geom poses and sizes are set back to the values of the model mjlab runs (``runtime_model.mjb``), and
the simulation options mjlab applies at runtime are set on the spec. The model LocoMuJoCo compiles from the result
equals mjlab's in every array that affects the simulation (check_mjlab.py compares them field by field).
"""
from __future__ import annotations

import os

import mujoco
import numpy as np

# mjlab's options for the flat G1 task (MujocoCfg and the G1 flat config), with the benchmark's Euler integrator
OPTIONS = dict(timestep=0.005, iterations=10, ls_iterations=20, ccd_iterations=50)


def load_spec(export_dir: str) -> mujoco.MjSpec:
    runtime = mujoco.MjModel.from_binary_path(os.path.join(export_dir, "runtime_model.mjb"))
    spec = mujoco.MjSpec.from_file(os.path.join(export_dir, "scene.xml"))
    for a in spec.actuators:
        i = runtime.actuator(a.name).id
        a.gainprm[:3], a.biasprm[:3] = runtime.actuator_gainprm[i, :3], runtime.actuator_biasprm[i, :3]
        a.ctrlrange, a.forcerange = runtime.actuator_ctrlrange[i], runtime.actuator_forcerange[i]
    for j in spec.joints:
        j.armature = float(runtime.dof_armature[runtime.jnt_dofadr[runtime.joint(j.name).id]])
    for g in spec.geoms:
        if g.name:
            i = runtime.geom(g.name).id
            g.fromto = [np.nan] * 6
            g.pos, g.quat, g.size = runtime.geom_pos[i], runtime.geom_quat[i], runtime.geom_size[i]
    for c in spec.cameras:
        if c.name:
            i = runtime.camera(c.name).id
            c.pos, c.quat = runtime.cam_pos[i], runtime.cam_quat[i]
    assert runtime.opt.integrator == mujoco.mjtIntegrator.mjINT_EULER
    for k, v in OPTIONS.items():
        assert np.isclose(getattr(runtime.opt, k), v), (k, getattr(runtime.opt, k), v)
        setattr(spec.option, k, v)
    return spec
