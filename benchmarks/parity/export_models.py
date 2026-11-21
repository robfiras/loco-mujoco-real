"""Exports the model of mjlab's G1 velocity task (Mjlab-Velocity-Flat-Unitree-G1), as mjlab compiles it.

Run in an environment with mjlab (tested with mjlab 1.6.1, MuJoCo 3.11) and a GPU:

    python export_models.py OUT_DIR

Builds the task with the benchmark's settings (domain randomization events removed, Euler integrator) once with
mjlab's full self-collision and once with its feet-only collision set, and writes for each, to OUT_DIR/full and
OUT_DIR/reduced: mjlab's scene (scene.xml via Scene.write, with its meshes copied to assets/), which carries the merged
keyframe and the native contact sensors mjlab adds, and the model mjlab runs (runtime_model.mjb, after
mjlab applies its simulation options), against which the model LocoMuJoCo compiles from scene.xml is checked.
"""
import re
import shutil
import sys
from pathlib import Path

import mujoco
import mjlab.tasks  # noqa: F401  registers the tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-Unitree-G1"
DR_EVENTS = ("push_robot", "foot_friction", "encoder_bias", "base_com")


def build(contacts: str, corruption: bool = True) -> ManagerBasedRlEnv:
    cfg = load_env_cfg(TASK)
    cfg.observations["actor"].enable_corruption = corruption
    cfg.scene.num_envs = 1
    cfg.sim.mujoco.integrator = "euler"
    for name in DR_EVENTS:
        cfg.events.pop(name, None)
    if contacts == "reduced":
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FEET_ONLY_COLLISION
        cfg.scene.entities["robot"].collisions = (FEET_ONLY_COLLISION,)
    return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for contacts in ("full", "reduced"):
        env = build(contacts)
        d = out / contacts
        env.scene.write(d)
        # mjlab's G1 reads its meshes from files, which Scene.write does not copy: copy the referenced ones
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
        (d / "assets").mkdir(exist_ok=True)
        for f in sorted(set(re.findall(r'file="([^"]+)"', (d / "scene.xml").read_text()))):
            shutil.copy(G1_XML.parent / "assets" / f, d / "assets" / f)
        mujoco.mj_saveModel(env.sim.mj_model, str(d / "runtime_model.mjb"), None)
        m = env.sim.mj_model
        print(f"{contacts}: {d}  nbody {m.nbody} ngeom {m.ngeom} nsensor {m.nsensor} nkey {m.nkey} "
              f"timestep {m.opt.timestep} iterations {m.opt.iterations}/{m.opt.ls_iterations}/{m.opt.ccd_iterations} "
              f"integrator {mujoco.mjtIntegrator(m.opt.integrator).name}")
        env.close()
