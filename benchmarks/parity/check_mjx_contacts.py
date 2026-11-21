"""Checks the MJX computation of mjlab's contact sensors (mjlab_contacts.py) against MuJoCo's own sensors.

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python check_mjx_contacts.py EXPORT_DIR/full

Steps mjlab's G1 in MuJoCo (CPU) with random position targets until it has fallen, and every 4th step compares the
feet and self-collision readings computed by MJX from its contact list (on a model without the contact sensors)
with the readings of MuJoCo's native contact sensors, on the same state.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from mjlab_contacts import ContactReadings, strip_contact_sensors
from mjlab_model import load_spec

mc = load_spec(sys.argv[1]).compile()
mx = strip_contact_sensors(load_spec(sys.argv[1])).compile()
native, computed = ContactReadings(mc, native=True), ContactReadings(mx, native=False)
sysx = mjx.put_model(mx)
read = jax.jit(lambda d: computed(sysx, mjx.forward(sysx, d), jnp))
dc = mujoco.MjData(mc)
mujoco.mj_resetDataKeyframe(mc, dc, 0)
rng = np.random.default_rng(0)
worst, n_self, n = np.zeros(4), 0, 0
for t in range(400):
    dc.ctrl[:] = mc.key_ctrl[0] + rng.uniform(-0.6, 0.6, mc.nu)
    mujoco.mj_step(mc, dc)
    if t % 4:
        continue
    mujoco.mj_forward(mc, dc)
    dx = mjx.put_data(mx, mujoco.MjData(mx)).replace(qpos=jnp.asarray(dc.qpos), qvel=jnp.asarray(dc.qvel),
                                                     ctrl=jnp.asarray(dc.ctrl), qacc_warmstart=jnp.asarray(dc.qacc_warmstart))
    ff, fF, sf, sF = (np.asarray(x) for x in read(dx))
    cf, cF, csf, csF = native(mc, dc, np)
    n += 1
    n_self += csf > 0
    worst = np.maximum(worst, [np.abs(ff - cf).max(), np.abs(fF - cF).max() / (1 + np.abs(cF).max()), abs(sf - csf),
                               abs(np.linalg.norm(sF) - np.linalg.norm(csF)) / (1 + np.linalg.norm(csF))])
print(f"{n} states ({n_self} with self-contact, final base height {dc.qpos[2]:.2f} m): max |feet found diff| {worst[0]:.0f}, "
      f"max rel. feet net force diff {worst[1]:.1e}, max |self found diff| {worst[2]:.0f}, "
      f"max rel. self-collision force diff {worst[3]:.1e}")
