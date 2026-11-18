"""G1Real environment on MJX / MjWarp.

``_modify_spec_for_mjx`` is a no-op: the MJCF already carries the ``*_collision`` geoms
with the right ``condim``/``priority``/``friction``, and its ``<contact><exclude>`` set
handles adjacent-link pairs.

Full collision is why the default backend is MjWarp rather than MJX: see the
``use_mjwarp`` note in ``loco_mujoco/core/mujoco_mjx.py``.
"""

from __future__ import annotations

from mujoco import MjSpec

from lmj_real.environments.g1_real import G1Real

MODEL_OPTIONS = {
    "iterations": 10,
    "ls_iterations": 20,
    "ccd_iterations": 50,
}

# These size the whole batched buffer, so they are set for the full self-collision geom
# set rather than per environment.
DEFAULT_NACONMAX = 50_000
DEFAULT_NJMAX = 2_048


class MjxG1Real(G1Real):

    mjx_enabled = True

    def __init__(
        self,
        timestep: float = 0.005,
        n_substeps: int = 4,
        naconmax: int = DEFAULT_NACONMAX,
        njmax: int = DEFAULT_NJMAX,
        use_mjwarp: bool = True,
        **kwargs,
    ):
        """
        Args:
            timestep: Physics timestep.
            n_substeps: Control decimation. 4 gives a 50 Hz policy.
            naconmax: Contact buffer size. Larger than the feet-only envs need,
                because self-collision is enabled.
            njmax: Constraint buffer size.
            use_mjwarp: Default True. MjWarp handles the full contact set far better
                than MJX. Note loco_mujoco flags MjWarp as experimental and its reset
                reuses the previous ``data`` with only ``qpos``/``qvel`` zeroed, so
                solver/contact state leaks across episodes. RSI itself is unaffected
                because the initial-state handler overwrites ``qpos``/``qvel``.
        """
        model_option_conf = dict(kwargs.pop("model_option_conf", None) or MODEL_OPTIONS)
        super().__init__(
            timestep=timestep,
            n_substeps=n_substeps,
            model_option_conf=model_option_conf,
            naconmax=naconmax,
            njmax=njmax,
            use_mjwarp=use_mjwarp,
            **kwargs,
        )

    def _modify_spec_for_mjx(self, spec: MjSpec) -> MjSpec:
        """No-op: the collision setup is already in the MJCF."""
        return spec
