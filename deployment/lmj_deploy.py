"""Runtime side of the lmj-real sim2sim / deploy bundle.

This module holds everything a *deploy* process needs and nothing else. It imports
**numpy and mujoco only** -- no jax, no flax, no ``loco_mujoco``, no ``lmj_real``. That
constraint is the whole point: if the observation can only be rebuilt by calling back
into the training stack, the pipeline has not actually been validated for a robot.

The three pieces:

* :class:`PolicySpec` / :func:`load_actor` -- the exported actor plus its frozen
  observation normalizer, run in numpy: :class:`MlpActor` for PPO,
  :class:`EncoderDecoderActor` for SuperMimic.
* :class:`ObservationBuilder` -- rebuilds the 283-dim actor observation from raw
  ``MjData`` (``qpos`` / ``qvel`` / ``xpos`` / ``xmat``) and one reference frame. This
  is a hand-written reimplementation of what the ``lmj_real`` observation pipeline
  computes; ``validate_bundle.py`` asserts the two agree term by term.
* :class:`MotionBundle` -- the reference clips, sliced to the tracked bodies.

Bundles are produced by ``export_bundle.py`` and consumed by
``rollout_mujoco.py`` and ``deploy_g1.py``.

Two ordering conventions coexist and mixing them is the single easiest way to get a
plausible-looking policy that falls over:

============================  ==========================================================
``spec.joint_names``          MuJoCo joint order (= ``robot.joints_to_consider``, legs
                              first). Used by ``joint_pos_rel``, ``joint_vel`` and by
                              the reference ``command`` term.
``spec.actuator_joint_names`` the actuator order (shoulders first). Used by the
                              **action vector**, ``last_action``, ``action_scale``,
                              ``kp`` / ``kd`` and ``ctrl``.
============================  ==========================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# Matches ``RunningMeanStd`` in loco_mujoco/algorithms/common/networks.py.
OBS_NORM_EPS = 1e-8

_ACTIVATIONS = {
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "elu": lambda x: np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0))),
}


# ---------------------------------------------------------------------------
# small math helpers (duplicated from lmj_real.core.utils.tracking on purpose --
# see the module docstring: no lmj_real import at deploy time)


def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo-convention (w, x, y, z) quaternions to rotation matrices.

    Args:
        q_wxyz: Shape ``(..., 4)``.

    Returns:
        Shape ``(..., 3, 3)``.
    """
    q = np.asarray(q_wxyz, dtype=np.float64)
    flat = q.reshape(-1, 4)
    w, x, y, z = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    m = np.empty((flat.shape[0], 3, 3), dtype=np.float64)
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - z * w)
    m[:, 0, 2] = 2 * (x * z + y * w)
    m[:, 1, 0] = 2 * (x * y + z * w)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - x * w)
    m[:, 2, 0] = 2 * (x * z - y * w)
    m[:, 2, 1] = 2 * (y * z + x * w)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_6d_cols01(mats: np.ndarray) -> np.ndarray:
    """First two **columns** of a rotation matrix, flattened row-major.

    The tracking *observations* use this encoding (``mat[..., :2].reshape(...)``).
    Do not confuse it with the columns-0-and-2 tan/norm encoding the privileged
    all-body observations use.
    """
    mats = np.asarray(mats)
    return mats[..., :, :2].reshape(mats.shape[:-2] + (6,))


def yaw_rot(mat: np.ndarray) -> np.ndarray:
    """Yaw-only part of a rotation matrix, ``Rz(atan2(R[1,0], R[0,0]))``."""
    ang = np.arctan2(mat[1, 0], mat[0, 0])
    c, s = np.cos(ang), np.sin(ang)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def anchor_delta(robot_anchor_pos, robot_anchor_mat, ref_anchor_pos, ref_anchor_mat):
    """The re-anchoring transform: robot xy + heading, reference z + roll/pitch."""
    delta_pos = np.concatenate([robot_anchor_pos[:2], ref_anchor_pos[2:3]])
    delta_rot = yaw_rot(robot_anchor_mat @ ref_anchor_mat.T)
    return delta_pos, delta_rot


def reanchor(ref_pos, ref_mat, ref_anchor_pos, delta_pos, delta_rot):
    """Apply :func:`anchor_delta` to a set of reference body poses."""
    pos_rel = delta_pos + (ref_pos - ref_anchor_pos) @ delta_rot.T
    mat_rel = np.einsum("ij,bjk->bik", delta_rot, ref_mat)
    return pos_rel, mat_rel


# ---------------------------------------------------------------------------
# policy spec + actor


@dataclass
class PolicySpec:
    """Everything about the trained policy that the deploy side must agree on."""

    # identity / provenance
    robot: str
    env_name: str
    source_checkpoint: str

    # timing
    sim_timestep: float
    n_substeps: int

    # compiled mj_option values from the training model, so a deploy model can be put
    # on the same solver and the remaining difference is the plant alone
    model_options: dict[str, float | int]

    # joints, in MuJoCo joint order (observations, reference command)
    joint_names: list[str]
    default_joint_pos: list[float]

    # joints, in actuator order (actions, ctrl, gains)
    actuator_joint_names: list[str]
    action_scale: list[float]
    action_offset: list[float]
    ctrl_range: list[list[float | None]]
    kp: list[float]
    kd: list[float]
    effort_limit: list[float]

    # tracking
    tracked_body_names: list[str]
    anchor_body_index: int
    end_effector_body_names: list[str]
    root_joint_name: str

    # network
    hidden_layers: list[int]
    activation: str
    obs_dim: int
    action_dim: int
    obs_terms: list[dict[str, Any]]

    # terminations (reported, not enforced, by the sim2sim rig)
    termination: dict[str, float | None] = field(default_factory=dict)

    #: Number of frames the actor's history group is stacked over. 1 = no history.
    #: A policy trained with history consumes obs_dim = base + (n-1)*len(history dims),
    #: so getting this wrong is a dimension mismatch, not a silent degradation.
    history_length: int = 1
    #: Names of the obs_terms that are stacked, in order. Empty when history_length is 1.
    history_term_names: list[str] = field(default_factory=list)

    #: Whether the goal observation carries the reference anchor pose. False drops
    #: motion_anchor_pos_b (3) + motion_anchor_ori_b (6), taking tracking_goal from 193
    #: to 184 dims and the policy observation from 286 to 277. Defaults True so bundles
    #: exported before this field existed keep their meaning.
    use_anchor: bool = True
    #: Whether the goal observation carries motion_anchor_pos_b (3). False takes
    #: tracking_goal from 193 to 190 dims and the policy observation from 286 to 283.
    #: Defaults True for the same reason as use_anchor.
    use_anchor_pos: bool = True

    #: "mlp" (:class:`MlpActor`) or "encoder_decoder" (:class:`EncoderDecoderActor`).
    actor_type: str = "mlp"
    #: encoder_decoder only: the encoder reads the first ``encoder_dim`` observation
    #: entries, the decoder the last ``actor_history_length`` frames of the rest.
    encoder_dim: int = 0
    actor_history_length: int = 1

    # the training environment's `model_contact_conf` (``self_collisions``, ``condim``),
    # which changes the robot's geoms after the MJCF is loaded; empty if it set none
    model_contact: dict[str, Any] = field(default_factory=dict)

    @property
    def control_dt(self) -> float:
        return self.sim_timestep * self.n_substeps

    @property
    def needs_world_position(self) -> bool:
        """Whether the actor observes the reference anchor's position, which a robot
        does not measure."""
        return (self.use_anchor and self.use_anchor_pos
                and any(t["type"] == "GoalTrajMimicAnchor" for t in self.obs_terms))

    @property
    def n_future(self) -> int:
        """How many reference frames ahead of the current one the observation reads."""
        return max((int(t.get("n_future", 0)) for t in self.obs_terms), default=0)

    @classmethod
    def load(cls, bundle_dir: str | Path) -> PolicySpec:
        with (Path(bundle_dir) / "policy.json").open() as f:
            d = json.load(f)
        known = {f_.name for f_ in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, bundle_dir: str | Path) -> None:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["control_dt"] = self.control_dt
        with (Path(bundle_dir) / "policy.json").open("w") as f:
            json.dump(d, f, indent=2)


class MlpActor:
    """The exported actor: frozen obs normalizer + a plain numpy MLP.

    Training normalizes with :class:`loco_mujoco...networks.RunningMeanStd`, which
    folds the incoming batch into its statistics *before* normalizing. At deploy the
    statistics are frozen, which is both the right thing to do and -- with a count in
    the billions -- numerically indistinguishable from what training did.
    """

    def __init__(self, spec: PolicySpec, weights: dict[str, np.ndarray],
                 std_floor: float = 0.0):
        """
        Args:
            spec: The bundle's :class:`PolicySpec`.
            weights: Arrays from ``policy.npz``.
            std_floor: Clamp the per-dimension normalizer std from below. 0.0 (the
                default) reproduces training exactly. 1e-3 is a sane value when
                running against a *different* plant: observation dims that were
                constant during training have a near-zero std, so any kinematic
                difference on such a dim reaches the network as a z-score of ~1e4.
        """
        self.spec = spec
        self.mean = np.asarray(weights["obs_mean"], dtype=np.float64)
        self.std = np.sqrt(np.asarray(weights["obs_var"], dtype=np.float64) + OBS_NORM_EPS)
        self.n_low_std = int((self.std < 1e-3).sum())
        if std_floor > 0.0:
            self.std = np.maximum(self.std, std_floor)
        self.log_std = np.asarray(weights["log_std"], dtype=np.float64)
        self.layers = _dense_layers(weights, "", np.float64)
        self.act_fn = _activation(spec.activation)

    @classmethod
    def load(cls, bundle_dir: str | Path, spec: PolicySpec | None = None,
             std_floor: float = 0.0) -> MlpActor:
        bundle_dir = Path(bundle_dir)
        spec = spec or PolicySpec.load(bundle_dir)
        with np.load(bundle_dir / "policy.npz") as d:
            weights = {k: d[k] for k in d.files}
        return cls(spec, weights, std_floor=std_floor)

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float64) - self.mean) / self.std

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic (mean) action for one observation vector."""
        x = self.normalize(obs)
        for w, b in self.layers[:-1]:
            x = self.act_fn(x @ w + b)
        w, b = self.layers[-1]
        return (x @ w + b).astype(np.float32)

    def z_scores(self, obs: np.ndarray) -> np.ndarray:
        """Normalized observation -- useful for spotting out-of-distribution inputs."""
        return self.normalize(obs)

    def reset(self) -> None:
        """Nothing to clear: this actor is stateless."""


class EncoderDecoderActor:
    """SuperMimic's actor, ``a = dec([tanh(enc(goal)), history])``, with its own history.

    The observation splits in two. The first ``spec.encoder_dim`` entries, the reference,
    go through the encoder. The rest is one frame of the robot's state, and the decoder
    sees the last ``spec.actor_history_length`` of those frames, oldest first and ending
    with the current one. After :meth:`reset` the first call fills the whole history with
    its frame, as training does at the start of an episode.

    Normalization is per entry with frozen statistics, so normalizing a frame once when it
    enters the history equals training's normalizing the stacked history at every step.

    The matmuls run in float32. Training ran them in bfloat16 if its ``net_dtype`` said
    so, a difference the exporter measures and reports.
    """

    def __init__(self, spec: PolicySpec, weights: dict[str, np.ndarray],
                 std_floor: float = 0.0):
        self.spec = spec
        self.mean = np.asarray(weights["obs_mean"], dtype=np.float32)
        std = np.sqrt(np.asarray(weights["obs_var"], dtype=np.float64)
                      + float(weights["obs_norm_eps"]))
        self.n_low_std = int((std < 1e-3).sum())
        if std_floor > 0.0:
            std = np.maximum(std, std_floor)
        self.std = std.astype(np.float32)
        self.encoder = _dense_layers(weights, "enc_", np.float32)
        self.decoder = _dense_layers(weights, "dec_", np.float32)
        self.act_fn = _activation(spec.activation)
        self.n_goal = int(spec.encoder_dim)
        self.n_frames = int(spec.actor_history_length)
        self._history: np.ndarray | None = None

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (np.asarray(obs, dtype=np.float32) - self.mean) / self.std

    def _mlp(self, layers, x: np.ndarray) -> np.ndarray:
        for w, b in layers[:-1]:
            x = self.act_fn(x @ w + b)
        w, b = layers[-1]
        return x @ w + b

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Action for the current observation, which also enters the history."""
        x = self.normalize(obs)
        frame = x[self.n_goal:]
        if self._history is None:
            self._history = np.tile(frame, (self.n_frames, 1))
        else:
            self._history = np.roll(self._history, -1, axis=0)
            self._history[-1] = frame
        z = np.tanh(self._mlp(self.encoder, x[:self.n_goal]))
        return self._mlp(self.decoder, np.concatenate([z, self._history.ravel()]))

    def z_scores(self, obs: np.ndarray) -> np.ndarray:
        return self.normalize(obs)

    def reset(self) -> None:
        """Forget the history. Call at the start of every episode."""
        self._history = None


def load_actor(bundle_dir: str | Path, spec: PolicySpec | None = None,
               std_floor: float = 0.0) -> MlpActor | EncoderDecoderActor:
    """The bundle's actor, whichever kind ``spec.actor_type`` says it is."""
    bundle_dir = Path(bundle_dir)
    spec = spec or PolicySpec.load(bundle_dir)
    cls = {"mlp": MlpActor, "encoder_decoder": EncoderDecoderActor}.get(spec.actor_type)
    if cls is None:
        raise ValueError(f"unknown actor_type {spec.actor_type!r} in policy.json")
    with np.load(bundle_dir / "policy.npz") as d:
        weights = {k: d[k] for k in d.files}
    return cls(spec, weights, std_floor=std_floor)


def _dense_layers(weights: dict[str, np.ndarray], prefix: str, dtype) -> list:
    layers, i = [], 0
    while f"{prefix}w{i}" in weights:
        layers.append((np.asarray(weights[f"{prefix}w{i}"], dtype=dtype),
                       np.asarray(weights[f"{prefix}b{i}"], dtype=dtype)))
        i += 1
    return layers


def _activation(name: str):
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"activation {name!r} not implemented in the numpy deploy "
            f"actor; add it to lmj_deploy._ACTIVATIONS"
        ) from None


# ---------------------------------------------------------------------------
# observation


class ObservationBuilder:
    """Rebuilds the actor observation from raw MuJoCo state plus a reference frame.

    Layout (283 for the shipped G1 config), in the order the trained policy saw it::

        tracking_goal     190    see :meth:`tracking_goal`
        last_action        29    previous raw policy output, actuator order
        joint_pos_rel      29    qpos[joints] - default_joint_pos
        joint_vel          29    qvel[joints]
        projected_gravity   3    R_root^T @ (0, 0, -1)
        base_ang_vel        3    qvel[3:6]  (root free joint, already body-frame)

    Everything is read from ``data.qpos`` / ``data.qvel`` / ``data.xpos`` /
    ``data.xmat`` of whatever model is handed in, resolved by *name*, so the same
    builder works on the training MJCF and on a different deploy MJCF.
    """

    def __init__(self, spec: PolicySpec, model: mujoco.MjModel):
        self.spec = spec
        self.model = model

        def jid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if i < 0:
                raise ValueError(f"joint {name!r} not in model")
            return i

        jids = [jid(n) for n in spec.joint_names]
        for n, i in zip(spec.joint_names, jids):
            if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"joint {n!r} is not a hinge; the 1-dof mapping breaks")
        self.qpos_adr = np.array([model.jnt_qposadr[i] for i in jids], dtype=np.int32)
        self.dof_adr = np.array([model.jnt_dofadr[i] for i in jids], dtype=np.int32)
        self.default_joint_pos = np.asarray(spec.default_joint_pos, dtype=np.float64)

        root = jid(spec.root_joint_name)
        if model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"joint {spec.root_joint_name!r} is not a free joint")
        self.root_qpos_adr = int(model.jnt_qposadr[root])
        self.root_dof_adr = int(model.jnt_dofadr[root])

        def bid(name: str) -> int:
            i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                raise ValueError(f"body {name!r} not in model")
            return i

        self.body_ids = np.array([bid(n) for n in spec.tracked_body_names], dtype=np.int32)
        self.anchor_row = int(spec.anchor_body_index)
        self.use_anchor = bool(getattr(spec, "use_anchor", True))
        self.use_anchor_pos = self.use_anchor and bool(getattr(spec, "use_anchor_pos", True))
        # Order of the per-component gyro terms, so _t_base_ang_vel_entry knows which
        # of qvel[3:6] each one is. Taken from the spec rather than assumed, because the
        # terms are declared individually (base_ang_vel_r/p/y) and could be reordered.
        self._ang_vel_order = [t["name"] for t in spec.obs_terms
                               if t["type"] == "EntryFromFreeJointVel"]
        self._hist_buf = None
        self._ref_heading = np.eye(3)
        self.anchor_body_id = int(self.body_ids[self.anchor_row])
        self.ee_rows = np.array(
            [spec.tracked_body_names.index(n) for n in spec.end_effector_body_names],
            dtype=np.int32,
        )

        # Slot offsets, for the per-term diffing in validate_bundle.py.
        self.term_slices: dict[str, slice] = {}
        for t in spec.obs_terms:
            self.term_slices[t["name"]] = slice(t["start"], t["start"] + t["dim"])

    # -- pieces ------------------------------------------------------------

    def robot_anchor_frame(self, data: mujoco.MjData):
        pos = np.asarray(data.xpos[self.anchor_body_id], dtype=np.float64)
        mat = np.asarray(data.xmat[self.anchor_body_id], dtype=np.float64).reshape(3, 3)
        return pos, mat

    def align_heading(self, data: mujoco.MjData, ref: dict[str, np.ndarray]) -> float:
        """Rotate the reference onto the robot's current heading. Returns the angle in rad.

        Training resets the robot onto the reference orientation, so relative yaw starts
        near zero. A real robot starts wherever it stands, on an arbitrary IMU yaw, so
        call this once when the policy starts. Only the terms that compare the robot's
        orientation with the reference's are affected: the anchor orientation and
        ``RootOriRelTraj``.
        """
        _, robot_mat = self.robot_anchor_frame(data)
        ref_mat = quat_to_mat(ref["body_quat_w"][self.anchor_row])
        self._ref_heading = yaw_rot(robot_mat @ ref_mat.T)
        return float(np.arctan2(self._ref_heading[1, 0], self._ref_heading[0, 0]))

    def state(self, data: mujoco.MjData) -> np.ndarray:
        """``joint_pos_rel``, ``joint_vel``, ``projected_gravity``, ``base_ang_vel``."""
        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        joint_pos_rel = qpos[self.qpos_adr] - self.default_joint_pos
        joint_vel = qvel[self.dof_adr]
        # The free joint's quaternion is the root body's world orientation, and MuJoCo
        # expresses free-joint angular velocity in the local body frame -- which is
        # exactly what the IMU gyro at the identity-quaternion
        # imu_in_pelvis site reads.
        root_quat = qpos[self.root_qpos_adr + 3: self.root_qpos_adr + 7]
        proj_grav = quat_to_mat(root_quat).T @ np.array([0.0, 0.0, -1.0])
        base_ang_vel = qvel[self.root_dof_adr + 3: self.root_dof_adr + 6]
        return np.concatenate([joint_pos_rel, joint_vel, proj_grav, base_ang_vel])

    def tracking_goal(self, data: mujoco.MjData, ref: dict[str, np.ndarray]) -> np.ndarray:
        """The ``tracking_goal`` group: 190 numbers, 193 with the anchor position, 184 without
        the anchor.

        ``command`` (reference joint pos/vel) + the reference anchor pose expressed in
        the *robot's* anchor frame + the robot's own tracked bodies in that same frame.
        Note the frame is the robot anchor's **full** orientation, not the heading-only
        frame the privileged all-body observations use.
        """
        robot_anchor_pos, robot_anchor_mat = self.robot_anchor_frame(data)
        robot_anchor_inv = robot_anchor_mat.T

        ref_anchor_pos = np.asarray(ref["body_pos_w"][self.anchor_row], dtype=np.float64)
        ref_anchor_mat = self._ref_heading @ quat_to_mat(ref["body_quat_w"][self.anchor_row])

        anchor_pos_b = robot_anchor_inv @ (ref_anchor_pos - robot_anchor_pos)
        anchor_ori_b = mat_to_6d_cols01(robot_anchor_inv @ ref_anchor_mat)

        body_pos_w = np.asarray(data.xpos[self.body_ids], dtype=np.float64)
        body_mat_w = np.asarray(data.xmat[self.body_ids], dtype=np.float64).reshape(-1, 3, 3)
        body_pos_b = (body_pos_w - robot_anchor_pos) @ robot_anchor_inv.T
        body_ori_b = mat_to_6d_cols01(np.einsum("ij,bjk->bik", robot_anchor_inv, body_mat_w))

        blocks = [np.asarray(ref["joint_pos"], dtype=np.float64),
                  np.asarray(ref["joint_vel"], dtype=np.float64)]
        # The only place the reference's spatial pose enters this observation. A policy
        # trained with use_anchor=False never saw these nine numbers, and one trained with
        # use_anchor_pos=False never saw the first three.
        if self.use_anchor_pos:
            blocks.append(anchor_pos_b)
        if self.use_anchor:
            blocks.append(anchor_ori_b)
        blocks += [body_pos_b.ravel(), body_ori_b.ravel()]
        return np.concatenate(blocks)

    #: obs_terms["type"] -> the method that produces it. Anything in the spec that is
    #: not here is a term this builder cannot reproduce, and build() says so rather than
    #: quietly emitting a wrong-length vector.
    _TERM_BUILDERS = {
        "G1RealJointPosRel": "_t_joint_pos_rel",
        "JointVelArray": "_t_joint_vel",
        "ProjectedGravityVector": "_t_projected_gravity",
        "EntryFromFreeJointVel": "_t_base_ang_vel_entry",
        "LastAction": "_t_last_action",
        "GoalTrajMimicAnchor": "_t_tracking_goal",
        "TrajFutureFrames": "_t_future_frames",
        "RootOriRelTraj": "_t_root_ori_rel",
    }

    def _term(self, term, data, ref, last_action):
        """One observation term, by spec entry."""
        fn = self._TERM_BUILDERS.get(term["type"])
        if fn is None:
            raise RuntimeError(
                f"observation term {term['name']!r} of type {term['type']!r} has no "
                "deploy-side implementation; the bundle cannot rebuild this policy's "
                "observation."
            )
        return np.atleast_1d(getattr(self, fn)(data, ref, last_action, term))

    def build(self, data: mujoco.MjData, ref: dict[str, np.ndarray],
              last_action: np.ndarray) -> np.ndarray:
        """The full actor observation vector.

        Assembled in the order given by ``spec.obs_terms`` rather than a hardcoded
        sequence, because the actor's group composition -- and therefore the order --
        is a config decision. Hardcoding it meant that renaming or reordering the
        observation groups silently produced a permuted vector.

        With ``history_length > 1`` the past frames of the history terms are appended
        after the current observation, matching NStepWrapper's layout
        ``[current][t-1][t-2]...`` newest-first.
        """
        parts = {t["name"]: self._term(t, data, ref, last_action)
                 for t in self.spec.obs_terms}
        cur = np.concatenate([parts[t["name"]] for t in self.spec.obs_terms])

        if self.spec.history_length > 1:
            hist = np.concatenate([parts[n] for n in self.spec.history_term_names])
            if self._hist_buf is None:          # first call of an episode
                self._hist_buf = np.tile(hist, (self.spec.history_length - 1, 1))
            out = np.concatenate([cur, self._hist_buf.reshape(-1)])
            self._hist_buf = np.roll(self._hist_buf, -1, axis=0)
            self._hist_buf[-1] = hist
        else:
            out = cur

        if out.shape[0] != self.spec.obs_dim:
            raise RuntimeError(
                f"built obs of dim {out.shape[0]}, spec says {self.spec.obs_dim}"
            )
        return out

    def reset_history(self) -> None:
        """Clear the history buffer. Call at episode boundaries.

        Without this a new episode's first frames carry the previous episode's tail,
        which is not a state the policy was trained on.
        """
        self._hist_buf = None

    # -- individual terms --------------------------------------------------

    def _t_joint_pos_rel(self, data, ref, last_action, term):
        qpos = np.asarray(data.qpos, dtype=np.float64)
        return qpos[self.qpos_adr] - self.default_joint_pos

    def _t_joint_vel(self, data, ref, last_action, term):
        return np.asarray(data.qvel, dtype=np.float64)[self.dof_adr]

    def _t_projected_gravity(self, data, ref, last_action, term):
        qpos = np.asarray(data.qpos, dtype=np.float64)
        root_quat = qpos[self.root_qpos_adr + 3: self.root_qpos_adr + 7]
        return quat_to_mat(root_quat).T @ np.array([0.0, 0.0, -1.0])

    def _t_base_ang_vel_entry(self, data, ref, last_action, term):
        # One gyro component per term (base_ang_vel_r/p/y), so the term's dim says which
        # slice: entries 3..5 of the free joint's qvel, in order of appearance.
        qvel = np.asarray(data.qvel, dtype=np.float64)
        ang = qvel[self.root_dof_adr + 3: self.root_dof_adr + 6]
        idx = self._ang_vel_order.index(term["name"])
        return ang[idx: idx + 1]

    def _t_last_action(self, data, ref, last_action, term):
        return np.asarray(last_action, dtype=np.float64)

    def _t_tracking_goal(self, data, ref, last_action, term):
        return self.tracking_goal(data, ref)

    def _t_future_frames(self, data, ref, last_action, term):
        # The next n frames of the reference alone, relative to its current frame: root
        # position and orientation in that frame's heading frame, root velocities in the
        # root frame, then joint positions and scaled joint velocities. Like training,
        # this indexes the trajectory's own qpos/qvel, free joint first.
        n = int(term["n_future"])
        qpos = np.asarray(ref["qpos_future"], dtype=np.float64)
        qvel = np.asarray(ref["qvel_future"], dtype=np.float64)
        heading_inv = yaw_rot(quat_to_mat(qpos[0, 3:7])).T
        rot = quat_to_mat(qpos[1:n + 1, 3:7])
        blocks = [(qpos[1:n + 1, :3] - qpos[0, :3]) @ heading_inv.T,
                  mat_to_6d_cols01(np.einsum("ij,njk->nik", heading_inv, rot)),
                  np.einsum("nij,ni->nj", rot, qvel[1:n + 1, :3]),
                  qvel[1:n + 1, 3:6],
                  qpos[1:n + 1, 7:],
                  float(term["joint_vel_scale"]) * qvel[1:n + 1, 6:]]
        return np.concatenate(blocks, axis=1).ravel()

    def _t_root_ori_rel(self, data, ref, last_action, term):
        # The reference root orientation in the robot root's frame, yaw included.
        quat = np.asarray(data.qpos, dtype=np.float64)[self.root_qpos_adr + 3:
                                                       self.root_qpos_adr + 7]
        ref_mat = self._ref_heading @ quat_to_mat(np.asarray(ref["qpos"])[3:7])
        return mat_to_6d_cols01(quat_to_mat(quat).T @ ref_mat)

    # -- terminations ------------------------------------------------------

    def termination_metrics(self, data: mujoco.MjData,
                            ref: dict[str, np.ndarray]) -> dict[str, float]:
        """The three z-only quantities ``G1RealTrackingTerminalStateHandler`` gates on.

        Returned rather than acted on: the sim2sim rig reports them so a run can be
        compared against training-time episode lengths without the rig silently
        cutting the rollout short.
        """
        robot_pos = np.asarray(data.xpos[self.body_ids], dtype=np.float64)
        robot_mat = np.asarray(data.xmat[self.body_ids], dtype=np.float64).reshape(-1, 3, 3)
        ref_pos = np.asarray(ref["body_pos_w"], dtype=np.float64)
        ref_mat = quat_to_mat(ref["body_quat_w"])

        r_apos, r_amat = robot_pos[self.anchor_row], robot_mat[self.anchor_row]
        f_apos, f_amat = ref_pos[self.anchor_row], ref_mat[self.anchor_row]

        delta_pos, delta_rot = anchor_delta(r_apos, r_amat, f_apos, f_amat)
        ref_pos_rel, _ = reanchor(ref_pos, ref_mat, f_apos, delta_pos, delta_rot)
        return {
            "anchor_pos_z": float(abs(f_apos[2] - r_apos[2])),
            # Compare projected gravity z, which is -R[2,2].
            "anchor_ori": float(abs(f_amat[2, 2] - r_amat[2, 2])),
            "ee_pos_z": float(np.max(np.abs(ref_pos_rel[self.ee_rows, 2]
                                            - robot_pos[self.ee_rows, 2]))),
        }

    def would_terminate(self, metrics: dict[str, float]) -> list[str]:
        hits = []
        for key, thr in self.spec.termination.items():
            if thr is not None and metrics.get(key, 0.0) > thr:
                hits.append(f"{key}={metrics[key]:.3f}>{thr}")
        return hits

    def reanchored_reference(self, data: mujoco.MjData, ref: dict[str, np.ndarray]):
        """Re-anchored reference body positions -- what the reward actually scores."""
        robot_pos, robot_mat = self.robot_anchor_frame(data)
        ref_pos = np.asarray(ref["body_pos_w"], dtype=np.float64)
        ref_mat = quat_to_mat(ref["body_quat_w"])
        delta_pos, delta_rot = anchor_delta(
            robot_pos, robot_mat, ref_pos[self.anchor_row], ref_mat[self.anchor_row]
        )
        pos_rel, _ = reanchor(ref_pos, ref_mat, ref_pos[self.anchor_row],
                              delta_pos, delta_rot)
        return pos_rel


# ---------------------------------------------------------------------------
# action


class PositionActionMapper:
    """Policy action -> joint-position target -> ``data.ctrl``.

    Mirrors :class:`lmj_real...g1_real_position.G1RealPositionControl`::

        target = default_joint_pos + action * scale       (actuator order)

    with ``scale = 0.25 * effort_limit / stiffness`` baked into the bundle. Whether
    ``target`` is written to ``ctrl`` directly or turned into a torque depends on the
    deploy model's actuators, which is resolved here rather than at the call site.
    """

    def __init__(self, spec: PolicySpec, model: mujoco.MjModel):
        self.spec = spec
        self.scale = np.asarray(spec.action_scale, dtype=np.float64)
        self.offset = np.asarray(spec.action_offset, dtype=np.float64)
        # A bound of None means the servo has no control range (the X2's do not), so the
        # target is not clipped on that side.
        self.ctrl_range = np.array(
            [[-np.inf if lo is None else lo, np.inf if hi is None else hi]
             for lo, hi in spec.ctrl_range], dtype=np.float64)
        self.kp = np.asarray(spec.kp, dtype=np.float64)
        self.kd = np.asarray(spec.kd, dtype=np.float64)
        self.effort_limit = np.asarray(spec.effort_limit, dtype=np.float64)

        # Resolve actuators through their transmission target, not by name: the MJCF names
        # the actuator after the joint it drives, but another MJCF may not.
        actuator_for_joint: dict[str, int] = {}
        for a in range(model.nu):
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[a, 0])
            )
            if name is not None:
                actuator_for_joint[name] = a
        missing = [n for n in spec.actuator_joint_names if n not in actuator_for_joint]
        if missing:
            raise ValueError(f"model has no actuator driving joints: {missing}")
        self.ctrl_index = np.array(
            [actuator_for_joint[n] for n in spec.actuator_joint_names], dtype=np.int32
        )

        # Joint state indices in the *actuator* order, needed for torque mode.
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                for n in spec.actuator_joint_names]
        self.qpos_adr = np.array([model.jnt_qposadr[i] for i in jids], dtype=np.int32)
        self.dof_adr = np.array([model.jnt_dofadr[i] for i in jids], dtype=np.int32)

        biastype = model.actuator_biastype[self.ctrl_index]
        self.position_actuators = bool(np.all(biastype == mujoco.mjtBias.mjBIAS_AFFINE))
        if not self.position_actuators and np.any(biastype == mujoco.mjtBias.mjBIAS_AFFINE):
            raise ValueError("mixed position/torque actuators are not supported")

    def target(self, action: np.ndarray) -> np.ndarray:
        """Joint-position target, in actuator order."""
        t = self.offset + np.asarray(action, dtype=np.float64) * self.scale
        return np.clip(t, self.ctrl_range[:, 0], self.ctrl_range[:, 1])

    def apply(self, data: mujoco.MjData, target: np.ndarray) -> None:
        """Write ``ctrl`` for one physics substep.

        For position servos the target *is* the ctrl and the engine closes the loop --
        which is what training did. For torque motors the PD loop runs here, at the
        simulation rate, using the gains lifted out of the training MJCF's servos. That
        is the sim2sim gap worth measuring: the same policy, the same gains, a
        Python-side loop and a plant with joint damping and friction.
        """
        if self.position_actuators:
            data.ctrl[self.ctrl_index] = target
        else:
            q = np.asarray(data.qpos[self.qpos_adr], dtype=np.float64)
            qd = np.asarray(data.qvel[self.dof_adr], dtype=np.float64)
            tau = self.kp * (target - q) - self.kd * qd
            tau = np.clip(tau, -self.effort_limit, self.effort_limit)
            data.ctrl[self.ctrl_index] = tau


# ---------------------------------------------------------------------------
# motions


MOTIONS_FILE = "motions.lmj.npz"


class MotionBundle:
    """The reference clips of a bundle: a single ``motions.lmj.npz`` Trajectory.

    The clips are stored in the native :class:`~loco_mujoco.core.trajectory.Trajectory`
    format rather than a bespoke per-clip npz plus a JSON manifest, so a bundle's
    motions can be loaded, inspected, replayed and re-retargeted with the same tools
    as the training data -- no separate format to keep in sync.

    That format is a plain npz, so it is read directly here. ``loco_mujoco`` imports
    jax, which has no business on the robot.

    Everything the old ``motions.json`` carried now comes from one of two places:
    the Trajectory's own ``info`` (frequency, joint and body names, ``split_points``
    for clip boundaries) or ``policy.json`` (tracked bodies, anchor index) -- which
    is where the policy-side truth lived anyway, so the two can no longer disagree.
    Per-clip names ride in ``info.metadata['clip_names']``.
    """

    def __init__(self, bundle_dir: str | Path, lazy: bool = True,
                 spec: "PolicySpec | None" = None):
        self.dir = Path(bundle_dir)
        self.spec = spec or PolicySpec.load(self.dir)

        # allow_pickle: `metadata` is a pickled dict. Bundles are our own exports.
        data = np.load(self.dir / MOTIONS_FILE, allow_pickle=True)

        self.dt: float = 1.0 / float(data["frequency"])
        self.body_names: list[str] = list(self.spec.tracked_body_names)
        self.anchor_body_index: int = int(self.spec.anchor_body_index)
        self.joint_names: list[str] = list(self.spec.joint_names)

        sp = np.asarray(data["split_points"])
        self._split = sp[sp >= 0].astype(np.int64)
        n_clips = len(self._split) - 1

        meta = data["metadata"].item() if "metadata" in data.files else None
        meta = meta or {}
        names = list(meta.get("clip_names") or [])
        if len(names) != n_clips:
            names = [f"motion_{i:04d}" for i in range(n_clips)]
        self.names: list[str] = names
        self._index = {n: i for i, n in enumerate(self.names)}

        # Resolve by NAME, never by position: the Trajectory carries all 30 joints
        # (including the free joint) and all 31 bodies, while the policy uses 29
        # actuated joints and 14 tracked bodies, in a different order.
        self.traj_joints = traj_joints = [str(j) for j in data["joint_names"]]
        free = np.asarray(data["jnt_type"]) == int(mujoco.mjtJoint.mjJNT_FREE)
        qpos_adr = np.concatenate([[0], np.cumsum(np.where(free, 7, 1))])
        qvel_adr = np.concatenate([[0], np.cumsum(np.where(free, 6, 1))])
        ind = [traj_joints.index(j) for j in self.joint_names]
        self._qpos_ind = qpos_adr[ind].astype(np.int64)
        self._qvel_ind = qvel_adr[ind].astype(np.int64)
        self.traj_bodies = traj_bodies = [str(b) for b in data["body_names"]]
        missing = [b for b in self.body_names if b not in traj_bodies]
        if missing:
            raise ValueError(
                f"{MOTIONS_FILE} has no body poses for {missing}; the bundle's "
                "tracked bodies are not all present in the trajectory."
            )
        self._body_ind = np.array([traj_bodies.index(b) for b in self.body_names],
                                  dtype=np.int64)

        self.n_future = self.spec.n_future
        self._qpos = np.asarray(data["qpos"])
        self._qvel = np.asarray(data["qvel"])
        self._xpos = np.asarray(data["xpos"])
        self._xquat = np.asarray(data["xquat"])
        self._cache: dict[str, dict[str, np.ndarray]] = {}
        if not lazy:
            for n in self.names:
                self.get(n)

    def __len__(self) -> int:
        return len(self.names)

    def _range(self, name: str) -> tuple[int, int]:
        i = self._index[name]
        return int(self._split[i]), int(self._split[i + 1])

    def get(self, name: str) -> dict[str, np.ndarray]:
        """One clip, sliced to the policy's joints and tracked bodies."""
        if name not in self._cache:
            lo, hi = self._range(name)
            self._cache[name] = {
                "joint_pos": self._qpos[lo:hi][:, self._qpos_ind],
                "joint_vel": self._qvel[lo:hi][:, self._qvel_ind],
                "body_pos_w": self._xpos[lo:hi][:, self._body_ind],
                "body_quat_w": self._xquat[lo:hi][:, self._body_ind],
                "qpos": self._qpos[lo:hi],
                "qvel": self._qvel[lo:hi],
            }
        return self._cache[name]

    def frame(self, name: str, t: int, end: int | None = None) -> dict[str, np.ndarray]:
        """One reference frame, in the dict shape :class:`ObservationBuilder` wants.

        ``end`` is the last frame the future frames may reach, the clip's own by default.
        """
        m = self.get(name)
        out = {k: m[k][t] for k in
               ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "qpos", "qvel")}
        if self.n_future:
            # Past the end of the clip, its last frame repeats, as in training.
            last = len(m["qpos"]) - 1 if end is None else min(end, len(m["qpos"]) - 1)
            idx = np.minimum(t + np.arange(self.n_future + 1), last)
            out["qpos_future"], out["qvel_future"] = m["qpos"][idx], m["qvel"][idx]
        return out

    def length(self, name: str) -> int:
        lo, hi = self._range(name)
        return hi - lo


# Model directory under lmj_real/models for each robot a bundle can name. Environment class
# names carry an ``Mjx`` prefix on the GPU backends; the model is the same.
_MODEL_FILES = {
    "G1Real": ("g1_real", "g1_real.xml"),
    "X2UltraReal": ("x2_ultra_real", "x2_ultra_real.xml"),
}


def training_model_xml(spec: PolicySpec) -> Path:
    """The MJCF in this repo that the bundle's policy was trained on."""
    robot = spec.robot.removeprefix("Mjx")
    if robot not in _MODEL_FILES:
        raise ValueError(f"no model known for robot {spec.robot!r}; add it to _MODEL_FILES")
    folder, name = _MODEL_FILES[robot]
    return Path(__file__).resolve().parents[1] / "lmj_real" / "models" / folder / name


def apply_model_options(model: mujoco.MjModel, spec: PolicySpec,
                        keys: tuple[str, ...] | None = None) -> None:
    """Copy the training model's solver settings onto a deploy model.

    Without this a stock MJCF runs at MuJoCo's default 2 ms timestep and default solver
    iteration counts, so a sim2sim comparison would be measuring the integrator as much
    as the plant.
    """
    for k, v in spec.model_options.items():
        if keys is not None and k not in keys:
            continue
        if not hasattr(model.opt, k):
            continue
        cur = getattr(model.opt, k)
        setattr(model.opt, k, type(cur)(v) if not isinstance(cur, np.ndarray) else v)


def apply_model_contact(model: mujoco.MjModel, spec: PolicySpec) -> None:
    """Give the training MJCF the contact setup the training environment gave it.

    Mirrors LocoMuJoCo's ``model_contact_conf``: it acts on every colliding geom outside
    the world body. Without it the model keeps the MJCF's own contact dimensions and
    self-collisions, and contacts between the robot's own bodies differ from training.
    """
    conf = spec.model_contact or {}
    self_collisions, condim = conf.get("self_collisions", True), conf.get("condim")
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0 or (model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0):
            continue
        if not self_collisions:
            model.geom_contype[g], model.geom_conaffinity[g] = 0, 1
        if condim is not None:
            model.geom_condim[g] = condim


def check_bundle_against_model(spec: PolicySpec, motions: MotionBundle,
                               model: mujoco.MjModel) -> list[str]:
    """Cheap consistency checks. Returns a list of human-readable warnings."""
    warn = []
    # MotionBundle now derives joint_names / body_names / anchor_body_index FROM the
    # spec, so comparing them against the spec would always pass. Check the trajectory
    # itself instead -- that it really contains what the spec asks to be sliced out,
    # and that the deploy model agrees.
    missing_j = [j for j in spec.joint_names if j not in motions.traj_joints]
    if missing_j:
        warn.append(f"{MOTIONS_FILE} is missing joints required by policy.json: "
                    f"{missing_j}")
    missing_b = [b for b in spec.tracked_body_names if b not in motions.traj_bodies]
    if missing_b:
        warn.append(f"{MOTIONS_FILE} is missing tracked bodies required by "
                    f"policy.json: {missing_b}")
    for j in spec.joint_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) < 0:
            warn.append(f"deploy model has no joint {j!r}")
    for b in spec.tracked_body_names:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) < 0:
            warn.append(f"deploy model has no body {b!r}")
    if not (0 <= spec.anchor_body_index < len(spec.tracked_body_names)):
        warn.append(f"anchor_body_index {spec.anchor_body_index} out of range for "
                    f"{len(spec.tracked_body_names)} tracked bodies")
    if abs(motions.dt - spec.control_dt) > 1e-9:
        warn.append(f"motion dt {motions.dt} != policy control dt {spec.control_dt}")
    if model.nu != len(spec.actuator_joint_names):
        warn.append(f"model nu={model.nu}, spec has "
                    f"{len(spec.actuator_joint_names)} actuated joints")
    return warn
