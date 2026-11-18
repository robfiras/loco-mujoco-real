"""Export a trained PPO or SuperMimic checkpoint, of any robot here, into a standalone deploy bundle.

Actor export and reference-motion export happen in one pass, because both need the same
reconstructed environment.

    <output>/
        policy.json      spec: joint/actuator orders, action scale, default pose, PD
                         gains, tracked bodies, termination thresholds, obs layout
        policy.npz       actor MLP weights + frozen observation-normalizer statistics
        motions.lmj.npz  the selected clips as a loco_mujoco Trajectory; boundaries in
                         split_points, names in info.metadata['clip_names']

The bundle references neither jax, flax, ``loco_mujoco`` nor ``lmj_real``:
``lmj_deploy.py`` reads it with numpy + mujoco only. The clips stay in the native
trajectory format, which is a plain npz -- see ``export_motions``.

Two facts about the training stack drive the layout:

* The actor is a slice of a shared observation. For PPO, ``ActorCritic`` normalizes the
  **full**
  observation (734 dims for the shipped config) with a ``RunningMeanStd`` and only then
  indexes ``actor_obs_ind`` (283 dims, the ``policy`` group). Normalization is
  elementwise, so the bundle stores the mean/var already restricted to that slice.
  SuperMimic's actor reads two slices instead, ``goal_obs_group`` for its encoder and
  ``history_obs_group`` for its decoder; the bundle stores them back to back.
* Actions are in **actuator** order (shoulders first), while observations and
  the reference ``command`` term are in MuJoCo **joint** order (legs first). Both lists
  are exported; see the table in ``lmj_deploy``'s docstring.

Usage::

    python deployment/export_bundle.py \\
        --checkpoint outputs/2026-08-.../PPOJax_saved.pkl \\
        --output deploy_bundles/g1_real \\
        --max-motions 20
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys

from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lmj_deploy import (  # noqa: E402
    MOTIONS_FILE,
    ObservationBuilder,
    PolicySpec,
    load_actor,
)

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True ")

# Namespaces / agent-side params that are not env constructor kwargs. Same list
# eval_viser.py uses; if they leak through they land in **viewer_params.
NON_ENV_KEYS = {"robot", "actor_obs_group", "critic_obs_group"}

# lmj_deploy's action mapping is this control function's. The observation side is
# checked term by term against ObservationBuilder instead: a checkpoint trained with
# anything else would produce a bundle that looks fine and silently computes the wrong
# action or observation, so refuse up front.
REQUIRED_CONTROL = "G1RealPositionControl"

#: Constructor arguments a term needs at deploy time, as policy.json key -> attribute.
TERM_PARAMS = {
    "TrajFutureFrames": {"n_future": "_n_future", "joint_vel_scale": "_joint_vel_scale"},
}

#: SuperMimic's RunningNorm divides by sqrt(var + eps**2), with eps = 1e-2.
SUPERMIMIC_NORM_EPS = 1e-4


# ---------------------------------------------------------------------------
# environment reconstruction


def load_checkpoint(path: str):
    """``(agent_conf, agent_state, algorithm name)`` from a PPOJax or SuperMimicJax file."""
    import pickle

    with open(path, "rb") as f:
        d = pickle.load(f)
    if "wm_state" in d["agent_state"]:
        # Only on loco-mujoco branches that have SuperMimic, hence imported here.
        from loco_mujoco.algorithms import SuperMimicJax as algorithm
    else:
        from loco_mujoco.algorithms import PPOJax as algorithm
    agent_conf, agent_state = algorithm.from_dict(d)
    return agent_conf, agent_state, algorithm.__name__


def build_env_from_checkpoint(path: str, *, headless: bool = True,
                              disable_randomization: bool = True):
    """Rebuild the training env + trajectory from a PPOJax or SuperMimicJax checkpoint.

    Returns ``(env, traj, agent_conf, agent_state, algorithm)``. Domain randomization is
    switched off by default and MjWarp is forced off, since
    everything here runs single-env on CPU.
    """
    import lmj_real  # noqa: F401  (registers the env + components)
    from loco_mujoco import TaskFactory

    agent_conf, agent_state, algorithm = load_checkpoint(path)
    config = agent_conf.config
    OmegaConf.set_struct(config, False)

    env_params = {
        k: v for k, v in config.experiment.env_params.items() if k not in NON_ENV_KEYS
    }
    env_params["headless"] = headless
    if "use_mjwarp" in env_params:
        env_params["use_mjwarp"] = False
    if disable_randomization:
        dr = dict(env_params.get("domain_randomization_params") or {})
        if dr:
            # Every observation-noise knob the randomizer was given, whatever its version.
            dr.update({k: 0.0 for k in dr if k.endswith("_obs_noise")})
            dr.update(enable_push=False, enable_encoder_bias=False,
                      enable_foot_friction=False, enable_base_com=False,
                      joint_pos_obs_noise=0.0, joint_vel_obs_noise=0.0,
                      base_ang_vel_obs_noise=0.0)
            env_params["domain_randomization_params"] = dr
        # Reset noise is MJX-only, and everything here runs on plain MuJoCo.
        init = dict(env_params.get("init_state_params") or {})
        if init:
            env_params["init_state_params"] = {
                k: (0.0 if k == "joint_pos_noise" else [0.0, 0.0, 0.0])
                if k.endswith("_noise") else v for k, v in init.items()}

    factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)
    factory_params = OmegaConf.to_container(
        config.experiment.task_factory.params, resolve=True
    )

    # The config carries a list of .npz paths rather than a Trajectory; resolve it the
    # way the training script does.
    custom = factory_params.get("custom_dataset_conf")
    if isinstance(custom, dict) and "paths" in custom:
        from loco_mujoco.core.trajectory import Trajectory
        from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

        paths = list(custom["paths"])
        print(f"[export] loading {len(paths)} trajectory file(s)", flush=True)
        loaded = [Trajectory.load(p) for p in paths]
        factory_params["custom_dataset_conf"] = CustomDatasetConf(
            traj=Trajectory.concatenate(loaded) if len(loaded) > 1 else loaded[0]
        )

    env, traj = factory.make(**env_params, **factory_params)
    return env, traj, agent_conf, agent_state, algorithm


def has_body_poses(traj) -> bool:
    return np.asarray(traj.data.xpos).size > 0


def require_body_poses(traj) -> None:
    """``GoalTrajMimicAnchor`` reads reference body poses straight off the trajectory.

    Anything G1Real was actually trained on therefore has them (``.lmj.npz`` files are
    written through ``ExtendTrajData``). Hitting
    this means the checkpoint's config points at different files than it trained on.
    """
    if not has_body_poses(traj):
        raise SystemExit(
            "the trajectory in this checkpoint's config carries no xpos/xquat, so the "
            "environment's own tracking goal cannot be evaluated. The training data "
            "must have had them -- check that "
            "the dataset config in experiment.task_factory.params still matches what "
            "the run used."
        )


def body_poses_by_fk(model, traj, body_ids: np.ndarray):
    """``(xpos, xquat)`` for the tracked bodies, from the trajectory if present.

    Falls back to a forward-kinematics pass over ``qpos``, which is enough for the
    bundle: the deploy side only needs body poses, not the ``cvel`` / ``subtree_com``
    that a full ``ExtendTrajData`` would also produce.
    """
    if has_body_poses(traj):
        return (np.asarray(traj.data.xpos)[:, body_ids],
                np.asarray(traj.data.xquat)[:, body_ids])

    import mujoco

    print("[export] trajectory has no xpos/xquat -- recomputing tracked-body poses "
          "by forward kinematics", flush=True)
    qpos = np.asarray(traj.data.qpos)
    data = mujoco.MjData(model)
    xpos = np.empty((qpos.shape[0], len(body_ids), 3))
    xquat = np.empty((qpos.shape[0], len(body_ids), 4))
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_kinematics(model, data)
        xpos[t] = data.xpos[body_ids]
        xquat[t] = data.xquat[body_ids]
    return xpos, xquat


# ---------------------------------------------------------------------------
# spec / weights


def _check_components(env, terms) -> None:
    problems = []
    control = type(env._control_func).__name__
    if control != REQUIRED_CONTROL:
        problems.append(f"  control: checkpoint has {control}, bundle format assumes "
                        f"{REQUIRED_CONTROL}")
    unknown = sorted({t["type"] for t in terms} - set(ObservationBuilder._TERM_BUILDERS))
    if unknown:
        problems.append(f"  actor observation terms lmj_deploy.py does not implement: "
                        f"{unknown}")
    if problems:
        raise SystemExit(
            "This exporter only understands the G1Real stack.\n"
            + "\n".join(problems)
            + "\nlmj_deploy.py would silently compute the wrong action or observation "
              "for anything else."
        )


def actor_obs_terms(env, actor_group):
    """Observation terms in the actor group(s), in the order the network sees them.

    ``actor_group`` is a group name or a list of them. As in ``get_obs_ind_by_group``,
    groups come in the order given and terms keep container order within each group.

    Returns ``(terms, actor_obs_ind)``. Raises if the group's indices are not exactly
    the concatenation of its terms' ``obs_ind`` -- if that ever stops holding, the
    hand-built vector would be silently permuted.
    """
    actor_ind = np.asarray(env.obs_container.get_obs_ind_by_group(actor_group))
    wanted = [actor_group] if isinstance(actor_group, str) else list(actor_group)
    terms, cursor, collected = [], 0, []
    for group in wanted:
        for name, obs in env.obs_container.items():
            # `group` may be a str, a list, or an omegaconf ListConfig depending on how
            # the observation was declared -- normalize rather than isinstance-checking.
            groups = [obs.group] if isinstance(obs.group, str) else list(obs.group)
            if group not in groups:
                continue
            ind = np.asarray(obs.obs_ind)
            term = {
                "name": name,
                "type": type(obs).__name__,
                "dim": int(ind.size),
                "start": cursor,
            }
            for key, attr in TERM_PARAMS.get(term["type"], {}).items():
                term[key] = getattr(obs, attr)
            terms.append(term)
            collected.append(ind)
            cursor += int(ind.size)
    collected = np.concatenate(collected)
    # With an observation history, actor_ind additionally contains the appended history
    # block, whose indices lie beyond the base observation and belong to no term. Those
    # are reconstructed at deploy time from the history terms, so compare only the
    # leading current-frame slice.
    head = actor_ind[:collected.size]
    if not np.array_equal(collected, head):
        raise SystemExit(
            "actor observation group is not the in-order concatenation of its terms; "
            "the bundle's obs layout would be wrong.\n"
            f"  group indices : {head}\n  term indices  : {collected}"
        )
    return terms, actor_ind


def extract_weights(agent_state, actor_ind: np.ndarray, n_seeds: int,
                    train_state_seed: int) -> dict[str, np.ndarray]:
    """Pull the actor MLP + normalizer statistics out of the flax train state."""
    ts = agent_state.train_state
    params, run_stats = ts.params, ts.run_stats
    if n_seeds > 1:
        import jax
        params = jax.tree.map(lambda x: x[train_state_seed], params)
        run_stats = jax.tree.map(lambda x: x[train_state_seed], run_stats)

    # ActorCritic builds the actor first, so it is FullyConnectedNet_0; the critic
    # (FullyConnectedNet_1) is deliberately not exported.
    actor = params["FullyConnectedNet_0"]
    dense_keys = sorted(
        (k for k in actor if k.startswith("Dense_")), key=lambda k: int(k.split("_")[1])
    )
    out: dict[str, np.ndarray] = {}
    for i, k in enumerate(dense_keys):
        out[f"w{i}"] = np.asarray(actor[k]["kernel"], dtype=np.float32)
        out[f"b{i}"] = np.asarray(actor[k]["bias"], dtype=np.float32)
    out["log_std"] = np.asarray(params["log_std"], dtype=np.float32)

    stats = run_stats["RunningMeanStd_0"]
    mean = np.asarray(stats["mean"], dtype=np.float64)
    var = np.asarray(stats["var"], dtype=np.float64)
    out["obs_mean"] = mean[actor_ind].astype(np.float32)
    out["obs_var"] = var[actor_ind].astype(np.float32)
    out["obs_norm_count"] = np.asarray(stats["count"], dtype=np.float64)

    if out["w0"].shape[0] != actor_ind.size:
        raise SystemExit(
            f"actor first layer expects {out['w0'].shape[0]} inputs but the actor "
            f"observation group has {actor_ind.size} dims"
        )
    return out


def extract_encoder_decoder(agent_state, actor_ind: np.ndarray, seed: int,
                            n_goal: int, n_frames: int) -> dict[str, np.ndarray]:
    """SuperMimic's deployed (EMA) actor + normalizer statistics, as numpy arrays."""
    params, norm = _supermimic_params(agent_state, seed)
    out: dict[str, np.ndarray] = {}
    for part in ("enc", "dec"):
        dense = sorted((k for k in params[part] if k.startswith("Dense_")),
                       key=lambda k: int(k.split("_")[1]))
        for i, k in enumerate(dense):
            out[f"{part}_w{i}"] = np.asarray(params[part][k]["kernel"], dtype=np.float32)
            out[f"{part}_b{i}"] = np.asarray(params[part][k]["bias"], dtype=np.float32)
    if norm is None:
        out["obs_mean"] = np.zeros(actor_ind.size, dtype=np.float32)
        out["obs_var"] = np.ones(actor_ind.size, dtype=np.float32)
        out["obs_norm_eps"] = np.asarray(0.0)
    else:
        out["obs_mean"] = np.asarray(norm.mean)[actor_ind].astype(np.float32)
        out["obs_var"] = np.asarray(norm.var)[actor_ind].astype(np.float32)
        out["obs_norm_eps"] = np.asarray(SUPERMIMIC_NORM_EPS)
        out["obs_norm_count"] = np.asarray(norm.count, dtype=np.float64)

    n_hist = actor_ind.size - n_goal
    latent = out[_last("enc", out)].shape[1]
    if out["enc_w0"].shape[0] != n_goal or out["dec_w0"].shape[0] != latent + n_frames * n_hist:
        raise SystemExit(
            f"actor expects {out['enc_w0'].shape[0]} goal and {out['dec_w0'].shape[0]} "
            f"decoder inputs, but the groups give {n_goal} goal dims and "
            f"{latent} + {n_frames} x {n_hist} decoder dims"
        )
    return out


def _last(part: str, weights: dict[str, np.ndarray]) -> str:
    i = 0
    while f"{part}_w{i + 1}" in weights:
        i += 1
    return f"{part}_w{i}"


def _supermimic_params(agent_state, seed: int):
    """Deployed actor parameters and normalizer, for one training seed."""
    import jax

    params, norm = agent_state.deployed_params, agent_state.actor_norm
    if np.ndim(params["enc"]["Dense_0"]["kernel"]) == 3:  # trained with a seed axis
        params = jax.tree.map(lambda x: x[seed], params)
        norm = None if norm is None else jax.tree.map(lambda x: x[seed], norm)
    return params, norm


def build_spec(env, config, checkpoint: str, terms, action_dim: int,
               history_length: int = 1,
               history_term_names=None, actor_type: str = "mlp",
               encoder_dim: int = 0, actor_history_length: int = 1) -> PolicySpec:
    import mujoco

    model = env._model
    cf = env._control_func
    props = env._get_all_info_properties()

    joint_names = list(config.experiment.env_params.robot.joints_to_consider)
    qpos_adr = []
    for n in joint_names:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        qpos_adr.append(int(model.jnt_qposadr[j]))

    # The goal's reference `command` term reads every non-free joint in *model* joint
    # order. The bundle ships reference columns in `joint_names` order, so the two
    # orders have to coincide -- if they ever diverge the command would be permuted and
    # nothing downstream would notice.
    goal = next((o for o in env.obs_container.values()
                 if type(o).__name__ == "GoalTrajMimicAnchor"), None)
    if goal is not None and not np.array_equal(np.asarray(goal._qpos_ind),
                                               np.asarray(qpos_adr)):
        raise SystemExit(
            "robot.joints_to_consider is not in model joint order; the exported "
            "reference command columns would be permuted.\n"
            f"  goal qpos indices : {np.asarray(goal._qpos_ind)}\n"
            f"  joints_to_consider: {np.asarray(qpos_adr)}"
        )

    from lmj_real.core.utils.keyframe import default_qpos

    dq = default_qpos(model)
    action_idx = np.asarray(env._action_indices)
    kp = np.asarray(model.actuator_gainprm[action_idx, 0], dtype=float)
    kd = -np.asarray(model.actuator_biasprm[action_idx, 2], dtype=float)
    effort = np.asarray(model.actuator_forcerange[action_idx, 1], dtype=float)

    tsh = env._terminal_state_handler
    termination = {
        "anchor_pos_z": getattr(tsh, "_anchor_pos_z_threshold", None),
        "anchor_ori": getattr(tsh, "_anchor_ori_threshold", None),
        "ee_pos_z": getattr(tsh, "_ee_body_pos_z_threshold", None),
    }

    hidden = config.experiment.hidden_layers
    hidden = list(hidden) if not isinstance(hidden, str) else json.loads(hidden)

    # Compiled option values, not the yaml: model_option_conf only names a few of them
    # and the rest come from MuJoCo defaults at compile time.
    opt = model.opt
    model_options = {
        k: (int(getattr(opt, k)) if isinstance(getattr(opt, k), (int, np.integer))
            else float(getattr(opt, k)))
        for k in ("timestep", "iterations", "ls_iterations", "ccd_iterations",
                  "integrator", "solver", "cone", "jacobian", "impratio",
                  "tolerance", "ls_tolerance", "disableflags", "enableflags")
        if hasattr(opt, k)
    }

    return PolicySpec(
        robot=type(env).__name__,
        env_name=str(config.experiment.env_params.env_name),
        source_checkpoint=str(Path(checkpoint).resolve()),
        sim_timestep=float(model.opt.timestep),
        n_substeps=int(env._n_substeps),
        model_options=model_options,
        model_contact=dict(config.experiment.env_params.get("model_contact_conf") or {}),
        joint_names=joint_names,
        default_joint_pos=[float(dq[i]) for i in qpos_adr],
        actuator_joint_names=list(cf._jnt_names),
        action_scale=[float(x) for x in cf.scale],
        action_offset=[float(x) for x in cf._nominal_joint_positions],
        # None for a servo without a control range: JSON has no infinity.
        ctrl_range=[[float(x) if np.isfinite(x) else None for x in r]
                    for r in cf._ctrl_ranges],
        kp=[float(x) for x in kp],
        kd=[float(x) for x in kd],
        effort_limit=[float(x) for x in effort],
        tracked_body_names=list(props["tracked_body_names"]),
        anchor_body_index=list(props["tracked_body_names"]).index(props["anchor_body_name"]),
        # Read off the live goal object, not the config, so the bundle records what
        # the policy actually saw. A 283-dim actor with a builder that emits 286 is
        # the failure this prevents.
        use_anchor=bool(getattr(env._goal, "_use_anchor", True)),
        use_anchor_pos=bool(getattr(env._goal, "_use_anchor_pos", True)),
        history_length=history_length,
        history_term_names=list(history_term_names or []),
        actor_type=actor_type,
        encoder_dim=int(encoder_dim),
        actor_history_length=int(actor_history_length),
        end_effector_body_names=list(props["end_effector_body_names"]),
        root_joint_name="root",
        hidden_layers=hidden,
        activation=str(config.experiment.activation),
        obs_dim=int(sum(t["dim"] for t in terms)),
        action_dim=int(action_dim),
        obs_terms=terms,
        termination=termination,
    )


# ---------------------------------------------------------------------------
# motions


def export_motions(env, traj, spec: PolicySpec, out_dir: Path, *,
                   max_motions: int = 0, name_patterns: str = "") -> dict:
    """Write the selected clips as a single loco_mujoco ``Trajectory``.

    The bundle stores motions in the native trajectory format (``motions.lmj.npz``)
    rather than per-clip arrays plus a JSON manifest. That costs a ``loco_mujoco``
    dependency on the motion side of the deploy runtime -- the policy side stays
    numpy-only -- and buys three things: the clips stay loadable by every existing
    tool (replay, retargeting, ``ExtendTrajData``), the per-frame quantities the old
    format dropped (``cvel``, ``subtree_com``, sites, and all 31 bodies rather than
    the 14 tracked ones) survive the round trip, and there is no second description
    of joint/body order that can drift away from ``policy.json``.

    Clip identity rides in ``info.metadata['clip_names']``; boundaries are the
    trajectory's own ``split_points``.
    """
    import dataclasses

    from loco_mujoco.core.trajectory import Trajectory

    # trim() first: `process_trajectory` pads every per-frame array out to
    # th_params.max_n_samples, so the arrays are longer than split_points[-1]. Without
    # this the `arr.shape[0] != n_frames` guard in _subset -- meant to skip fields that
    # are not per-frame -- skips *every* field instead, and the export silently ships
    # the whole padding (measured: 655 MB for one 610-frame clip at max_n_samples
    # 200_000). split_points would still be rewritten correctly, so MotionBundle reads
    # the right frames and nothing looks wrong except the file size.
    data = traj.data.trim()
    split = np.asarray(data.split_points)
    split = split[split >= 0].astype(np.int64)
    n_traj = len(split) - 1
    n_frames = int(split[-1])

    # Clip names, in order of preference:
    #   1. the trajectory's own info.metadata['clip_names'] -- set by converters, and the
    #      only source that survives concatenation;
    #   2. the task names or source filenames from the training config;
    #   3. positional.
    # Preference matters for concatenated datasets: `paths` holds ONE entry for a file
    # containing many clips, so relying on it named clip 0 after the file and left the
    # rest as motion_0001.. -- useless for picking a motion in sim2sim.
    sources = list(env.__dict__.get("_export_source_paths") or [])
    meta_names = list((traj.info.metadata or {}).get("clip_names") or [])
    if meta_names and len(meta_names) != n_traj:
        print(f"[export] WARNING: metadata has {len(meta_names)} clip_names for "
              f"{n_traj} clips; ignoring them.", flush=True)
        meta_names = []

    patterns = [p.strip() for p in name_patterns.split(",") if p.strip()]
    keep, names, kept_sources = [], [], []
    for i in range(n_traj):
        if meta_names:
            name = meta_names[i]
        elif i < len(sources):
            name = Path(sources[i]).name.replace(".lmj.npz", "").replace(".npz", "")
        else:
            name = f"motion_{i:04d}"
        if patterns and not any(fnmatch.fnmatch(name, p) for p in patterns):
            continue
        if max_motions and len(keep) >= max_motions:
            break
        keep.append(i)
        names.append(name)
        kept_sources.append(sources[i] if i < len(sources) else None)

    if not keep:
        raise SystemExit("no clips selected -- check --motion-names / --max-motions")

    ranges = [(int(split[i]), int(split[i + 1])) for i in keep]
    lengths = [hi - lo for lo, hi in ranges]

    def _subset(value):
        """Slice a per-frame array to the kept clips; pass anything else through."""
        if value is None:
            return None
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] != n_frames:
            return value
        return np.concatenate([arr[lo:hi] for lo, hi in ranges], axis=0)

    updates = {}
    for f in dataclasses.fields(data):
        if f.name == "split_points":
            continue
        updates[f.name] = _subset(getattr(data, f.name))
    new_split = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    new_data = data.replace(split_points=new_split, **updates)

    # Reuse the training info verbatim -- joint names, body names, frequency and the
    # reduced model all stay exactly as the policy was trained against -- and record
    # provenance in metadata.
    info = traj.info
    meta = dict(info.metadata or {})
    meta.update(
        clip_names=list(names),
        clip_sources=[str(s) if s is not None else None for s in kept_sources],
        source_checkpoint=spec.source_checkpoint,
        n_trajectories_in_checkpoint=int(n_traj),
        exported_by="deployment/export_bundle.py",
    )
    info.metadata = meta

    # Remove the previous format's artefacts so a re-export into an existing
    # directory cannot leave a stale motions.json that nothing reads any more.
    legacy_json = out_dir / "motions.json"
    if legacy_json.exists():
        legacy_json.unlink()
    legacy_dir = out_dir / "motions"
    if legacy_dir.is_dir():
        for stale in legacy_dir.glob("*.npz"):
            stale.unlink()
        try:
            legacy_dir.rmdir()
        except OSError:
            pass

    Trajectory(info=info, data=new_data).save(str(out_dir / MOTIONS_FILE))

    return {
        "file": MOTIONS_FILE,
        "names": names,
        "lengths": lengths,
        "n_trajectories_in_checkpoint": n_traj,
    }


# ---------------------------------------------------------------------------


def self_test_encoder_decoder(agent_conf, agent_state, actor, goal_ind, hist_ind,
                              obs_dim_full: int, seed: int, steps: int = 16):
    """numpy actor vs SuperMimic's own policy function, over a stream of observations.

    Returns two numbers. The first runs the jax network in float32, as the numpy actor
    does, so it checks the export itself. The second runs it as trained, possibly in
    bfloat16, and is what deployment changes by computing in float32.
    """
    import dataclasses

    import jax.numpy as jnp
    from loco_mujoco.algorithms import SuperMimicJax

    params, norm = _supermimic_params(agent_state, seed)
    n_frames = int(agent_conf.config.experiment.history_len)
    rng = np.random.default_rng(0)
    mean = np.zeros(obs_dim_full) if norm is None else np.asarray(norm.mean)
    std = np.ones(obs_dim_full) if norm is None else np.sqrt(np.asarray(norm.var) + 1e-8)
    stream = mean + std * rng.standard_normal((steps, obs_dim_full))

    def run(net):
        act = SuperMimicJax.policy_fn(dataclasses.replace(agent_conf, actor_net=net))
        history, out = None, []
        for o in stream:
            o = jnp.asarray(o)[None]
            frame = o[:, None, jnp.asarray(hist_ind)]
            history = (jnp.repeat(frame, n_frames, axis=1) if history is None
                       else jnp.concatenate([history[:, 1:], frame], axis=1))
            out.append(np.asarray(act(params, norm, o, history))[0])
        return np.stack(out)

    actor_ind = np.concatenate([goal_ind, hist_ind])
    actor.reset()
    mine = np.stack([actor.act(o[actor_ind]) for o in stream])
    exact = run(agent_conf.actor_net.clone(compute_dtype="float32"))
    trained = run(agent_conf.actor_net)
    return float(np.abs(exact - mine).max()), float(np.abs(trained - mine).max())


def frozen_run_stats(run_stats):
    """The normalizer statistics with a count so large that new samples no longer move them.

    Deployment normalizes with fixed statistics. LocoMuJoCo's ``RunningMeanStd`` has no
    such mode: it updates on every call, which shifts the output noticeably while the
    count is small. A huge count makes the update vanish.
    """
    import jax

    return {k: ({**v, "count": jax.numpy.full_like(v["count"], 1e15)} if "count" in v else v)
            for k, v in run_stats.items()}


def self_test(agent_conf, agent_state, actor, actor_ind: np.ndarray,
              obs_dim_full: int, n_seeds: int, seed: int, batch: int = 8) -> float:
    """numpy actor vs the jax network, on random full-width observations.

    The jax ``RunningMeanStd`` folds the incoming batch into its statistics before
    normalizing, while the bundle's are frozen. The comparison freezes the jax side too
    (see :func:`frozen_run_stats`), so it holds for a checkpoint of any age.
    """
    import jax
    import jax.numpy as jnp

    ts = agent_state.train_state
    params, run_stats = ts.params, ts.run_stats
    if n_seeds > 1:
        params = jax.tree.map(lambda x: x[seed], params)
        run_stats = jax.tree.map(lambda x: x[seed], run_stats)

    rng = np.random.default_rng(0)
    mean = np.asarray(run_stats["RunningMeanStd_0"]["mean"])
    std = np.sqrt(np.asarray(run_stats["RunningMeanStd_0"]["var"]) + 1e-8)
    obs = mean + std * rng.standard_normal((batch, obs_dim_full))

    y, _ = agent_conf.network.apply(
        {"params": params, "run_stats": frozen_run_stats(run_stats)}, jnp.asarray(obs),
        mutable=["run_stats"],
    )
    ref = np.asarray(y[0].mean())
    out = np.stack([actor.act(o[actor_ind]) for o in obs])
    return float(np.abs(ref - out).max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="PPOJax_saved.pkl or SuperMimicJax_saved.pkl")
    ap.add_argument("--output", required=True, help="bundle output dir")
    ap.add_argument("--train-state-seed", type=int, default=0,
                    help="which training seed to export when n_seeds > 1")
    ap.add_argument("--max-motions", type=int, default=0,
                    help="cap the number of exported clips (0 = all). "
                         "Datasets with many clips produce a very large bundle otherwise.")
    ap.add_argument("--motion-names", default="",
                    help="comma-separated glob patterns on the clip name")
    ap.add_argument("--no-motions", action="store_true",
                    help="export the policy only")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    env, traj, agent_conf, agent_state, algorithm = build_env_from_checkpoint(
        args.checkpoint)
    config = agent_conf.config
    print(f"[export] {algorithm} checkpoint, env={type(env).__name__} "
          f"obs={env.info.observation_space.shape} "
          f"act={env.info.action_space.shape}", flush=True)
    action_dim = int(env.info.action_space.shape[0])
    obs_dim_full = int(env.info.observation_space.shape[0])

    if algorithm == "SuperMimicJax":
        spec = export_supermimic(env, agent_conf, agent_state, args, out, action_dim,
                                 obs_dim_full)
    else:
        spec = export_ppo(env, agent_conf, agent_state, args, out, action_dim,
                          obs_dim_full)
    _export_motions_step(env, traj, spec, config, args, out)

    print(f"\n[export] bundle written to {out}")
    print("[export] next: python deployment/validate_bundle.py "
          f"--bundle {out} --checkpoint {args.checkpoint}")


def export_supermimic(env, agent_conf, agent_state, args, out: Path, action_dim: int,
                      obs_dim_full: int) -> PolicySpec:
    exp = agent_conf.config.experiment
    n_frames = int(getattr(exp, "history_len", 0) or 0)
    if n_frames <= 0:
        raise SystemExit("only SuperMimic's encoder-decoder actor (history_len > 0) is "
                         "supported; this checkpoint has a plain MLP actor")
    goal_terms, goal_ind = actor_obs_terms(env, exp.goal_obs_group)
    hist_terms, hist_ind = actor_obs_terms(env, exp.history_obs_group)
    for what, mine, theirs in (("goal", goal_ind, agent_conf.goal_ind),
                               ("history", hist_ind, agent_conf.history_ind)):
        if not np.array_equal(mine, np.asarray(theirs)):
            raise SystemExit(f"the checkpoint's {what} indices differ from the env's "
                             f"{what} group: the bundle's layout would be wrong")
    terms = goal_terms + [{**t, "start": t["start"] + goal_ind.size} for t in hist_terms]
    _check_components(env, terms)
    actor_ind = np.concatenate([goal_ind, hist_ind])

    weights = extract_encoder_decoder(agent_state, actor_ind, args.train_state_seed,
                                      goal_ind.size, n_frames)
    spec = build_spec(env, agent_conf.config, args.checkpoint, terms,
                      action_dim=action_dim, actor_type="encoder_decoder",
                      encoder_dim=goal_ind.size, actor_history_length=n_frames)
    spec.save(out)
    np.savez(out / "policy.npz", **weights)

    _print_terms(spec, f"encoder {goal_ind.size} dims, decoder {n_frames} frames of "
                       f"{hist_ind.size}")
    print(f"[export] net_dtype {getattr(exp, 'net_dtype', 'float32')}, "
          f"{int((np.sqrt(weights['obs_var'] + weights['obs_norm_eps']) < 1e-3).sum())} "
          f"of {spec.obs_dim} dims have std < 1e-3")

    actor = load_actor(out)
    exact, trained = self_test_encoder_decoder(agent_conf, agent_state, actor, goal_ind,
                                               hist_ind, obs_dim_full,
                                               args.train_state_seed)
    print(f"[export] numpy actor vs jax network in float32, max |diff| = {exact:.3e}")
    print(f"[export] numpy actor vs jax network as trained,  max |diff| = {trained:.3e}"
          "  (float32 vs the training dtype; expected, not checked)")
    if exact > 1e-4:
        raise SystemExit(f"self-test FAILED: max diff {exact:.3e}")
    print("[export] PASS: the bundle reproduces the checkpoint's action.")
    return spec


def _print_terms(spec: PolicySpec, what: str) -> None:
    print(f"[export] actor observation: {spec.obs_dim} dims, {what}")
    for t in spec.obs_terms:
        print(f"           {t['start']:4d}..{t['start'] + t['dim'] - 1:4d}  "
              f"{t['name']:20s} {t['type']}")
    print(f"[export] hidden {spec.hidden_layers} {spec.activation}, "
          f"action dim {spec.action_dim}, control {1 / spec.control_dt:.1f} Hz")


def export_ppo(env, agent_conf, agent_state, args, out: Path, action_dim: int,
               obs_dim_full: int) -> PolicySpec:
    config = agent_conf.config
    actor_group = config.experiment.actor_obs_group
    if actor_group is None:
        raise SystemExit(
            "checkpoint has no experiment.actor_obs_group -- actor and critic shared "
            "one observation, which this bundle format does not describe"
        )
    terms, actor_ind = actor_obs_terms(env, actor_group)
    _check_components(env, terms)

    # Observation history. The deploy side must rebuild it, or a policy trained with
    # history would be fed a single frame -- the network would see the right dimension
    # count only by accident and the wrong values regardless.
    history_length = int(getattr(config.experiment, "len_obs_history", 1) or 1)
    history_group = getattr(config.experiment, "obs_history_group", None)
    history_term_names = []
    if history_length > 1:
        if history_group is None:
            raise SystemExit(
                "len_obs_history > 1 but obs_history_group is unset: the policy was "
                "trained with the legacy whole-observation stacking, which this bundle "
                "format does not describe."
            )
        hist_ind = set(np.asarray(
            env.obs_container.get_obs_ind_by_group(history_group)).tolist())
        for name, obs in env.obs_container.items():
            if set(np.asarray(obs.obs_ind).tolist()) <= hist_ind:
                history_term_names.append(name)
        print(f"[export] history: {history_length} frames of "
              f"{history_term_names} ({len(hist_ind)} dims/frame)", flush=True)
    n_seeds = int(getattr(config.experiment, "n_seeds", 1) or 1)

    weights = extract_weights(agent_state, actor_ind, n_seeds, args.train_state_seed)
    spec = build_spec(env, config, args.checkpoint, terms,
                      history_length=history_length,
                      history_term_names=history_term_names,
                      action_dim=action_dim)
    spec.save(out)
    np.savez(out / "policy.npz", **weights)

    _print_terms(spec, f"group {actor_group!r}")
    print(f"[export] normalizer count {float(weights['obs_norm_count']):.3e}; "
          f"{int((np.sqrt(weights['obs_var'] + 1e-8) < 1e-3).sum())} of {spec.obs_dim} "
          f"dims have std < 1e-3 (see MlpActor's std_floor)")

    actor = load_actor(out)
    diff = self_test(agent_conf, agent_state, actor, actor_ind, obs_dim_full, n_seeds,
                     args.train_state_seed)
    print(f"[export] numpy actor vs jax network, max |diff| = {diff:.3e}")
    if diff > 1e-4:
        raise SystemExit(f"self-test FAILED: max diff {diff:.3e}")
    print("[export] PASS: the bundle reproduces the checkpoint's mean action.")
    return spec


def _export_motions_step(env, traj, spec: PolicySpec, config, args, out: Path) -> None:
    if not args.no_motions:
        # Record the task names or source filenames so exported clips keep their identity.
        params = OmegaConf.to_container(config.experiment.task_factory.params,
                                        resolve=True)
        custom = params.get("custom_dataset_conf")
        if isinstance(custom, dict) and "paths" in custom:
            env._export_source_paths = list(custom["paths"])
        default = params.get("default_dataset_conf")
        if isinstance(default, dict):
            task = default.get("task", "walk")
            env._export_source_paths = [task] if isinstance(task, str) else list(task)
        lafan = params.get("lafan1_dataset_conf")
        if isinstance(lafan, dict):
            names = lafan.get("dataset_name") or []
            env._export_source_paths = [names] if isinstance(names, str) else list(names)
        summary = export_motions(env, traj, spec, out,
                                 max_motions=args.max_motions,
                                 name_patterns=args.motion_names)
        frames = sum(summary["lengths"])
        print(f"[export] {len(summary['names'])} of "
              f"{summary['n_trajectories_in_checkpoint']} clips "
              f"({frames} frames, {frames * spec.control_dt:.1f} s) "
              f"-> {summary['file']}")
        if args.max_motions and summary["n_trajectories_in_checkpoint"] > args.max_motions:
            print(f"[export] NOTE: capped at --max-motions "
                  f"{args.max_motions}; the policy trained on all "
                  f"{summary['n_trajectories_in_checkpoint']}.")


if __name__ == "__main__":
    main()
