"""Train for a few steps, export, validate and roll out: the whole chain on a synthetic clip.

The policy is untrained, so it falls in the rollout. What is checked is that every stage
runs and that the bundle reproduces the training environment's observations and actions.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import make_slide_clip

REPO = Path(__file__).resolve().parents[1]
TINY = ["experiment.num_envs=4", "experiment.num_steps=8", "experiment.num_minibatches=2",
        "experiment.update_epochs=1", "experiment.total_timesteps=64",
        "experiment.timesteps_per_chunk=64", "experiment.hidden_layers=[64,64]",
        "experiment.env_params.th_params.max_n_samples=1000"]


def run(*args):
    done = subprocess.run([sys.executable, *map(str, args)], cwd=REPO, capture_output=True, text=True,
                          timeout=900, env={**os.environ, "WANDB_MODE": "disabled"})
    assert done.returncode == 0, (done.stdout[-3000:] + done.stderr[-3000:])
    return done.stdout


@pytest.mark.parametrize("robot,env_name", [("g1", "G1Real"), ("x2", "X2UltraReal")])
def test_train_export_validate_rollout(robot, env_name, tmp_path):
    clip, out, bundle = tmp_path / "clip.lmj.npz", tmp_path / "run", tmp_path / "bundle"
    make_slide_clip(env_name).save(str(clip))

    run(f"training/{robot}/train.py", f"hydra.run.dir={out}",
        "~experiment.task_factory.params.default_dataset_conf",
        f"+experiment.task_factory.params.custom_dataset_conf.paths=[{clip}]", *TINY)
    checkpoint = out / "PPOJax_saved.pkl"
    assert checkpoint.exists()

    run("deployment/export_bundle.py", "--checkpoint", checkpoint, "--output", bundle)
    for name in ("policy.json", "policy.npz", "motions.lmj.npz"):
        assert (bundle / name).exists()

    assert "PASS" in run("deployment/validate_bundle.py", "--bundle", bundle, "--checkpoint", checkpoint)
    assert "steps" in run("deployment/rollout_mujoco.py", "--bundle", bundle)
