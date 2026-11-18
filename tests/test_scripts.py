"""Every entry script that needs no robot middleware starts and prints its help."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "training/g1/train.py", "training/x2/train.py",
    "training/g1/train_supermimic.py", "training/x2/train_supermimic.py",
    "training/eval_viser.py", "training/replay_trajectory.py", "training/g1/bones_to_g1.py",
    "deployment/export_bundle.py", "deployment/validate_bundle.py",
    "deployment/rollout_mujoco.py", "deployment/rollout_viser.py",
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_help(script):
    done = subprocess.run([sys.executable, str(REPO / script), "--help"], cwd=REPO,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


def test_all_python_compiles():
    import compileall

    for folder in ("lmj_real", "training", "deployment"):
        assert compileall.compile_dir(str(REPO / folder), quiet=1, force=True)
