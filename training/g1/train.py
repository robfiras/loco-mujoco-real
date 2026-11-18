"""Train a motion-tracking policy for the Unitree G1 (`G1Real`).

::

    python training/g1/train.py
    python training/g1/train.py experiment.num_envs=16384

The settings are in ``conf.yaml`` and ``env/`` next to this file, and any of them can be
overridden on the command line. The training loop itself is shared: ``../ppo.py``.
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo import experiment  # noqa: E402


@hydra.main(version_base=None, config_path="./", config_name="conf")
def main(config: DictConfig):
    experiment(config)


if __name__ == "__main__":
    main()
