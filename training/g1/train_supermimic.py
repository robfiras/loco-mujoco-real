"""Train a SuperMimic motion-tracking policy for the Unitree G1 (`G1Real`).

::

    python training/g1/train_supermimic.py
    python training/g1/train_supermimic.py experiment.seed=1

The settings are in ``supermimic.yaml`` and ``env/`` next to this file, and any of them can be
overridden on the command line. The training loop itself is shared: ``../supermimic.py``.
"""
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic import experiment  # noqa: E402


@hydra.main(version_base=None, config_path="./", config_name="supermimic")
def main(config: DictConfig):
    experiment(config)


if __name__ == "__main__":
    main()
