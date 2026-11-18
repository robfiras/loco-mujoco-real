"""X2UltraReal environment on MJX / MjWarp."""

from __future__ import annotations

from lmj_real.environments.g1_real_mjx import MjxG1Real
from lmj_real.environments.x2_ultra_real import X2UltraReal


class MjxX2UltraReal(X2UltraReal, MjxG1Real):
    """Robot properties from ``X2UltraReal``, backend setup from ``MjxG1Real``.

    ``X2UltraReal`` defines no ``__init__`` and no ``mjx_enabled``, so both resolve to
    ``MjxG1Real`` through the MRO.
    """
