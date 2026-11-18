"""Environment registration."""


def register_environments():
    from lmj_real.environments.g1_real import G1Real
    from lmj_real.environments.g1_real_mjx import MjxG1Real
    from lmj_real.environments.x2_ultra_real import X2UltraReal
    from lmj_real.environments.x2_ultra_real_mjx import MjxX2UltraReal

    G1Real.register()
    MjxG1Real.register()
    X2UltraReal.register()
    MjxX2UltraReal.register()
