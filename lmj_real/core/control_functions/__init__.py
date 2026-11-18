"""Control function registry."""


def register_control_functions():
    from lmj_real.core.control_functions.g1_real_position import G1RealPositionControl

    G1RealPositionControl.register()
