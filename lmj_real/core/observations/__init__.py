"""Observation function registry."""


def register_observations():
    from lmj_real.core.observations.g1_real_obs import (
        G1RealBodyAngVelRel,
        G1RealBodyLinVelRel,
        G1RealBodyOriRel,
        G1RealBodyPosRel,
        G1RealJointPosRel,
    )
    from lmj_real.core.observations.goals_anchor import GoalTrajMimicAnchor

    GoalTrajMimicAnchor.register()
    G1RealJointPosRel.register()
    G1RealBodyPosRel.register()
    G1RealBodyOriRel.register()
    G1RealBodyLinVelRel.register()
    G1RealBodyAngVelRel.register()
