"""Reward function registry."""


def register_rewards():
    from lmj_real.core.reward.g1_real_tracking_reward import G1RealTrackingReward

    G1RealTrackingReward.register()
