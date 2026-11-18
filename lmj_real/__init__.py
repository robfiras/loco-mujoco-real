"""LocoMuJoCo extension: train motion-tracking policies for the real Unitree G1."""


def register_lmj():
    """Register all lmj_real components (envs, rewards, observations, control, terminal)."""
    from lmj_real.core.control_functions import register_control_functions
    from lmj_real.core.domain_randomizer import register_domain_randomizers
    from lmj_real.core.observations import register_observations
    from lmj_real.core.reward import register_rewards
    from lmj_real.core.terminal_state_handler import register_terminal_state_handlers
    from lmj_real.environments import register_environments

    register_environments()
    register_control_functions()
    register_rewards()
    register_observations()
    register_terminal_state_handlers()
    register_domain_randomizers()


register_lmj()
