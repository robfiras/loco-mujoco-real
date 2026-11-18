"""Terminal state handler registry."""


def register_terminal_state_handlers():
    from lmj_real.core.terminal_state_handler.g1_real_tracking import (
        G1RealTrackingTerminalStateHandler,
    )

    G1RealTrackingTerminalStateHandler.register()
