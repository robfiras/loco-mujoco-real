"""Domain randomizer registry."""


def register_domain_randomizers():
    from lmj_real.core.domain_randomizer.g1_real_randomizer import G1RealRandomizer

    G1RealRandomizer.register()
