"""Strategy plug-ins, by name. Add a strategy by writing a class that satisfies
research.strategy.Strategy and registering it here."""
from .sweep_fvg import SweepFvg

STRATEGIES = {SweepFvg.name: SweepFvg}


def get_strategy(name: str, **params):
    try:
        return STRATEGIES[name](**params)
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
