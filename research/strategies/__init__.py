"""Strategy plug-ins, by name. Add a strategy by writing a class that satisfies
research.strategy.Strategy and registering it here."""
from .daily_reversion import DailyReversion
from .donchian_trend import DonchianTrend
from .random_control import RandomControl
from .range_breakout import RangeBreakout
from .sweep_fvg import SweepFvg, SweepFvgNoRollover

STRATEGIES = {cls.name: cls for cls in (SweepFvg, SweepFvgNoRollover, DonchianTrend,
                                        RangeBreakout, DailyReversion, RandomControl)}


def get_strategy(name: str, **params):
    try:
        cls = STRATEGIES[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
    return cls(**params)
