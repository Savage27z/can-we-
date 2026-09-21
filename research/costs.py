"""What trading a signal would have cost.

The candle data is midpoint prices, which nobody can trade at: you buy at the ask and sell
at the bid. The stored `spread` column (closing ask minus closing bid) lets a backtest
charge for that.

SpreadCost charges one full spread per round trip, taken at the entry candle's close (the
moment the trade is placed), as if entering at the ask and leaving at the bid. That is
slightly pessimistic for a target order (a limit fills at its price) and slightly
optimistic for a stop that slips, and it ignores that spreads widen in fast markets, so
treat it as a realistic floor rather than an upper bound. `slippage_pips` adds a flat
extra allowance on top.

The cost is converted into R (cost / the trade's risk in price) and subtracted from the
trade's result, so a tight stop makes the same spread cost more R than a wide one.
"""
import math
from dataclasses import dataclass


class MissingSpreadData(RuntimeError):
    pass


@dataclass(frozen=True)
class NoCost:
    name: str = "none"
    uses_spread: bool = False

    def price(self, spread: float, typical_spread: float, pip_size: float) -> float:
        return 0.0


@dataclass(frozen=True)
class SpreadCost:
    multiplier: float = 1.0
    slippage_pips: float = 0.0
    name: str = "spread"
    uses_spread: bool = True

    def price(self, spread: float, typical_spread: float, pip_size: float) -> float:
        """Round-trip cost in price units. A missing spread at the entry candle falls back
        to the instrument's typical (median) spread; with no spread data at all this
        raises rather than quietly charging nothing."""
        if not math.isnan(spread):
            paid = spread
        elif not math.isnan(typical_spread):
            paid = typical_spread
        else:
            raise MissingSpreadData(
                "no spread data for this instrument, so trading costs cannot be applied; "
                "re-fetch it with python -m data_pipeline.fetch_historical (candles fetched "
                "before spread capture lack it) or run with costs 'none'")
        return self.multiplier * paid + self.slippage_pips * pip_size


def get_cost(name: str, slippage_pips: float = 0.0, multiplier: float = 1.0):
    if name == "none":
        return NoCost()
    if name == "spread":
        return SpreadCost(multiplier=multiplier, slippage_pips=slippage_pips)
    raise ValueError(f"unknown cost model {name!r}; choose 'none' or 'spread'")
