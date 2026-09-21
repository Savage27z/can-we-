"""§6.5 trade resolution on H1 closes, implemented ONCE.

The legacy per-setup evaluator (setup.evaluate_setup) and the research engine's
close-based exit both need "walk forward from the entry and stop at the first close that
invalidates the trade or reaches the target". Two copies of that loop could drift apart,
so both call this function.
"""
from typing import Optional, Sequence

from . import invalidation


def resolve_on_closes(direction: str, closes: Sequence[float], start: int,
                      invalidation_price: float,
                      target_price: float) -> Optional[tuple[int, str]]:
    """The first close at or after index `start` that ends the trade, as
    (index, "invalidation" | "target"), or None if none has by the end of the data.

    Invalidation (§3) is checked before the target, as in §6.5: a trade is a win only
    if the target is reached "before the invalidation condition triggers".
    """
    for j in range(start, len(closes)):
        close = closes[j]
        if invalidation.is_invalidated(direction, invalidation_price, close):
            return j, "invalidation"
        reached = close >= target_price if direction == "bullish" else close <= target_price
        if reached:
            return j, "target"
    return None


def resolve_on_plan(direction: str, opens: Sequence[float], highs: Sequence[float],
                    lows: Sequence[float], closes: Sequence[float], start: int,
                    invalidation_price: float, stop_price: float,
                    target_price: float) -> Optional[tuple[int, str, float]]:
    """How a trade ends if it is run the way the live trade plan tells a person to run it:
    a hard stop-loss and take-profit that fire the moment price touches them (wicks
    included), plus closing by hand when an H1 candle CLOSES beyond the invalidation
    level. Returns (index, "stop" | "target" | "invalidation", fill_price), or None.

    Within one candle the stop is assumed to fill first when both levels are touched (the
    candle's internal order is unknown, so this is the pessimistic reading). A stop that
    price gaps through fills at the candle's open; a target fills at the target price.
    This is the same policy as research.exits.PlanExit, which a test holds it to.
    """
    bullish = direction == "bullish"
    for j in range(start, len(closes)):
        if bullish:
            if lows[j] <= stop_price:
                return j, "stop", float(min(opens[j], stop_price))
            if highs[j] >= target_price:
                return j, "target", float(target_price)
        else:
            if highs[j] >= stop_price:
                return j, "stop", float(max(opens[j], stop_price))
            if lows[j] <= target_price:
                return j, "target", float(target_price)
        if invalidation.is_invalidated(direction, invalidation_price, closes[j]):
            return j, "invalidation", float(closes[j])
    return None
