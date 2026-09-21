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
