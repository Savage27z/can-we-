"""§3 invalidation condition, implemented ONCE.

Per §3's "Implementation note (single source of truth)": this check is used in
exactly two places (§2.3 pre-confirmation setup expiry, and §3 post-confirmation
trade close) and must never be duplicated — both call this function.
"""


def is_invalidated(direction: str, sweep_extreme: float, close_price: float) -> bool:
    if direction == "bullish":
        return close_price < sweep_extreme
    if direction == "bearish":
        return close_price > sweep_extreme
    raise ValueError(f"direction must be 'bullish' or 'bearish', got {direction!r}")
