"""Numeric parameters from strategy_rules.md, in one place so the §9 'flagged for
post-backtest revision' parameters are easy to find and tune without hunting through
the engine code.
"""

# §1.5 / §2.3 window-counting rule: plain array-index distance, inclusive.
FVG_FORMATION_WINDOW_H4 = 10       # §1.5
CONFIRMATION_WINDOW_H1 = 20        # §2.3

# §2.4 liquidity/lookback scope, anchored to the sweep candle.
LIQUIDITY_LOOKBACK_H4 = 90

# §3 stop buffer beyond the sweep extreme.
STOP_BUFFER_PIPS_STANDARD = 5
STOP_BUFFER_PIPS_JPY = 5  # same pip count; pip size differs (see pip_size())

# §6.4 minimum acceptable reward:risk at confirmation.
MIN_RR = 1.5

# §5 session windows, UTC.
LONDON_SESSION_UTC = (7, 16)   # [07:00, 16:00)
NEWYORK_SESSION_UTC = (12, 21)  # [12:00, 21:00)

# §4 daily bias lookback (number of closed daily candles compared).
DAILY_BIAS_CANDLES = 3


def pip_size(instrument: str) -> float:
    """Price-per-pip for the instrument. JPY pairs quote 2 decimal places (1 pip =
    0.01); everything else in this strategy's scope quotes 4 decimal places
    (1 pip = 0.0001), per §3's "0.05 for JPY pairs, 0.0005 for others" (5 pips each).
    """
    return 0.01 if instrument.endswith("_JPY") else 0.0001


def stop_buffer(instrument: str) -> float:
    pips = STOP_BUFFER_PIPS_JPY if instrument.endswith("_JPY") else STOP_BUFFER_PIPS_STANDARD
    return pips * pip_size(instrument)
