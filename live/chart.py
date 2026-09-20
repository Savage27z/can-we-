"""Renders the H4 chart sent above /analysis reports and alerts.

Drawn purely from the already-computed LiveState plus the stored candles: it decides
nothing and invents nothing, it only shows what the engine found (sweep, FVG,
confirmation level, entry/stop/target, liquidity levels). Uses matplotlib's
object-oriented Figure API rather than pyplot, which keeps no global state and is
safe to call from worker threads.

Sizing: Telegram shows a photo in the chat bubble at well under half its real size,
so the canvas is kept small (1000x620) and the fonts large; text that reads fine on a
1280px canvas is unreadable in the bubble.
"""
import io
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from data_pipeline import storage

from .state import ActiveSetup, LiveState

CANDLES_SHOWN = 60
RIGHT_PADDING_SLOTS = 18      # empty space right of the last candle, for price labels
MAX_SETUPS_DRAWN = 2          # the most recent few; more would just clutter the chart
MAX_LIQUIDITY_LINES = 3       # per side, nearest to price first
FIG_SIZE = (10, 6.2)
DPI = 100                     # -> 1000x620 pixels

FONT_TITLE = 24
FONT_SUBTITLE = 13.5
FONT_BADGE = 14
FONT_TICK = 12
FONT_LABEL = 13
FONT_NOTE = 10

BG = "#0d1117"
GRID = "#1c2330"
TEXT = "#c9d1d9"
MUTED = "#8b949e"
UP = "#26a69a"
DOWN = "#ef5350"
SWEEP = "#e3b341"
BUY_SIDE = "#d29922"
SELL_SIDE = "#58a6ff"
BIAS_COLORS = {"bullish": "#238636", "bearish": "#da3633", "neutral": "#6e7681"}
STATUS_LABELS = {
    "pending_fvg": "awaiting FVG",
    "pending_confirmation": "awaiting confirmation",
    "live_trade": "live trade",
}


def load_candles(pair: str, count: int = CANDLES_SHOWN) -> pd.DataFrame:
    return storage.load(pair, "H4").tail(count).reset_index(drop=True)


def _decimals(pair: str) -> int:
    return 3 if pair.endswith("_JPY") else 5


def _position(times: pd.Series, iso: Optional[str]) -> Optional[int]:
    """Index of the candle that OPENS at `iso`, or None if it isn't in the window."""
    if not iso:
        return None
    ts = pd.Timestamp(iso)
    pos = int(times.searchsorted(ts, side="left"))
    if pos < len(times) and times.iloc[pos] == ts:
        return pos
    return None


def _position_containing(times: pd.Series, iso: Optional[str]) -> Optional[int]:
    """Index of the candle whose period contains `iso` (last candle opening at or
    before it), or None if it falls before the window."""
    if not iso:
        return None
    pos = int(times.searchsorted(pd.Timestamp(iso), side="right")) - 1
    return pos if pos >= 0 else None


def _spread_labels(labels: list[tuple[float, str, str]], min_gap: float):
    """Nudges label heights apart so neighbouring price labels don't overprint.
    Input/output: (y, text, color), sorted by y."""
    placed: list[tuple[float, str, str]] = []
    for y, text, color in sorted(labels):
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, text, color))
    return placed


def _status_line(state: LiveState) -> str:
    if not state.active_setups:
        return "No active setup"
    setup = state.active_setups[-1]
    return f"{setup.direction.title()} setup: {STATUS_LABELS.get(setup.status, setup.status)}"


def render_chart(candles: pd.DataFrame, state: LiveState) -> bytes:
    if candles.empty:
        raise ValueError("no candles to draw")

    n = len(candles)
    times = candles["time"]
    opens = candles["open"].to_numpy(dtype=float)
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    closes = candles["close"].to_numpy(dtype=float)
    fmt = f".{_decimals(state.pair)}f"
    right_edge = n - 1 + RIGHT_PADDING_SLOTS
    line_end = n - 0.4          # where horizontal lines stop, just past the last candle
    label_x = n + 0.6

    setups = state.active_setups[-MAX_SETUPS_DRAWN:]

    # Y range: the candles, plus every level of a drawn setup, plus liquidity levels
    # only when they're near the candles (a far-away level would squash the chart).
    lo, hi = float(lows.min()), float(highs.max())
    span = max(hi - lo, 1e-9)
    ys = [lo, hi, state.current_price]
    setup_levels: list[float] = []
    for s in setups:
        setup_levels += [v for v in (s.sweep_extreme, s.fvg_low, s.fvg_high, s.confirmation_level,
                                     s.entry_price, s.stop_price, s.target_price) if v is not None]
        if s.status == "pending_confirmation" and s.plan is not None:
            setup_levels += [v for v in (s.plan.stop, s.plan.target) if v is not None]
        # Room for the sweep marker and its caption beyond the extreme.
        ys.append(s.sweep_extreme - span * 0.12 if s.direction == "bullish"
                  else s.sweep_extreme + span * 0.12)
    ys += setup_levels

    def not_a_setup_level(level: float) -> bool:
        # A liquidity level at the same price as a setup level would just print a
        # second, overlapping label for the same line.
        return all(abs(level - v) > span * 0.004 for v in setup_levels)

    near = (lo - 0.25 * span, hi + 0.25 * span)
    buy_levels = [v for v in state.liquidity_buy_side
                  if near[0] <= v <= near[1] and not_a_setup_level(v)][:MAX_LIQUIDITY_LINES]
    sell_levels = [v for v in state.liquidity_sell_side
                   if near[0] <= v <= near[1] and not_a_setup_level(v)][:MAX_LIQUIDITY_LINES]
    ys += buy_levels + sell_levels
    y_lo, y_hi = min(ys), max(ys)
    pad = (y_hi - y_lo) * 0.05
    y_lo, y_hi = y_lo - pad, y_hi + pad

    fig = Figure(figsize=FIG_SIZE, dpi=DPI, facecolor=BG)
    ax = fig.add_axes([0.03, 0.10, 0.86, 0.73])  # right margin holds the price ticks
    ax.set_facecolor(BG)
    ax.set_xlim(-1, right_edge)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=FONT_TICK)
    ax.yaxis.tick_right()
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    # Candles
    x = np.arange(n)
    colors = np.where(closes >= opens, UP, DOWN)
    ax.vlines(x, lows, highs, colors=colors, linewidth=1.3, zorder=3)
    body = np.maximum(np.abs(closes - opens), (y_hi - y_lo) * 0.0006)
    ax.bar(x, body, bottom=np.minimum(opens, closes), width=0.66, color=colors, zorder=4)

    tick_positions = list(range(0, n, 12))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{times.iloc[i]:%b %d}" for i in tick_positions])
    ax.yaxis.set_major_formatter(lambda v, _: format(v, fmt))

    labels: list[tuple[float, str, str]] = []

    # Liquidity levels
    for level in buy_levels:
        ax.hlines(level, max(0, n - 30), line_end, colors=BUY_SIDE, linestyles=":",
                  linewidth=1.4, alpha=0.9, zorder=2)
        labels.append((level, f"BSL {format(level, fmt)}", BUY_SIDE))
    for level in sell_levels:
        ax.hlines(level, max(0, n - 30), line_end, colors=SELL_SIDE, linestyles=":",
                  linewidth=1.4, alpha=0.9, zorder=2)
        labels.append((level, f"SSL {format(level, fmt)}", SELL_SIDE))

    # Setups
    for setup in setups:
        _draw_setup(ax, setup, times, line_end, fmt, labels, y_hi - y_lo)

    # Current price
    ax.hlines(state.current_price, 0, line_end, colors=TEXT, linestyles="--",
              linewidth=1.0, alpha=0.6, zorder=2)
    price_color = UP if closes[-1] >= opens[-1] else DOWN
    # The price tag sits on the axis edge (x in axes units, y in price), over the tick
    # labels, so it can never collide with the level labels inside the plot.
    ax.text(1.005, state.current_price, format(state.current_price, fmt), color="white",
            fontsize=FONT_LABEL, fontweight="bold", va="center", ha="left", zorder=6,
            clip_on=False, transform=ax.get_yaxis_transform(),
            bbox={"facecolor": price_color, "edgecolor": "none", "pad": 3})

    for y, text, color in _spread_labels(labels, (y_hi - y_lo) * 0.055):
        ax.text(label_x, y, text, color=color, fontsize=FONT_LABEL, va="center", ha="left",
                zorder=5, clip_on=False)

    # Header
    pair_label = state.pair.replace("_", "")
    fig.text(0.03, 0.945, f"{pair_label}  ·  H4", color=TEXT, fontsize=FONT_TITLE,
             fontweight="bold", ha="left", va="center")
    fig.text(0.03, 0.885, _status_line(state), color=MUTED, fontsize=FONT_SUBTITLE,
             ha="left", va="center")
    bias = state.daily_bias
    fig.text(0.975, 0.945, f" DAILY BIAS: {bias.upper()} ", color="white", fontsize=FONT_BADGE,
             fontweight="bold", ha="right", va="center",
             bbox={"facecolor": BIAS_COLORS.get(bias, BIAS_COLORS["neutral"]),
                   "edgecolor": "none", "pad": 5})
    if state.news.status == "blackout" and state.news.blackout_events:
        event = state.news.blackout_events[0]
        fig.text(0.975, 0.885, f"NEWS BLACKOUT · {event.currency} {event.title} · {event.when}",
                 color="#f85149", fontsize=FONT_SUBTITLE - 1, fontweight="bold",
                 ha="right", va="center")
    fig.text(0.03, 0.018, f"as of {state.as_of[:16].replace('T', ' ')} UTC", color=MUTED,
             fontsize=FONT_NOTE, ha="left", va="center")
    fig.text(0.975, 0.018, "Structural analysis, not financial advice", color=MUTED,
             fontsize=FONT_NOTE, ha="right", va="center")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    return buf.getvalue()


def _draw_setup(ax, setup: ActiveSetup, times: pd.Series, line_end: float,
                fmt: str, labels: list, y_span: float) -> None:
    bullish = setup.direction == "bullish"
    sweep_pos = _position(times, setup.sweep_time)
    start = sweep_pos if sweep_pos is not None else 0

    # The sweep: a marker on the sweeping candle and a line at its extreme.
    ax.hlines(setup.sweep_extreme, start, line_end, colors=SWEEP, linestyles="--",
              linewidth=1.5, zorder=3)
    labels.append((setup.sweep_extreme, f"Sweep {format(setup.sweep_extreme, fmt)}", SWEEP))
    if sweep_pos is not None:
        offset = y_span * 0.035
        y = setup.sweep_extreme - offset if bullish else setup.sweep_extreme + offset
        ax.plot([sweep_pos], [y], marker="^" if bullish else "v", color=SWEEP,
                markersize=12, zorder=6)
        ax.text(sweep_pos, y - offset * 0.9 if bullish else y + offset * 0.9, "Sweep",
                color=SWEEP, fontsize=FONT_LABEL, fontweight="bold", ha="center",
                va="top" if bullish else "bottom", zorder=6)

    # The fair value gap, from its first candle to the right edge of the price area.
    fvg_pos = _position(times, setup.fvg_start_time)
    if setup.fvg_low is not None and setup.fvg_high is not None:
        x0 = fvg_pos if fvg_pos is not None else start
        fill = UP if bullish else DOWN
        ax.add_patch(Rectangle((x0 - 0.4, setup.fvg_low), line_end - x0 + 0.4,
                               setup.fvg_high - setup.fvg_low, facecolor=fill,
                               edgecolor=fill, alpha=0.25, linewidth=1.0, zorder=1))
        ax.text(x0 - 0.3, setup.fvg_high, "FVG", color=fill, fontsize=FONT_TICK,
                fontweight="bold", va="bottom", ha="right", zorder=5)

    # Waiting for confirmation: the level a close must cross.
    if setup.status == "pending_confirmation" and setup.confirmation_level is not None:
        ax.hlines(setup.confirmation_level, fvg_pos if fvg_pos is not None else start, line_end,
                  colors="white", linestyles=":", linewidth=1.5, zorder=3)
        sign = ">" if bullish else "<"
        labels.append((setup.confirmation_level,
                       f"Confirm {sign} {format(setup.confirmation_level, fmt)}", "white"))
        # The planned trade, dashed so it can't be mistaken for one that is live.
        plan = setup.plan
        if plan is not None:
            # Spans the recent candles, like the liquidity lines, so the levels can be
            # read against price action rather than being a short stub at the edge.
            x0 = max(0, len(times) - 30)
            ax.hlines(plan.stop, x0, line_end, colors=DOWN, linestyles="--", linewidth=1.6,
                      alpha=0.85, zorder=3)
            labels.append((plan.stop, f"Plan stop {format(plan.stop, fmt)}", DOWN))
            if plan.target is not None:
                ax.hlines(plan.target, x0, line_end, colors=UP, linestyles="--",
                          linewidth=1.6, alpha=0.85, zorder=3)
                reward = f" (+{plan.rr:.1f}R)" if plan.rr is not None else ""
                labels.append((plan.target, f"Plan TP {format(plan.target, fmt)}{reward}", UP))

    # A confirmed trade: entry, stop, target and the risk/reward zones.
    if setup.status == "live_trade" and setup.entry_price is not None:
        confirm_pos = _position_containing(times, setup.confirm_time)
        x0 = confirm_pos if confirm_pos is not None else 0
        if setup.stop_price is not None:
            ax.add_patch(Rectangle((x0, min(setup.entry_price, setup.stop_price)), line_end - x0,
                                   abs(setup.entry_price - setup.stop_price), facecolor=DOWN,
                                   alpha=0.14, linewidth=0, zorder=1))
            ax.hlines(setup.stop_price, x0, line_end, colors=DOWN, linewidth=1.8, zorder=3)
            labels.append((setup.stop_price, f"Stop {format(setup.stop_price, fmt)}", DOWN))
        if setup.target_price is not None:
            ax.add_patch(Rectangle((x0, min(setup.entry_price, setup.target_price)), line_end - x0,
                                   abs(setup.target_price - setup.entry_price), facecolor=UP,
                                   alpha=0.14, linewidth=0, zorder=1))
            ax.hlines(setup.target_price, x0, line_end, colors=UP, linewidth=1.8, zorder=3)
            reward = f" (+{setup.rr:.1f}R)" if setup.rr is not None else ""
            labels.append((setup.target_price,
                           f"TP {format(setup.target_price, fmt)}{reward}", UP))
        ax.hlines(setup.entry_price, x0, line_end, colors=TEXT, linewidth=1.8, zorder=3)
        labels.append((setup.entry_price, f"Entry {format(setup.entry_price, fmt)}", TEXT))
