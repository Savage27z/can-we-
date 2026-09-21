"""Optional second chart renderer: the same picture as live/chart.py, drawn with TradingView's
open-source Lightweight Charts library in a headless browser.

Off by default (tgbot.config.CHART_RENDERER = "matplotlib"). It needs Playwright and a
Chromium/Chrome build, neither of which is in the production image, so enabling it there is a
deliberate step (see requirements-tv.txt and live/vendor/README.md). Any failure here falls back to the
matplotlib chart, so the report is never lost to a browser problem.

This module decides nothing and invents nothing, like chart.py: it turns the already-computed
LiveState plus the stored candles into a declarative spec (candles, lines, boxes, markers,
labels) and hands that to live/chart_tv.html, which only draws it. The rules for which levels
appear (nearest liquidity, the latest setups, the plan levels) mirror chart.py's.

Charts by TradingView Lightweight Charts(TM), Apache-2.0: see live/vendor/.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

from .chart import (AXIS, BG, GRID, MIN_ZONE_FOR_TEXT, MUTED, PLAN_ALPHA, POSITION_ALPHA, _zone_text, BIAS_COLORS, BUY_SIDE, DOWN, MAX_LIQUIDITY_LINES, MAX_SETUPS_DRAWN,
                    RIGHT_PADDING_SLOTS, SELL_SIDE, SWEEP, TEXT, UP, _decimals, _position,
                    _position_containing, _status_line)
from .state import ActiveSetup, LiveState

log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "chart_tv.html"
LIBRARY = HERE / "vendor" / "lightweight-charts.standalone.production.js"

WIDTH, HEIGHT = 1000, 620          # same canvas as the matplotlib chart
RENDER_TIMEOUT_MS = 20_000
_browser_lock = threading.Lock()   # one Chromium at a time: they are memory-hungry


def _unix(value) -> int:
    return int(pd.Timestamp(value).timestamp())


def _level_spec(price: float, color: str, style: str, width: float, text: Optional[str],
                lines: list, labels: list) -> None:
    lines.append({"price": float(price), "color": color, "style": style, "width": width,
                  "tag": False})
    if text:
        labels.append({"price": float(price), "text": text, "color": color})


def build_spec(candles: pd.DataFrame, state: LiveState) -> dict:
    if candles.empty:
        raise ValueError("no candles to draw")

    n = len(candles)
    times = candles["time"]
    opens = candles["open"].to_numpy(dtype=float)
    highs = candles["high"].to_numpy(dtype=float)
    lows = candles["low"].to_numpy(dtype=float)
    closes = candles["close"].to_numpy(dtype=float)
    decimals = _decimals(state.pair)

    def fmt(v: float) -> str:
        return format(v, f".{decimals}f")

    setups = state.active_setups[-MAX_SETUPS_DRAWN:]

    # The price range: candles, every drawn setup level, near liquidity only (a far-away
    # level would squash the chart), and room for the sweep marker: as in chart.py.
    lo, hi = float(lows.min()), float(highs.max())
    span = max(hi - lo, 1e-9)
    ys = [lo, hi, state.current_price]
    setup_levels: list[float] = []
    for s in setups:
        setup_levels += [v for v in (s.sweep_extreme, s.fvg_low, s.fvg_high, s.confirmation_level,
                                     s.entry_price, s.stop_price, s.target_price) if v is not None]
        if s.status == "pending_confirmation" and s.plan is not None:
            setup_levels += [v for v in (s.plan.stop, s.plan.target) if v is not None]
        ys.append(s.sweep_extreme - span * 0.12 if s.direction == "bullish"
                  else s.sweep_extreme + span * 0.12)
    ys += setup_levels

    def not_a_setup_level(level: float) -> bool:
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

    lines: list[dict] = []
    labels: list[dict] = []
    boxes: list[dict] = []
    markers: list[dict] = []

    for level in buy_levels:
        _level_spec(level, BUY_SIDE, "dotted", 1, f"BSL {fmt(level)}", lines, labels)
    for level in sell_levels:
        _level_spec(level, SELL_SIDE, "dotted", 1, f"SSL {fmt(level)}", lines, labels)
    for setup in setups:
        _setup_spec(setup, times, n, fmt, lines, labels, boxes, markers)

    # The current price: a coloured dotted line with its value tag on the price axis.
    price_color = UP if closes[-1] >= opens[-1] else DOWN
    lines.append({"price": float(state.current_price), "color": price_color, "style": "dotted",
                  "width": 1, "tag": True})

    last = candles.iloc[-1]
    return {
        "theme": {"bg": BG, "grid": GRID, "axis": AXIS, "text": TEXT, "muted": MUTED},
        "n": n, "pad": RIGHT_PADDING_SLOTS, "decimals": decimals,
        "y_lo": float(y_lo), "y_hi": float(y_hi),
        "candles": [{"time": _unix(t), "open": float(o), "high": float(h), "low": float(l),
                     "close": float(c)} for t, o, h, l, c in zip(times, opens, highs, lows, closes)],
        "lines": lines, "labels": labels, "boxes": boxes, "min_zone": MIN_ZONE_FOR_TEXT,
        "span": float(y_hi - y_lo),
        "markers": sorted(markers, key=lambda m: m["time"]),
        "header": {
            "title": f"{state.pair.replace('_', '')}  ·  4h  ·  OANDA",
            "status": _status_line(state),
            "legend": "Last closed 4h:  O {0}  H {1}  L {2}  C {3}".format(
                *(fmt(float(last[c])) for c in ("open", "high", "low", "close"))),
            "legend_color": price_color,
            "bias": f"DAILY BIAS: {state.daily_bias.upper()}",
            "bias_color": BIAS_COLORS.get(state.daily_bias, BIAS_COLORS["neutral"]),
            "news": (f"NEWS BLACKOUT · {state.news.blackout_events[0].currency} "
                     f"{state.news.blackout_events[0].title} · {state.news.blackout_events[0].when}"
                     if state.news.status == "blackout" and state.news.blackout_events else ""),
            "as_of": f"as of {state.as_of[:16].replace('T', ' ')} UTC",
        },
    }


def _setup_spec(setup: ActiveSetup, times: pd.Series, n: int, fmt, lines: list, labels: list,
                boxes: list, markers: list) -> None:
    bullish = setup.direction == "bullish"
    sweep_pos = _position(times, setup.sweep_time)

    _level_spec(setup.sweep_extreme, SWEEP, "dashed", 1.5,
                f"Sweep {fmt(setup.sweep_extreme)}", lines, labels)
    if sweep_pos is not None:
        markers.append({"time": _unix(times.iloc[sweep_pos]),
                        "position": "belowBar" if bullish else "aboveBar",
                        "shape": "arrowUp" if bullish else "arrowDown",
                        "color": SWEEP, "text": "Sweep"})

    fvg_pos = _position(times, setup.fvg_start_time)
    if setup.fvg_low is not None and setup.fvg_high is not None:
        start = fvg_pos if fvg_pos is not None else sweep_pos
        boxes.append({"i0": start,
                      "low": float(setup.fvg_low), "high": float(setup.fvg_high),
                      "color": UP if bullish else DOWN, "alpha": 0.25, "label": "FVG"})

    if setup.status == "pending_confirmation" and setup.confirmation_level is not None:
        sign = ">" if bullish else "<"
        _level_spec(setup.confirmation_level, TEXT, "dotted", 1.5,
                    f"Confirm {sign} {fmt(setup.confirmation_level)}", lines, labels)
        plan = setup.plan
        if plan is not None:
            _level_spec(plan.stop, DOWN, "dashed", 1.5, f"Plan stop {fmt(plan.stop)}", lines, labels)
            entry = plan.entry if plan.entry is not None else setup.confirmation_level
            boxes.append({"i0": n - 1, "edge": True, "low": min(entry, plan.stop),
                          "high": max(entry, plan.stop), "color": DOWN, "alpha": PLAN_ALPHA,
                          "label": "", "note": _zone_text(plan, "stop", True), "dashed": True})
            if plan.target is not None:
                reward = f" (+{plan.rr:.1f}R)" if plan.rr is not None else ""
                _level_spec(plan.target, UP, "dashed", 1.5,
                            f"Plan TP {fmt(plan.target)}{reward}", lines, labels)
                boxes.append({"i0": n - 1, "edge": True, "low": min(entry, plan.target),
                              "high": max(entry, plan.target), "color": UP, "alpha": PLAN_ALPHA,
                              "label": "", "note": _zone_text(plan, "target", True),
                              "dashed": True})

    if setup.status == "live_trade" and setup.entry_price is not None:
        confirm_pos = _position_containing(times, setup.confirm_time)
        t0 = confirm_pos
        if setup.stop_price is not None:
            boxes.append({"i0": t0, "edge": True, "low": min(setup.entry_price, setup.stop_price),
                          "high": max(setup.entry_price, setup.stop_price),
                          "color": DOWN, "alpha": POSITION_ALPHA, "label": "",
                          "note": _zone_text(setup.plan, "stop", False)})
            _level_spec(setup.stop_price, DOWN, "solid", 2, f"Stop {fmt(setup.stop_price)}",
                        lines, labels)
        if setup.target_price is not None:
            boxes.append({"i0": t0, "edge": True, "low": min(setup.entry_price, setup.target_price),
                          "high": max(setup.entry_price, setup.target_price),
                          "color": UP, "alpha": POSITION_ALPHA, "label": "",
                          "note": _zone_text(setup.plan, "target", False)})
            reward = f" (+{setup.rr:.1f}R)" if setup.rr is not None else ""
            _level_spec(setup.target_price, UP, "solid", 2,
                        f"TP {fmt(setup.target_price)}{reward}", lines, labels)
        _level_spec(setup.entry_price, TEXT, "solid", 2, f"Entry {fmt(setup.entry_price)}",
                    lines, labels)


def build_html(spec: dict) -> str:
    """The page with the library and the spec inlined, so it needs no network."""
    page = TEMPLATE.read_text(encoding="utf-8")
    library = LIBRARY.read_text(encoding="utf-8")
    # A literal "</script>" inside the data would end the tag early.
    payload = json.dumps(spec).replace("</", "<\\/")
    return (page.replace("/*__LIB__*/", library, 1)
                .replace("/*__SPEC__*/null", payload, 1))


def _launch(playwright):
    channel = os.environ.get("CHART_BROWSER_CHANNEL") or None
    return playwright.chromium.launch(channel=channel, headless=True,
                                      args=["--no-sandbox", "--disable-gpu",
                                            "--disable-dev-shm-usage"])


def render_chart_tv(candles: pd.DataFrame, state: LiveState) -> bytes:
    from playwright.sync_api import sync_playwright   # optional dependency: imported lazily

    html = build_html(build_spec(candles, state))
    with _browser_lock, sync_playwright() as playwright:
        browser = _launch(playwright)
        try:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT},
                                    device_scale_factor=1)
            page.set_content(html, wait_until="load", timeout=RENDER_TIMEOUT_MS)
            page.wait_for_function("window.__ready === true", timeout=RENDER_TIMEOUT_MS)
            return page.screenshot(type="png", clip={"x": 0, "y": 0, "width": WIDTH,
                                                     "height": HEIGHT})
        finally:
            browser.close()
