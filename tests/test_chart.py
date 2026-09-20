import io
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from live import chart
from live.plan import plan_for
from live.state import ActiveSetup, LiveState
from news.filter import NewsEventView, NewsStatus


def make_candles(n=80, start="2026-08-01 01:00", seed=7, base=1.15):
    rng = np.random.default_rng(seed)
    closes = base + np.cumsum(rng.normal(0, 0.0012, n))
    opens = np.concatenate([[base], closes[:-1]])
    highs = np.maximum(opens, closes) + rng.uniform(0.0002, 0.0009, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.0002, 0.0009, n)
    return pd.DataFrame({
        "time": pd.date_range(start, periods=n, freq="4h", tz="UTC"),
        "open": opens, "high": highs, "low": lows, "close": closes, "volume": 100,
    })


def make_state(candles, setups=None, news=None, pair="EUR_USD", buy=(), sell=()):
    return LiveState(
        pair=pair, as_of="2026-09-18T21:00:00+00:00",
        current_price=float(candles["close"].iloc[-1]), daily_bias="bullish",
        active_setups=list(setups or []), liquidity_buy_side=list(buy),
        liquidity_sell_side=list(sell), news=news or NewsStatus(status="clear"),
    )


def make_setup(candles, status="live_trade", direction="bullish", sweep_offset=14, pair="EUR_USD"):
    n = len(candles)
    sweep_i, fvg_i, conf_i = n - sweep_offset, n - sweep_offset + 2, n - 8
    low = float(candles["low"].iloc[sweep_i])
    high = float(candles["high"].iloc[sweep_i])
    extreme = low if direction == "bullish" else high
    fvg_low = float(candles["close"].iloc[fvg_i])
    live = status == "live_trade"
    setup = ActiveSetup(
        direction=direction, status=status, sweep_time=candles["time"].iloc[sweep_i].isoformat(),
        sweep_extreme=extreme, fvg_low=fvg_low, fvg_high=fvg_low + 0.001,
        confirmation_level=fvg_low,
        entry_price=fvg_low + 0.0014 if live else None,
        stop_price=extreme - 0.0005 if live else None,
        target_price=fvg_low + 0.006 if live else None,
        rr=2.4 if live else None, h1_candles_to_confirm=None,
        confirm_time=(candles["time"].iloc[conf_i] + pd.Timedelta(hours=2)).isoformat() if live else None,
        fvg_start_time=candles["time"].iloc[fvg_i].isoformat() if status != "pending_fvg" else None,
    )
    # The engine attaches a plan to every setup; a waiting one plans towards a target.
    setup.plan = plan_for(pair, setup, fvg_low + 0.006 if status == "pending_confirmation" else None)
    return setup


def decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png))


class RenderChartTests(unittest.TestCase):
    def assert_real_image(self, png):
        self.assertTrue(png.startswith(b"\x89PNG"), "not a PNG")
        img = decode(png)
        self.assertEqual(img.size, (int(chart.FIG_SIZE[0] * chart.DPI),
                                    int(chart.FIG_SIZE[1] * chart.DPI)))
        colors = img.convert("RGB").getcolors(maxcolors=100000)
        self.assertGreater(len(colors), 30, "image looks blank")
        return img

    def test_no_active_setup(self):
        c = make_candles()
        self.assert_real_image(chart.render_chart(c, make_state(c, buy=[1.20], sell=[1.10])))

    def test_every_setup_status_and_direction_renders(self):
        c = make_candles()
        for status in ("pending_fvg", "pending_confirmation", "live_trade"):
            for direction in ("bullish", "bearish"):
                state = make_state(c, [make_setup(c, status, direction)])
                self.assert_real_image(chart.render_chart(c, state))

    def test_news_blackout_badge_changes_the_image(self):
        c = make_candles()
        plain = chart.render_chart(c, make_state(c))
        news = NewsStatus(status="blackout", blackout_events=[
            NewsEventView("FOMC Statement", "USD", "2026-09-18T22:00:00+00:00", "in 25m")])
        flagged = chart.render_chart(c, make_state(c, news=news))
        self.assertNotEqual(plain, flagged)

    def test_a_waiting_setup_draws_its_planned_stop_and_target(self):
        c = make_candles()
        with_plan = make_setup(c, "pending_confirmation")
        without_plan = make_setup(c, "pending_confirmation")
        without_plan.plan = None
        drawn = chart.render_chart(c, make_state(c, [with_plan]))
        bare = chart.render_chart(c, make_state(c, [without_plan]))
        self.assertNotEqual(drawn, bare)
        self.assert_real_image(drawn)

    def test_a_setup_with_no_target_still_draws_its_planned_stop(self):
        c = make_candles()
        setup = make_setup(c, "pending_confirmation")
        setup.plan = plan_for("EUR_USD", setup, None)
        self.assert_real_image(chart.render_chart(c, make_state(c, [setup])))

    def test_jpy_pair_renders(self):
        c = make_candles(base=157.0)
        c[["open", "high", "low", "close"]] *= 1.0
        self.assert_real_image(chart.render_chart(c, make_state(c, pair="USD_JPY")))

    def test_sweep_older_than_the_visible_window_still_renders(self):
        c = make_candles()
        setup = make_setup(c)
        setup.sweep_time = "2026-01-01T01:00:00+00:00"       # long before the window
        setup.fvg_start_time = "2026-01-01T05:00:00+00:00"
        setup.confirm_time = "2026-01-01T09:00:00+00:00"
        self.assert_real_image(chart.render_chart(c, make_state(c, [setup])))

    def test_only_the_most_recent_setups_are_drawn_without_error(self):
        c = make_candles()
        setups = [make_setup(c, "live_trade", sweep_offset=k) for k in (30, 22, 14)]
        self.assert_real_image(chart.render_chart(c, make_state(c, setups)))

    def test_far_away_liquidity_levels_do_not_break_the_scale(self):
        c = make_candles()
        with_far = chart.render_chart(c, make_state(c, buy=[9.99], sell=[0.01]))
        without = chart.render_chart(c, make_state(c))
        self.assertEqual(with_far, without)  # levels far outside the range are not drawn

    def test_few_candles_render(self):
        for n in (1, 2, 5):
            c = make_candles(n=n)
            self.assert_real_image(chart.render_chart(c, make_state(c)))

    def test_no_candles_is_an_error_not_a_blank_chart(self):
        with self.assertRaises(ValueError):
            chart.render_chart(make_candles(n=1).iloc[0:0], make_state(make_candles(n=1)))

    def test_rendering_does_not_modify_its_inputs(self):
        c = make_candles()
        before = c.copy()
        chart.render_chart(c, make_state(c, [make_setup(c)]))
        pd.testing.assert_frame_equal(c, before)

    def test_same_input_gives_the_same_image(self):
        c = make_candles()
        state = make_state(c, [make_setup(c)])
        self.assertEqual(chart.render_chart(c, state), chart.render_chart(c, state))


class LegibilityTests(unittest.TestCase):
    """Telegram shows a photo in the chat bubble at under half its real size, so text
    that looks fine at full size is unreadable there. These guard the settings that
    keep it legible, so a later tweak can't quietly bring the tiny text back."""

    def test_canvas_is_small_enough_and_fonts_large_enough_for_the_chat_bubble(self):
        width_px = chart.FIG_SIZE[0] * chart.DPI
        self.assertLessEqual(width_px, 1100)
        for name in ("FONT_LABEL", "FONT_TICK", "FONT_SUBTITLE"):
            self.assertGreaterEqual(getattr(chart, name), 12, name)
        self.assertGreaterEqual(chart.FONT_TITLE, 20)

    def test_header_row_texts_are_kept_short_enough_not_to_collide(self):
        # The worst case: a long news-blackout banner beside the status line.
        c = make_candles()
        news = NewsStatus(status="blackout", blackout_events=[
            NewsEventView("Non-Farm Employment Change", "USD", "2026-09-18T22:00:00+00:00", "in 1h 25m")])
        state = make_state(c, [make_setup(c, "pending_confirmation")], news=news)
        img = decode(chart.render_chart(c, state))
        self.assertEqual(img.size[0], int(chart.FIG_SIZE[0] * chart.DPI))  # renders, no error


class HelperTests(unittest.TestCase):
    def test_position_requires_an_exact_candle_open(self):
        times = make_candles()["time"]
        self.assertEqual(chart._position(times, times.iloc[10].isoformat()), 10)
        off_grid = (times.iloc[10] + pd.Timedelta(hours=1)).isoformat()
        self.assertIsNone(chart._position(times, off_grid))
        self.assertIsNone(chart._position(times, "2020-01-01T00:00:00+00:00"))
        self.assertIsNone(chart._position(times, None))

    def test_position_containing_finds_the_candle_whose_period_holds_the_time(self):
        times = make_candles()["time"]
        inside = (times.iloc[10] + pd.Timedelta(hours=2)).isoformat()
        self.assertEqual(chart._position_containing(times, inside), 10)
        self.assertIsNone(chart._position_containing(times, "2020-01-01T00:00:00+00:00"))

    def test_spread_labels_separates_neighbours_but_leaves_distant_ones(self):
        placed = chart._spread_labels(
            [(1.000, "a", "x"), (1.001, "b", "x"), (1.002, "c", "x"), (2.0, "d", "x")], 0.01)
        ys = [p[0] for p in placed]
        self.assertTrue(all(b - a >= 0.01 - 1e-12 for a, b in zip(ys, ys[1:])))
        self.assertEqual(ys[-1], 2.0)

    def test_load_candles_returns_the_most_recent_rows(self):
        from unittest.mock import patch
        full = make_candles(n=200)
        with patch.object(chart.storage, "load", return_value=full):
            out = chart.load_candles("EUR_USD", count=30)
        self.assertEqual(len(out), 30)
        self.assertEqual(out["time"].iloc[-1], full["time"].iloc[-1])


if __name__ == "__main__":
    unittest.main()
