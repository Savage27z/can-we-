import io
import unittest
from unittest.mock import patch

import pandas as pd
from PIL import Image

from live import chart_tv
from news.filter import NewsEventView, NewsStatus
from tests.test_chart import make_candles, make_setup, make_state
from tgbot import config, service


def spec_for(setups=None, **kwargs):
    candles = make_candles()
    return candles, chart_tv.build_spec(candles, make_state(candles, setups, **kwargs))


class BuildSpecTests(unittest.TestCase):
    def test_every_candle_is_passed_through_with_unix_times(self):
        candles, spec = spec_for()
        self.assertEqual(len(spec["candles"]), len(candles))
        first = spec["candles"][0]
        self.assertEqual(first["time"], int(candles["time"].iloc[0].timestamp()))
        self.assertAlmostEqual(first["close"], float(candles["close"].iloc[0]))

    def test_the_current_price_is_a_tagged_line_coloured_by_the_last_candle(self):
        candles, spec = spec_for()
        tagged = [line for line in spec["lines"] if line["tag"]]
        self.assertEqual(len(tagged), 1)
        self.assertAlmostEqual(tagged[0]["price"], float(candles["close"].iloc[-1]))

    def test_a_live_trade_draws_entry_stop_target_and_both_risk_zones(self):
        candles = make_candles()
        setup = make_setup(candles, "live_trade")
        spec = chart_tv.build_spec(candles, make_state(candles, [setup]))
        prices = {round(line["price"], 6) for line in spec["lines"]}
        for level in (setup.entry_price, setup.stop_price, setup.target_price):
            self.assertIn(round(level, 6), prices)
        texts = " ".join(label["text"] for label in spec["labels"])
        for word in ("Entry", "Stop", "TP", "Sweep"):
            self.assertIn(word, texts)
        self.assertIn("(+2.4R)", texts)
        # the FVG plus the stop and target zones
        self.assertEqual(len(spec["boxes"]), 3)

    def test_a_waiting_setup_draws_its_planned_stop_and_target_not_a_live_trade(self):
        candles = make_candles()
        setup = make_setup(candles, "pending_confirmation")
        spec = chart_tv.build_spec(candles, make_state(candles, [setup]))
        texts = " ".join(label["text"] for label in spec["labels"])
        self.assertIn("Plan stop", texts)
        self.assertIn("Plan TP", texts)
        self.assertIn("Confirm >", texts)
        self.assertNotIn("Entry", texts)

    def test_the_sweep_is_marked_on_its_candle(self):
        candles = make_candles()
        setup = make_setup(candles, "live_trade", direction="bearish")
        spec = chart_tv.build_spec(candles, make_state(candles, [setup]))
        marker = spec["markers"][0]
        self.assertEqual(marker["position"], "aboveBar")
        self.assertEqual(marker["shape"], "arrowDown")
        self.assertIn(marker["time"], {c["time"] for c in spec["candles"]})

    def test_a_far_away_liquidity_level_is_left_off_and_does_not_stretch_the_scale(self):
        candles, spec = spec_for(buy=[1.16, 9.0], sell=[0.1])
        self.assertLess(spec["y_hi"], 2.0)
        self.assertGreater(spec["y_lo"], 0.5)
        self.assertNotIn(9.0, [line["price"] for line in spec["lines"]])

    def test_the_price_range_holds_every_drawn_level(self):
        candles = make_candles()
        setup = make_setup(candles, "live_trade")
        spec = chart_tv.build_spec(candles, make_state(candles, [setup]))
        for line in spec["lines"]:
            self.assertGreaterEqual(line["price"], spec["y_lo"])
            self.assertLessEqual(line["price"], spec["y_hi"])

    def test_jpy_pairs_use_three_decimals(self):
        candles = make_candles(base=150.0)
        spec = chart_tv.build_spec(candles, make_state(candles, pair="USD_JPY"))
        self.assertEqual(spec["decimals"], 3)
        self.assertIn("USDJPY", spec["header"]["title"])

    def test_the_header_carries_bias_status_and_a_news_blackout(self):
        news = NewsStatus(status="blackout", blackout_events=[
            NewsEventView("Non-Farm Employment Change", "USD", "2026-09-18T22:00:00+00:00", "in 1h")])
        candles, spec = spec_for(news=news)
        self.assertIn("BULLISH", spec["header"]["bias"])
        self.assertIn("NEWS BLACKOUT", spec["header"]["news"])
        self.assertTrue(spec["header"]["as_of"].startswith("as of 2026-09-18"))

    def test_no_candles_is_an_error_not_a_blank_chart(self):
        candles = make_candles()
        with self.assertRaises(ValueError):
            chart_tv.build_spec(candles.iloc[0:0], make_state(candles))

    def test_building_the_spec_does_not_modify_its_inputs(self):
        candles = make_candles()
        state = make_state(candles, [make_setup(candles, "live_trade")])
        before = candles.copy()
        chart_tv.build_spec(candles, state)
        pd.testing.assert_frame_equal(candles, before)


class BuildHtmlTests(unittest.TestCase):
    def test_the_page_is_self_contained_and_carries_the_data(self):
        candles, spec = spec_for()
        html = chart_tv.build_html(spec)
        self.assertIn("TradingView Lightweight Charts", html)       # the vendored library
        self.assertIn('"candles"', html)
        self.assertNotIn("__LIB__", html)
        self.assertNotIn("__SPEC__", html)
        self.assertNotIn("http://", html.split("<script>")[0])      # nothing to fetch

    def test_data_cannot_close_the_script_tag_early(self):
        candles, spec = spec_for()
        spec["header"]["status"] = "</script><b>x</b>"
        html = chart_tv.build_html(spec)
        self.assertNotIn("</script><b>", html)


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            chart_tv._launch(playwright).close()
        return True
    except Exception:
        return False


@unittest.skipUnless(_browser_available(),
                     "needs Playwright and a browser (set CHART_BROWSER_CHANNEL=chrome)")
class RenderInBrowserTests(unittest.TestCase):
    def test_renders_a_real_png_at_the_same_size_as_the_matplotlib_chart(self):
        candles = make_candles()
        state = make_state(candles, [make_setup(candles, "live_trade")])
        png = chart_tv.render_chart_tv(candles, state)
        self.assertTrue(png.startswith(b"\x89PNG"))
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.size, (chart_tv.WIDTH, chart_tv.HEIGHT))
        self.assertGreater(len(image.convert("RGB").getcolors(maxcolors=100000)), 30)


class RendererSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candles = make_candles()
        self.state = make_state(self.candles)
        for target, value in (("tgbot.service.load_candles", self.candles),):
            patcher = patch(target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_matplotlib_is_the_default_and_never_touches_the_browser_module(self):
        self.assertEqual(config.CHART_RENDERER, "matplotlib")
        with patch("tgbot.service.render_chart", return_value=b"MPL") as mpl, \
                patch.dict("sys.modules", {"live.chart_tv": None}):
            self.assertEqual(service._default_chart(self.state), b"MPL")
        mpl.assert_called_once()

    def test_the_tradingview_renderer_is_used_when_chosen(self):
        with patch.object(config, "CHART_RENDERER", "tradingview"), \
                patch("live.chart_tv.render_chart_tv", return_value=b"TV") as tv, \
                patch("tgbot.service.render_chart", return_value=b"MPL"):
            self.assertEqual(service._default_chart(self.state), b"TV")
        tv.assert_called_once()

    def test_a_browser_failure_falls_back_to_the_matplotlib_chart(self):
        with patch.object(config, "CHART_RENDERER", "tradingview"), \
                patch("live.chart_tv.render_chart_tv", side_effect=RuntimeError("no chromium")), \
                patch("tgbot.service.render_chart", return_value=b"MPL"):
            self.assertEqual(service._default_chart(self.state), b"MPL")

    def test_a_missing_playwright_also_falls_back(self):
        with patch.object(config, "CHART_RENDERER", "tradingview"), \
                patch("live.chart_tv.render_chart_tv", side_effect=ModuleNotFoundError("playwright")), \
                patch("tgbot.service.render_chart", return_value=b"MPL"):
            self.assertEqual(service._default_chart(self.state), b"MPL")


if __name__ == "__main__":
    unittest.main()
