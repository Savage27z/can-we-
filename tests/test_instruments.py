import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from backtest import rules
from data_pipeline import instruments


def raw(name, pip=-4, digits=5, **extra):
    return {"name": name, "type": "CURRENCY", "pipLocation": pip, "displayPrecision": digits,
            "tradeUnitsPrecision": 0, "minimumTradeSize": "1", "marginRate": "0.02", **extra}


class BuildCatalogTests(unittest.TestCase):
    def test_fields_are_converted_and_the_name_is_split(self):
        (eur_usd,) = instruments.build_catalog(
            [raw("EUR_USD")], {"EUR_USD": datetime(2002, 5, 6, 21, tzinfo=timezone.utc)})
        self.assertEqual((eur_usd.base, eur_usd.quote), ("EUR", "USD"))
        self.assertEqual(eur_usd.pip_location, -4)
        self.assertEqual(eur_usd.minimum_trade_size, 1.0)
        self.assertEqual(eur_usd.margin_rate, 0.02)
        self.assertEqual(eur_usd.history_start, "2002-05-06")

    def test_sorted_by_name_and_a_missing_first_candle_is_none(self):
        built = instruments.build_catalog([raw("USD_JPY", -2, 3), raw("EUR_USD")], {})
        self.assertEqual([i.name for i in built], ["EUR_USD", "USD_JPY"])
        self.assertIsNone(built[0].history_start)

    def test_pip_size_is_a_power_of_ten(self):
        built = {i.name: i for i in instruments.build_catalog(
            [raw("EUR_USD"), raw("USD_JPY", -2, 3)], {})}
        self.assertEqual(built["EUR_USD"].pip_size, 0.0001)
        self.assertEqual(built["USD_JPY"].pip_size, 0.01)


class SaveLoadTests(unittest.TestCase):
    def test_round_trip_through_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catalog.json"
            original = instruments.build_catalog([raw("EUR_USD"), raw("USD_JPY", -2, 3)], {})
            instruments.save_catalog(original, path)
            self.assertEqual(list(instruments.load_catalog(path).values()), original)

    def test_a_missing_catalog_file_is_an_empty_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(instruments.load_catalog(Path(tmp) / "nope.json"), {})


class ShippedCatalogTests(unittest.TestCase):
    """The committed catalog is what every environment uses, so it is worth pinning."""

    def test_it_covers_the_pairs_the_project_uses(self):
        for name in ("EUR_USD", "GBP_USD", "USD_JPY"):
            self.assertIsNotNone(instruments.find(name), name)

    def test_the_majors_have_the_pip_sizes_the_strategy_was_built_on(self):
        self.assertEqual(instruments.get("EUR_USD").pip_size, 0.0001)
        self.assertEqual(instruments.get("GBP_USD").pip_size, 0.0001)
        self.assertEqual(instruments.get("USD_JPY").pip_size, 0.01)

    def test_every_instrument_is_internally_consistent(self):
        catalog = instruments.load_catalog()
        self.assertGreaterEqual(len(catalog), 60)
        for i in catalog.values():
            self.assertEqual(i.name, f"{i.base}_{i.quote}")
            self.assertLess(i.pip_location, 0, i.name)
            # A pip is never finer than the quote's last displayed digit.
            self.assertGreaterEqual(i.display_precision, -i.pip_location, i.name)
            self.assertIsNotNone(i.history_start, i.name)

    def test_an_unknown_instrument_raises_a_helpful_error(self):
        with self.assertRaises(KeyError) as ctx:
            instruments.get("XXX_YYY")
        self.assertIn("--refresh", str(ctx.exception))


class PipSizeRuleTests(unittest.TestCase):
    def test_rules_take_the_pip_size_from_the_catalog(self):
        for name, expected in (("EUR_USD", 0.0001), ("USD_JPY", 0.01), ("EUR_JPY", 0.01)):
            self.assertEqual(rules.pip_size(name), expected, name)

    def test_the_four_pairs_the_old_jpy_rule_got_wrong_are_now_right(self):
        for name in ("EUR_HUF", "USD_HUF", "USD_THB"):
            self.assertEqual(rules.pip_size(name), 0.01, name)
        self.assertEqual(rules.pip_size("HKD_JPY"), 0.0001)

    def test_the_stop_buffer_follows_the_real_pip_size(self):
        self.assertAlmostEqual(rules.stop_buffer("EUR_USD"), 0.0005)
        self.assertAlmostEqual(rules.stop_buffer("USD_HUF"), 0.05)
        self.assertAlmostEqual(rules.stop_buffer("HKD_JPY"), 0.0005)

    def test_an_instrument_outside_the_catalog_falls_back_to_the_old_rule(self):
        self.assertEqual(rules.pip_size("XXX_JPY"), 0.01)
        self.assertEqual(rules.pip_size("XXX_USD"), 0.0001)


if __name__ == "__main__":
    unittest.main()
