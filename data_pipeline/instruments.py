"""Catalog of the instruments the OANDA account can trade: pip size, display
precision, minimum size, margin rate, and where each instrument's history begins.

The catalog is a committed JSON file rather than something fetched at run time, so
every environment (a laptop, the Railway container) sees the same pip sizes, and a
change to one is a visible diff. Refresh it deliberately:

    python -m data_pipeline.instruments --refresh
    python -m data_pipeline.instruments --list
"""
import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

CATALOG_PATH = Path(__file__).with_name("instrument_catalog.json")


@dataclass(frozen=True)
class Instrument:
    name: str                       # OANDA code, e.g. "EUR_USD"
    base: str
    quote: str
    type: str                       # OANDA's own label: CURRENCY, CFD, METAL
    pip_location: int               # power of ten of one pip: -4 for EUR_USD, -2 for USD_JPY
    display_precision: int          # decimals OANDA quotes
    trade_units_precision: int
    minimum_trade_size: float
    margin_rate: float
    history_start: Optional[str]    # date of the earliest daily candle, if it was looked up

    @property
    def pip_size(self) -> float:
        return 10.0 ** self.pip_location


def _from_raw(raw: dict, first_candle: Optional[datetime]) -> Instrument:
    name = raw["name"]
    base, _, quote = name.partition("_")
    return Instrument(
        name=name, base=base, quote=quote, type=raw["type"],
        pip_location=int(raw["pipLocation"]),
        display_precision=int(raw["displayPrecision"]),
        trade_units_precision=int(raw.get("tradeUnitsPrecision", 0)),
        minimum_trade_size=float(raw.get("minimumTradeSize", 1)),
        margin_rate=float(raw.get("marginRate", 0)),
        history_start=first_candle.date().isoformat() if first_candle else None,
    )


def build_catalog(raw_instruments: list[dict],
                  first_candles: dict[str, Optional[datetime]]) -> list[Instrument]:
    """Pure: OANDA's instrument records plus each one's first candle, sorted by name."""
    built = [_from_raw(raw, first_candles.get(raw["name"])) for raw in raw_instruments]
    return sorted(built, key=lambda i: i.name)


def save_catalog(instruments: list[Instrument], path: Path = CATALOG_PATH) -> None:
    payload = {
        "source": "OANDA v20 /accounts/{id}/instruments, plus each first daily candle",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "instruments": [asdict(i) for i in instruments],
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    load_catalog.cache_clear()


@lru_cache(maxsize=None)
def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Instrument]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {row["name"]: Instrument(**row) for row in data["instruments"]}


def find(name: str) -> Optional[Instrument]:
    return load_catalog().get(name)


def get(name: str) -> Instrument:
    found = find(name)
    if found is None:
        raise KeyError(f"{name!r} is not in the instrument catalog "
                       f"({len(load_catalog())} instruments); refresh it with "
                       f"`python -m data_pipeline.instruments --refresh`")
    return found


def names() -> list[str]:
    return sorted(load_catalog())


def refresh_catalog(path: Path = CATALOG_PATH,
                    progress: Callable[[str], None] = print) -> list[Instrument]:
    """Fetches the account's instruments and each one's first daily candle. Read-only."""
    from . import config, oanda_client

    config.require_credentials()
    raw = oanda_client.fetch_instruments()
    progress(f"{len(raw)} instruments listed; looking up where each history starts")
    first: dict[str, Optional[datetime]] = {}
    for record in raw:
        try:
            first[record["name"]] = oanda_client.first_candle_time(record["name"])
        except oanda_client.OandaAPIError as err:
            progress(f"  {record['name']}: could not find its first candle ({err})")
            first[record["name"]] = None
    catalog = build_catalog(raw, first)
    save_catalog(catalog, path)
    progress(f"wrote {len(catalog)} instruments to {path.name}")
    return catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true",
                        help="Re-read the instrument list from OANDA and rewrite the catalog.")
    parser.add_argument("--list", action="store_true", help="Print the catalog.")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.refresh:
        refresh_catalog()
    if args.list or not args.refresh:
        for i in load_catalog().values():
            print(f"{i.name:8} pip {i.pip_size:<7g} digits {i.display_precision}  "
                  f"min {i.minimum_trade_size:<6g} margin {i.margin_rate:<5g} "
                  f"from {i.history_start}")


if __name__ == "__main__":
    main()
