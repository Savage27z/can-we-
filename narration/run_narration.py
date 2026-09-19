"""CLI: refresh a pair's data, compute its structured state, and narrate it via
DeepSeek in the Phase 4 report format.

Usage:
    python -m narration.run_narration                 # EUR_USD
    python -m narration.run_narration --pair EUR_USD
    python -m narration.run_narration --no-refresh     # use local data as-is
"""
import argparse
import sys

from live.refresh import refresh_pair
from live.state import compute_current_state

from .generate import narrate_state


def main():
    # Windows consoles often default to a codepage (e.g. cp1252) that can't
    # encode the report's emoji headers; force UTF-8 for this process's stdout.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--no-refresh", action="store_true")
    parser.add_argument("--no-news", action="store_true")
    args = parser.parse_args()

    if not args.no_refresh:
        refresh_pair(args.pair)

    state = compute_current_state(args.pair, with_news=not args.no_news)
    print(narrate_state(state))


if __name__ == "__main__":
    main()
