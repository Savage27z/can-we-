"""CLI: refresh a pair's data from OANDA and print its current structured state
(Phase 3 deliverable — bias, level, invalidation, target; no prose).

Usage:
    python -m live.run_live                  # EUR_USD (the only pair validated in Phase 2)
    python -m live.run_live --pair EUR_USD
    python -m live.run_live --pair EUR_USD --no-refresh   # use local data as-is, skip the API call
"""
import argparse
import json

from .refresh import refresh_pair
from .state import compute_current_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--no-refresh", action="store_true",
                         help="Skip pulling fresh candles; compute state from local data as-is.")
    parser.add_argument("--no-news", action="store_true",
                         help="Skip the economic-calendar check (news status = not_checked).")
    args = parser.parse_args()

    if not args.no_refresh:
        refresh_pair(args.pair)

    state = compute_current_state(args.pair, with_news=not args.no_news)
    print(json.dumps(state.to_dict(), indent=2))


if __name__ == "__main__":
    main()
