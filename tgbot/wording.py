"""What the bot tells people about what its reports are. One place, so the alert header, the
footer on every report and the replies that mention the rules cannot drift apart.

The claim is bounded by the research (research/README.md): across 65 pairs and 21 years the sweep/FVG
rules did not beat random entries, and EUR_USD, the one live pair, is positive but on too few trades
(116) to tell skill from luck. So a report is an analysis of what the rules say, not a signal, and
says no more than that.
"""

ALERT_HEADER = "🔔 New setup confirmed by the rules (analysis, not a trade signal)"

NOTICE = (
    "ℹ️ This is a mechanical read of fixed rules, not a recommendation. Tested across 21 years of "
    "forex history, the rules did not beat random entries, and EUR_USD's record is too short to "
    "tell skill from luck. Not financial advice."
)

ALERTS_LINE = "Pushed when a new setup confirms (paused around high-impact news). Analysis, not signals."

PROFILE_DESCRIPTION = (
    "Forex structural-analysis bot.\n\n"
    "Reads liquidity sweeps, fair value gaps and confirmation across the Daily, "
    "H4 and H1 timeframes, with a high-impact news check.\n\n"
    "Analysis, not signals: in historical tests these rules did not beat random entries. "
    "Not financial advice. You make your own decisions."
)


def not_enabled(pair: str, enabled: list[str]) -> str:
    return f"{pair} isn't enabled. This bot covers {', '.join(enabled)} only."
