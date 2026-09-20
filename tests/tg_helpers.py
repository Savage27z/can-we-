from datetime import datetime, timedelta, timezone

from live.plan import plan_for
from live.state import ActiveSetup, LiveState
from news.filter import NewsEventView, NewsStatus

NOW = datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc)


def setup(status="live_trade", direction="bullish", sweep_time="2026-09-22T09:00:00+00:00",
          confirmed_ago=timedelta(minutes=30)):
    confirmed = status == "live_trade"
    made = ActiveSetup(
        direction=direction, status=status, sweep_time=sweep_time, sweep_extreme=1.1455,
        fvg_low=1.1460, fvg_high=1.1470, confirmation_level=1.1460,
        entry_price=1.1465 if confirmed else None,
        stop_price=1.1450 if confirmed else None,
        target_price=1.1500 if confirmed else None,
        rr=2.33 if confirmed else None,
        h1_candles_to_confirm=2 if confirmed else None,
        confirm_time=(NOW - confirmed_ago).isoformat() if confirmed else None,
    )
    made.plan = plan_for("EUR_USD", made)  # every setup the engine emits carries its plan
    return made


def blackout_news():
    return NewsStatus(status="blackout", blackout_events=[
        NewsEventView("FOMC Statement", "USD", "2026-09-22T15:30:00+00:00", "in 30m")])


def make_state(setups=None, news=None):
    return LiveState(
        pair="EUR_USD", as_of=NOW.isoformat(), current_price=1.1470, daily_bias="neutral",
        active_setups=list(setups or []), liquidity_buy_side=[1.15], liquidity_sell_side=[1.14],
        news=news or NewsStatus(status="clear"),
    )
