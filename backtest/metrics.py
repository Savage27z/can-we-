"""Phase 2 deliverable: win rate, avg R:R, max drawdown, signals/month — plus a
funnel breakdown (how many sweep events fell out at each stage) since that's what
actually explains *why* the strategy does or doesn't show an edge, which is what
the Phase 2 gate decision needs.
"""
from collections import Counter
from dataclasses import dataclass

from .setup import SetupResult

TRADED_OUTCOMES = {"win", "loss", "open"}
RESOLVED_OUTCOMES = {"win", "loss"}


@dataclass
class BacktestReport:
    pair: str
    total_sweep_events: int
    funnel: dict
    num_signals: int          # traded (win + loss + open)
    num_wins: int
    num_losses: int
    num_open: int
    win_rate: float | None            # wins / (wins + losses), None if no resolved trades
    avg_planned_rr: float | None      # mean of `rr` across all traded signals
    avg_realized_r: float | None      # mean of realized_r across resolved (win/loss) trades
    expectancy_r: float | None        # win_rate * avg_win_r + loss_rate * avg_loss_r
    max_drawdown_r: float | None      # peak-to-trough decline of cumulative realized R
    months_spanned: float
    signals_per_month: float | None


def build_report(pair: str, results: list[SetupResult], months_spanned: float) -> BacktestReport:
    funnel = Counter(r.outcome for r in results)
    traded = [r for r in results if r.outcome in TRADED_OUTCOMES]
    resolved = [r for r in results if r.outcome in RESOLVED_OUTCOMES]
    wins = [r for r in resolved if r.outcome == "win"]
    losses = [r for r in resolved if r.outcome == "loss"]
    opens = [r for r in results if r.outcome == "open"]

    win_rate = len(wins) / len(resolved) if resolved else None
    avg_planned_rr = sum(r.rr for r in traded) / len(traded) if traded else None
    avg_realized_r = sum(r.realized_r for r in resolved) / len(resolved) if resolved else None

    if resolved and win_rate is not None:
        avg_win_r = sum(r.realized_r for r in wins) / len(wins) if wins else 0.0
        avg_loss_r = sum(r.realized_r for r in losses) / len(losses) if losses else 0.0
        expectancy_r = win_rate * avg_win_r + (1 - win_rate) * avg_loss_r
    else:
        expectancy_r = None

    # Max drawdown over the cumulative realized-R equity curve, in chronological
    # order of confirmation (resolved trades only — an 'open' position has no
    # realized R yet, per §6.5).
    max_dd = None
    if resolved:
        chronological = sorted(resolved, key=lambda r: r.confirm_time)
        cum = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in chronological:
            cum += r.realized_r
            peak = max(peak, cum)
            max_dd = max(max_dd, peak - cum)

    signals_per_month = len(traded) / months_spanned if months_spanned > 0 else None

    return BacktestReport(
        pair=pair,
        total_sweep_events=len(results),
        funnel=dict(funnel),
        num_signals=len(traded),
        num_wins=len(wins),
        num_losses=len(losses),
        num_open=len(opens),
        win_rate=win_rate,
        avg_planned_rr=avg_planned_rr,
        avg_realized_r=avg_realized_r,
        expectancy_r=expectancy_r,
        max_drawdown_r=max_dd,
        months_spanned=months_spanned,
        signals_per_month=signals_per_month,
    )


def format_report(report: BacktestReport) -> str:
    lines = [f"=== {report.pair} ===",
             f"Sweep events found: {report.total_sweep_events}",
             "Funnel:"]
    for stage in ["no_fvg", "bias_block", "invalidated_pre_confirm", "expired",
                  "no_target", "low_rr", "win", "loss", "open"]:
        if stage in report.funnel:
            lines.append(f"  {stage:>24}: {report.funnel[stage]}")
    lines.append(f"Signals (traded, RR>=min): {report.num_signals}  "
                 f"(win={report.num_wins}, loss={report.num_losses}, open={report.num_open})")
    lines.append(f"Win rate (resolved only): "
                 f"{report.win_rate:.1%}" if report.win_rate is not None else "Win rate: n/a")
    lines.append(f"Avg planned R:R: {report.avg_planned_rr:.2f}" if report.avg_planned_rr is not None else "Avg planned R:R: n/a")
    lines.append(f"Avg realized R: {report.avg_realized_r:.2f}" if report.avg_realized_r is not None else "Avg realized R: n/a")
    lines.append(f"Expectancy (R/trade): {report.expectancy_r:.2f}" if report.expectancy_r is not None else "Expectancy: n/a")
    lines.append(f"Max drawdown: {report.max_drawdown_r:.2f}R" if report.max_drawdown_r is not None else "Max drawdown: n/a")
    lines.append(f"Months spanned: {report.months_spanned:.1f}")
    lines.append(f"Signals/month: {report.signals_per_month:.2f}" if report.signals_per_month is not None else "Signals/month: n/a")
    return "\n".join(lines)
