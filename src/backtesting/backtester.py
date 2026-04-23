import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import structlog

from config.settings import Settings
from src.models.database import (
    ConvictionScore,
    Enrichment,
    SmartMoneyEvent,
    get_engine,
    get_session_factory,
    init_db,
)

logger = structlog.get_logger()


@dataclass
class PeriodMetrics:
    hold_days: int
    total_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0


@dataclass
class BacktestResult:
    date_range: tuple[str, str]
    total_signals: int
    filtered_signals: int
    conviction_threshold: float
    filtered_metrics: dict[int, PeriodMetrics] = field(default_factory=dict)
    unfiltered_metrics: dict[int, PeriodMetrics] = field(default_factory=dict)


class Backtester:
    HOLD_PERIODS = [7, 14, 30, 60, 90]
    # Default fallback transaction cost (used when enrichment is unavailable)
    DEFAULT_COST_PER_TRADE = 0.0030  # 30bps round-trip
    # Minimum average daily volume to include a signal (shares/day)
    MIN_AVG_VOLUME = 100_000

    def __init__(self, settings: Settings, price_client=None, execution_delay_days: int = 1):
        self.settings = settings
        self.price_client = price_client
        self.execution_delay_days = execution_delay_days
        engine = get_engine(settings.database_url)
        self.session_factory = get_session_factory(engine)

    @staticmethod
    def _estimate_transaction_cost(avg_volume: float | None, market_cap: float | None) -> float:
        """Estimate round-trip transaction cost based on liquidity tier."""
        if market_cap and market_cap > 50e9:
            return 0.0010  # 10bps — mega-cap
        elif market_cap and market_cap > 10e9:
            return 0.0020  # 20bps — large-cap
        elif market_cap and market_cap > 2e9:
            return 0.0030  # 30bps — mid-cap
        elif avg_volume and avg_volume > 500_000:
            return 0.0040  # 40bps — small-cap with decent volume
        else:
            return 0.0060  # 60bps — illiquid small/micro-cap

    async def run(
        self,
        start_date: str,
        end_date: str,
        conviction_threshold: float = 0.6,
    ) -> BacktestResult:
        session = self.session_factory()
        try:
            # Get all events with conviction scores and enrichment in range
            events = (
                session.query(SmartMoneyEvent, ConvictionScore, Enrichment)
                .join(ConvictionScore, SmartMoneyEvent.id == ConvictionScore.event_id)
                .outerjoin(Enrichment, SmartMoneyEvent.id == Enrichment.event_id)
                .filter(
                    SmartMoneyEvent.trade_date >= datetime.strptime(start_date, "%Y-%m-%d"),
                    SmartMoneyEvent.trade_date <= datetime.strptime(end_date, "%Y-%m-%d"),
                )
                .all()
            )
        finally:
            session.close()

        if not events:
            logger.warning("backtester.no_events", start=start_date, end=end_date)
            return BacktestResult(
                date_range=(start_date, end_date),
                total_signals=0,
                filtered_signals=0,
                conviction_threshold=conviction_threshold,
            )

        # Separate filtered vs unfiltered
        all_events = [(e, cs, enr) for e, cs, enr in events]
        filtered = [(e, cs, enr) for e, cs, enr in events if cs.conviction >= conviction_threshold]

        logger.info(
            "backtester.loaded_events",
            total=len(all_events),
            filtered=len(filtered),
            threshold=conviction_threshold,
        )

        # Compute returns for each hold period
        result = BacktestResult(
            date_range=(start_date, end_date),
            total_signals=len(all_events),
            filtered_signals=len(filtered),
            conviction_threshold=conviction_threshold,
        )

        for period in self.HOLD_PERIODS:
            all_returns = await self._compute_returns(all_events, period)
            filtered_returns = await self._compute_returns(filtered, period)

            result.unfiltered_metrics[period] = self._compute_metrics(
                all_returns, period
            )
            result.filtered_metrics[period] = self._compute_metrics(
                filtered_returns, period
            )

        return result

    async def _compute_returns(
        self,
        events: list[tuple],
        hold_days: int,
    ) -> list[float]:
        """For each event, get price at signal and price after hold_days."""
        returns = []
        illiquid_skipped = 0
        sem = asyncio.Semaphore(5)

        async def _get_return(event, _cs, enrichment):
            nonlocal illiquid_skipped
            async with sem:
                try:
                    # Liquidity gate: skip signals on stocks with insufficient
                    # average daily volume to trade at realistic sizes.
                    avg_vol = getattr(enrichment, "avg_volume_30d", None) if enrichment else None
                    if avg_vol is not None and avg_vol < self.MIN_AVG_VOLUME:
                        logger.debug("backtester.illiquid_skip", ticker=event.ticker, avg_volume=avg_vol)
                        illiquid_skipped += 1
                        return None

                    # Use disclosure_date as entry point (when we actually
                    # learn about the trade) to avoid look-ahead bias.
                    # Fall back to trade_date only if disclosure_date missing.
                    signal_date = getattr(event, "disclosure_date", None) or event.trade_date
                    # Add execution delay: can't trade until next trading day
                    # since Form 4 filings typically arrive after market close.
                    entry_date = signal_date + timedelta(days=self.execution_delay_days)
                    exit_date = entry_date + timedelta(days=hold_days)
                    start_str = entry_date.strftime("%Y-%m-%d")
                    end_str = exit_date.strftime("%Y-%m-%d")

                    prices = await self.price_client.get_price_range(
                        event.ticker, start_str, end_str
                    )
                    if len(prices) < 2:
                        # No price data — likely delisted. Treat as total loss
                        # to prevent survivorship bias (only for events old
                        # enough that data SHOULD exist).
                        days_since = (datetime.utcnow() - entry_date).days
                        if days_since > hold_days + 10:
                            logger.debug("backtester.delisted_total_loss", ticker=event.ticker)
                            return -1.0
                        return None  # too recent, data not yet available

                    entry_price = prices[0].get("adjClose")
                    exit_price = prices[-1].get("adjClose")

                    if not entry_price or entry_price <= 0:
                        return None
                    if not exit_price or exit_price <= 0:
                        # Exit price missing (delisted mid-hold) — total loss
                        return -1.0

                    ret = (exit_price - entry_price) / entry_price
                    # Invert return for sells
                    if event.direction.value == "sell":
                        ret = -ret
                    # Dynamic transaction costs based on stock liquidity tier
                    mkt_cap = getattr(enrichment, "market_cap", None) if enrichment else None
                    cost = self._estimate_transaction_cost(avg_vol, mkt_cap)
                    ret -= cost
                    return ret
                except Exception as e:
                    logger.debug(
                        "backtester.return_failed",
                        ticker=event.ticker,
                        error=str(e),
                    )
                    return None

        tasks = [_get_return(e, cs, enr) for e, cs, enr in events]
        for coro in asyncio.as_completed(tasks):
            ret = await coro
            if ret is not None:
                returns.append(ret)

        if illiquid_skipped:
            logger.info("backtester.illiquid_skipped", count=illiquid_skipped, hold_days=hold_days)

        return returns

    def _compute_metrics(self, returns: list[float], hold_days: int) -> PeriodMetrics:
        if not returns:
            return PeriodMetrics(hold_days=hold_days)

        n = len(returns)
        avg_ret = sum(returns) / n
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]
        win_rate = len(wins) / n if n > 0 else 0

        # Annualization factor
        ann_factor = math.sqrt(252 / max(hold_days, 1))

        # Standard deviation
        if n > 1:
            variance = sum((r - avg_ret) ** 2 for r in returns) / (n - 1)
            std = math.sqrt(variance)
        else:
            std = 0

        # Sharpe ratio (annualized, assuming 0 risk-free for simplicity)
        sharpe = (avg_ret / std * ann_factor) if std > 0 else 0

        # Sortino ratio (downside deviation). No downside → Sortino is
        # undefined; surface that as inf (positive avg), -inf (negative avg),
        # or nan (no activity) rather than silently flattening to a magic 99.
        downside = [r for r in returns if r < 0]
        if downside:
            downside_var = sum(r**2 for r in downside) / len(downside)
            downside_std = math.sqrt(downside_var)
            sortino = (avg_ret / downside_std * ann_factor) if downside_std > 0 else (
                float("inf") if avg_ret > 0 else float("-inf") if avg_ret < 0 else 0.0
            )
        else:
            sortino = float("inf") if avg_ret > 0 else float("-inf") if avg_ret < 0 else 0.0

        # Profit factor
        gross_profits = sum(wins) if wins else 0
        gross_losses = abs(sum(losses)) if losses else 0
        profit_factor = (
            gross_profits / gross_losses if gross_losses > 0 else float("inf")
        )

        # Max drawdown (sequential)
        peak = 0
        max_dd = 0
        cumulative = 0
        for r in returns:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Preserve inf/-inf for "no downside" cases; round only finite values
        # so downstream consumers can branch on np.isinf explicitly.
        sortino_out = sortino if np.isinf(sortino) else round(sortino, 4)
        profit_factor_out = (
            profit_factor if np.isinf(profit_factor) else round(profit_factor, 4)
        )

        return PeriodMetrics(
            hold_days=hold_days,
            total_trades=n,
            win_rate=round(win_rate, 4),
            avg_return=round(avg_ret, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=sortino_out,
            profit_factor=profit_factor_out,
            max_drawdown=round(max_dd, 6),
        )

    async def run_oos(
        self,
        start_date: str,
        end_date: str,
        conviction_threshold: float = 0.6,
        train_pct: float = 0.7,
    ) -> dict:
        """
        Out-of-sample backtest: split data into train (first 70%) and test
        (last 30%) by date. Never touch test until final evaluation.

        Returns:
            Dict with train_result, test_result, and overfit diagnostics.
        """
        session = self.session_factory()
        try:
            events = (
                session.query(SmartMoneyEvent, ConvictionScore, Enrichment)
                .join(ConvictionScore, SmartMoneyEvent.id == ConvictionScore.event_id)
                .outerjoin(Enrichment, SmartMoneyEvent.id == Enrichment.event_id)
                .filter(
                    SmartMoneyEvent.trade_date >= datetime.strptime(start_date, "%Y-%m-%d"),
                    SmartMoneyEvent.trade_date <= datetime.strptime(end_date, "%Y-%m-%d"),
                )
                .order_by(SmartMoneyEvent.trade_date)
                .all()
            )
        finally:
            session.close()

        if len(events) < 10:
            return {"status": "insufficient_data", "n_events": len(events)}

        # Split by date order (no shuffling — preserves temporal order)
        split_idx = int(len(events) * train_pct)
        train_events = events[:split_idx]
        test_events = events[split_idx:]

        train_filtered = [(e, cs, enr) for e, cs, enr in train_events if cs.conviction >= conviction_threshold]
        test_filtered = [(e, cs, enr) for e, cs, enr in test_events if cs.conviction >= conviction_threshold]

        train_date_range = (
            train_events[0][0].trade_date.strftime("%Y-%m-%d"),
            train_events[-1][0].trade_date.strftime("%Y-%m-%d"),
        )
        test_date_range = (
            test_events[0][0].trade_date.strftime("%Y-%m-%d"),
            test_events[-1][0].trade_date.strftime("%Y-%m-%d"),
        )

        # Compute metrics for each split and hold period
        train_metrics = {}
        test_metrics = {}
        overfit_diagnostics = {}

        for period in self.HOLD_PERIODS:
            train_returns = await self._compute_returns(train_filtered, period)
            test_returns = await self._compute_returns(test_filtered, period)

            tm = self._compute_metrics(train_returns, period)
            tsm = self._compute_metrics(test_returns, period)
            train_metrics[period] = tm
            test_metrics[period] = tsm

            # Overfit detection: large train-test gap = overfitting
            sharpe_decay = tm.sharpe_ratio - tsm.sharpe_ratio if tsm.total_trades > 0 else None
            wr_decay = tm.win_rate - tsm.win_rate if tsm.total_trades > 0 else None
            overfit_diagnostics[period] = {
                "sharpe_decay": round(sharpe_decay, 4) if sharpe_decay is not None else None,
                "win_rate_decay": round(wr_decay, 4) if wr_decay is not None else None,
                "is_overfit": sharpe_decay is not None and sharpe_decay > 0.5,
            }

        def _metrics_dict(m):
            return {
                "hold_days": m.hold_days,
                "total_trades": m.total_trades,
                "win_rate": m.win_rate,
                "avg_return": m.avg_return,
                "sharpe_ratio": m.sharpe_ratio,
                "sortino_ratio": m.sortino_ratio,
                "profit_factor": m.profit_factor,
                "max_drawdown": m.max_drawdown,
            }

        logger.info(
            "backtester.oos_complete",
            train_n=len(train_filtered),
            test_n=len(test_filtered),
            train_range=train_date_range,
            test_range=test_date_range,
        )

        return {
            "status": "complete",
            "train_pct": train_pct,
            "conviction_threshold": conviction_threshold,
            "train": {
                "date_range": list(train_date_range),
                "n_signals": len(train_filtered),
                "metrics": {str(k): _metrics_dict(v) for k, v in train_metrics.items()},
            },
            "test": {
                "date_range": list(test_date_range),
                "n_signals": len(test_filtered),
                "metrics": {str(k): _metrics_dict(v) for k, v in test_metrics.items()},
            },
            "overfit_diagnostics": {str(k): v for k, v in overfit_diagnostics.items()},
        }

    def compare_filtered_vs_unfiltered(self, result: BacktestResult) -> str:
        """Generate a readable comparison report."""
        lines = [
            f"=== Backtest: {result.date_range[0]} to {result.date_range[1]} ===",
            f"Total signals: {result.total_signals}",
            f"Filtered (conviction >= {result.conviction_threshold}): {result.filtered_signals}",
            "",
        ]
        for period in self.HOLD_PERIODS:
            f = result.filtered_metrics.get(period)
            u = result.unfiltered_metrics.get(period)
            if not f or not u:
                continue
            lines.append(f"--- {period}-day hold ---")
            lines.append(
                f"  {'Metric':<20} {'Filtered':>12} {'Unfiltered':>12} {'Delta':>12}"
            )
            lines.append(f"  {'Trades':<20} {f.total_trades:>12} {u.total_trades:>12}")
            lines.append(
                f"  {'Win Rate':<20} {f.win_rate:>11.1%} {u.win_rate:>11.1%} "
                f"{f.win_rate - u.win_rate:>+11.1%}"
            )
            lines.append(
                f"  {'Avg Return':<20} {f.avg_return:>11.3%} {u.avg_return:>11.3%} "
                f"{f.avg_return - u.avg_return:>+11.3%}"
            )
            lines.append(
                f"  {'Sharpe':<20} {f.sharpe_ratio:>12.3f} {u.sharpe_ratio:>12.3f} "
                f"{f.sharpe_ratio - u.sharpe_ratio:>+12.3f}"
            )
            lines.append(
                f"  {'Sortino':<20} {f.sortino_ratio:>12.3f} {u.sortino_ratio:>12.3f}"
            )
            lines.append(
                f"  {'Profit Factor':<20} {f.profit_factor:>12.3f} {u.profit_factor:>12.3f}"
            )
            lines.append(
                f"  {'Max Drawdown':<20} {f.max_drawdown:>11.3%} {u.max_drawdown:>11.3%}"
            )
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Money Follows Backtester")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--threshold", type=float, default=0.6, help="Conviction threshold")
    args = parser.parse_args()

    from src.clients.yahoo import YahooClient

    settings = Settings()
    yahoo = YahooClient()
    bt = Backtester(settings, yahoo)

    async def _run():
        result = await bt.run(args.start, args.end, args.threshold)
        print(bt.compare_filtered_vs_unfiltered(result))

    asyncio.run(_run())
