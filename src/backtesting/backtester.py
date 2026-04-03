import argparse
import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from config.settings import Settings
from src.models.database import (
    ConvictionScore,
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

    def __init__(self, settings: Settings, price_client=None):
        self.settings = settings
        self.price_client = price_client
        engine = get_engine(settings.database_url)
        self.session_factory = get_session_factory(engine)

    async def run(
        self,
        start_date: str,
        end_date: str,
        conviction_threshold: float = 0.6,
    ) -> BacktestResult:
        session = self.session_factory()
        try:
            # Get all events with conviction scores in range
            events = (
                session.query(SmartMoneyEvent, ConvictionScore)
                .join(ConvictionScore, SmartMoneyEvent.id == ConvictionScore.event_id)
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
        all_events = [(e, cs) for e, cs in events]
        filtered = [(e, cs) for e, cs in events if cs.conviction >= conviction_threshold]

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
        sem = asyncio.Semaphore(5)

        async def _get_return(event, _cs):
            async with sem:
                try:
                    trade_date = event.trade_date
                    exit_date = trade_date + timedelta(days=hold_days)
                    start_str = trade_date.strftime("%Y-%m-%d")
                    end_str = exit_date.strftime("%Y-%m-%d")

                    prices = await self.price_client.get_price_range(
                        event.ticker, start_str, end_str
                    )
                    if len(prices) < 2:
                        return None

                    entry_price = prices[0].get("adjClose")
                    exit_price = prices[-1].get("adjClose")

                    if not entry_price or not exit_price or entry_price <= 0:
                        return None

                    ret = (exit_price - entry_price) / entry_price
                    # Invert return for sells
                    if event.direction.value == "sell":
                        ret = -ret
                    return ret
                except Exception as e:
                    logger.debug(
                        "backtester.return_failed",
                        ticker=event.ticker,
                        error=str(e),
                    )
                    return None

        tasks = [_get_return(e, cs) for e, cs in events]
        for coro in asyncio.as_completed(tasks):
            ret = await coro
            if ret is not None:
                returns.append(ret)

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

        # Sortino ratio (downside deviation)
        downside = [r for r in returns if r < 0]
        if downside:
            downside_var = sum(r**2 for r in downside) / len(downside)
            downside_std = math.sqrt(downside_var)
            sortino = (avg_ret / downside_std * ann_factor) if downside_std > 0 else 0
        else:
            sortino = float("inf") if avg_ret > 0 else 0

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

        return PeriodMetrics(
            hold_days=hold_days,
            total_trades=n,
            win_rate=round(win_rate, 4),
            avg_return=round(avg_ret, 6),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(min(sortino, 99.0), 4),
            profit_factor=round(min(profit_factor, 99.0), 4),
            max_drawdown=round(max_dd, 6),
        )

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
