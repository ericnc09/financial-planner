"""
Granger Causality Test — validates the foundational premise that insider/
congressional trading signals actually predict future returns.

Tests: H0: signal_activity does NOT Granger-cause forward_returns

If p > 0.05, the signal source has no predictive power and should be
downweighted or excluded from the ensemble. Run quarterly to detect
regime changes in signal informativeness.
"""

import numpy as np
import structlog
from datetime import datetime, timedelta

logger = structlog.get_logger()


class GrangerCausalityAnalyzer:
    """
    Tests whether smart money signal activity Granger-causes stock returns.

    Constructs a binary signal time series (1 on days with insider/congressional
    trades, 0 otherwise) and tests whether lagged values of this series improve
    predictions of forward returns beyond what lagged returns alone provide.
    """

    def __init__(self, max_lag: int = 20, significance: float = 0.05):
        """
        Args:
            max_lag: Maximum lag (trading days) to test. 20 ≈ 1 month.
            significance: p-value threshold for rejecting H0.
        """
        self.max_lag = max_lag
        self.significance = significance

    def test_signal_source(
        self,
        events: list[dict],
        returns: dict[str, list[tuple]],
        source_type: str | None = None,
        direction: str | None = None,
    ) -> dict:
        """
        Run Granger causality test for a signal source.

        Args:
            events: List of signal dicts with "ticker", "trade_date" or "date",
                    "source_type" (insider/congressional), "direction".
            returns: Dict of ticker -> list of (date, daily_return) tuples,
                     sorted chronologically.
            source_type: Filter events to this source type (optional).
            direction: Filter events to this direction (optional).

        Returns:
            Dict with test results per ticker and aggregate conclusion.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            logger.error("granger.import_failed", hint="pip install statsmodels")
            return {"status": "error", "message": "statsmodels not installed"}

        # Filter events
        filtered = events
        if source_type:
            filtered = [e for e in filtered if e.get("source_type") == source_type]
        if direction:
            filtered = [e for e in filtered if e.get("direction") == direction]

        if not filtered:
            return {
                "status": "no_events",
                "source_type": source_type,
                "direction": direction,
            }

        # Group events by ticker
        ticker_events: dict[str, list[str]] = {}
        for e in filtered:
            ticker = e.get("ticker")
            date = e.get("trade_date") or e.get("date") or e.get("disclosure_date")
            if ticker and date:
                ticker_events.setdefault(ticker, []).append(str(date)[:10])

        ticker_results = {}
        p_values = []

        for ticker, event_dates in ticker_events.items():
            if ticker not in returns or len(returns[ticker]) < self.max_lag * 3:
                continue

            result = self._test_ticker(
                ticker, event_dates, returns[ticker], grangercausalitytests
            )
            if result:
                ticker_results[ticker] = result
                if result.get("min_p_value") is not None:
                    p_values.append(result["min_p_value"])

        if not ticker_results:
            return {
                "status": "insufficient_data",
                "n_tickers_attempted": len(ticker_events),
                "source_type": source_type,
                "direction": direction,
            }

        # Aggregate results
        n_significant = sum(
            1 for r in ticker_results.values() if r.get("is_significant")
        )
        n_tested = len(ticker_results)

        # Overall verdict
        sig_fraction = n_significant / n_tested if n_tested > 0 else 0
        if sig_fraction >= 0.5:
            verdict = "PREDICTIVE"
        elif sig_fraction >= 0.25:
            verdict = "WEAKLY_PREDICTIVE"
        else:
            verdict = "NOT_PREDICTIVE"

        aggregate_p = float(np.median(p_values)) if p_values else 1.0

        result = {
            "status": "complete",
            "source_type": source_type,
            "direction": direction,
            "n_tickers_tested": n_tested,
            "n_significant": n_significant,
            "significance_fraction": round(sig_fraction, 4),
            "median_p_value": round(aggregate_p, 4),
            "verdict": verdict,
            "ticker_results": ticker_results,
            "tested_at": datetime.utcnow().isoformat(),
        }

        logger.info(
            "granger.tested",
            source=source_type, direction=direction,
            n_tested=n_tested, n_sig=n_significant,
            verdict=verdict, median_p=round(aggregate_p, 4),
        )
        return result

    def _test_ticker(
        self,
        ticker: str,
        event_dates: list[str],
        return_series: list[tuple],
        grangercausalitytests,
    ) -> dict | None:
        """
        Run Granger test for a single ticker.

        Constructs a binary signal series aligned with the return series,
        then tests if lagged signal values Granger-cause returns.
        """
        # Build date -> return mapping
        date_return = {str(d)[:10]: r for d, r in return_series}
        dates_sorted = sorted(date_return.keys())

        if len(dates_sorted) < self.max_lag * 3:
            return None

        event_set = set(event_dates)

        # Build aligned arrays: [returns, signal_indicator]
        ret_arr = []
        sig_arr = []
        for d in dates_sorted:
            ret_arr.append(date_return[d])
            sig_arr.append(1.0 if d in event_set else 0.0)

        ret_arr = np.array(ret_arr)
        sig_arr = np.array(sig_arr)

        # Need sufficient signal variation
        if sig_arr.sum() < 3:
            return {"ticker": ticker, "status": "too_few_events", "n_events": int(sig_arr.sum())}

        # Stack into 2-column matrix: [returns, signal]
        data = np.column_stack([ret_arr, sig_arr])

        # Handle any NaN/Inf
        mask = np.all(np.isfinite(data), axis=1)
        data = data[mask]

        if len(data) < self.max_lag * 3:
            return None

        try:
            results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
        except Exception as e:
            logger.debug("granger.test_failed", ticker=ticker, error=str(e))
            return {"ticker": ticker, "status": "test_failed", "error": str(e)}

        # Extract p-values from F-test at each lag
        lag_results = {}
        min_p = 1.0
        best_lag = None

        for lag in range(1, self.max_lag + 1):
            if lag not in results:
                continue
            test_result = results[lag]
            # test_result is a tuple: (test_dict, ols_results)
            f_test = test_result[0].get("ssr_ftest")
            if f_test is None:
                continue
            p_value = float(f_test[1])  # (F-stat, p-value, df_denom, df_num)
            f_stat = float(f_test[0])

            lag_results[lag] = {
                "f_statistic": round(f_stat, 4),
                "p_value": round(p_value, 4),
                "is_significant": p_value < self.significance,
            }

            if p_value < min_p:
                min_p = p_value
                best_lag = lag

        is_significant = min_p < self.significance

        return {
            "ticker": ticker,
            "status": "tested",
            "n_observations": len(data),
            "n_events": int(sig_arr.sum()),
            "best_lag": best_lag,
            "min_p_value": round(min_p, 4),
            "is_significant": is_significant,
            "lag_results": lag_results,
        }

    def compute_source_weights(self, test_results: dict) -> dict[str, float]:
        """
        Convert Granger test results into source-level weight adjustments.

        Returns a multiplier per source type:
        - PREDICTIVE: 1.0 (full weight)
        - WEAKLY_PREDICTIVE: 0.7
        - NOT_PREDICTIVE: 0.3
        """
        verdict = test_results.get("verdict", "NOT_PREDICTIVE")
        weight_map = {
            "PREDICTIVE": 1.0,
            "WEAKLY_PREDICTIVE": 0.7,
            "NOT_PREDICTIVE": 0.3,
        }
        weight = weight_map.get(verdict, 0.5)

        return {
            "source_type": test_results.get("source_type"),
            "direction": test_results.get("direction"),
            "verdict": verdict,
            "weight_multiplier": weight,
            "median_p_value": test_results.get("median_p_value"),
        }


def format_granger_report(result: dict) -> str:
    """Human-readable Granger causality report."""
    if result.get("status") != "complete":
        return f"Granger Causality: {result.get('status', 'unknown')} — {result.get('message', '')}"

    lines = [
        f"GRANGER CAUSALITY TEST — {result.get('source_type', 'all')} / {result.get('direction', 'all')}",
        "=" * 65,
        f"Tickers tested: {result['n_tickers_tested']}",
        f"Significant (p<0.05): {result['n_significant']} ({result['significance_fraction']:.0%})",
        f"Median p-value: {result['median_p_value']:.4f}",
        f"Verdict: {result['verdict']}",
        "",
        f"{'Ticker':<10} {'Events':>8} {'Best Lag':>10} {'Min p':>10} {'Sig?':>8}",
        "-" * 50,
    ]

    for ticker, tr in sorted(result.get("ticker_results", {}).items()):
        if tr.get("status") != "tested":
            continue
        sig_mark = "YES" if tr["is_significant"] else "no"
        lines.append(
            f"{ticker:<10} {tr['n_events']:>8} {tr['best_lag'] or '-':>10} "
            f"{tr['min_p_value']:>10.4f} {sig_mark:>8}"
        )

    return "\n".join(lines)
