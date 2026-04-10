"""
Information Coefficient (IC) Monitor — tracks whether signal sources are
still generating alpha over time.

IC = rank_correlation(signal_score, forward_return)
ICIR = mean(IC) / std(IC) — should be > 0.5 for a reliable signal

Computes rolling IC on configurable windows and alerts when a signal
source degrades below threshold. This catches slow alpha decay that
single-point backtests miss.
"""

import numpy as np
import structlog
from datetime import datetime
from scipy.stats import spearmanr

logger = structlog.get_logger()


class ICMonitor:
    """
    Rolling Information Coefficient tracker for signal quality monitoring.
    """

    def __init__(
        self,
        window_months: int = 6,
        step_months: int = 1,
        min_signals_per_window: int = 15,
        icir_threshold: float = 0.5,
    ):
        """
        Args:
            window_months: Rolling window size for IC computation.
            step_months: Step size between windows.
            min_signals_per_window: Minimum signals required per window.
            icir_threshold: ICIR below this triggers a degradation alert.
        """
        self.window_months = window_months
        self.step_months = step_months
        self.min_signals = min_signals_per_window
        self.icir_threshold = icir_threshold

    def compute_rolling_ic(
        self,
        records: list[dict],
        score_field: str = "conviction",
        return_field: str = "realized_return",
        date_field: str = "date",
        source_field: str | None = None,
        source_value: str | None = None,
    ) -> dict:
        """
        Compute rolling IC and ICIR across time windows.

        Args:
            records: List of signal records with scores and returns.
            score_field: Field containing the signal score.
            return_field: Field containing the realized return.
            date_field: Field containing the signal date.
            source_field: Optional field to filter by source type.
            source_value: Optional value to filter on source_field.

        Returns:
            Dict with IC series, ICIR, trend analysis, and degradation alerts.
        """
        # Filter by source if specified
        filtered = records
        if source_field and source_value:
            filtered = [r for r in filtered if r.get(source_field) == source_value]

        # Filter to records with both score and return
        valid = [
            r for r in filtered
            if r.get(score_field) is not None and r.get(return_field) is not None
        ]

        if len(valid) < self.min_signals * 2:
            return {
                "status": "insufficient_data",
                "n_records": len(valid),
                "min_required": self.min_signals * 2,
            }

        # Sort by date
        valid = sorted(valid, key=lambda r: str(r.get(date_field, "")))

        # Build monthly buckets
        monthly_buckets = self._bucket_by_month(valid, date_field)
        month_keys = sorted(monthly_buckets.keys())

        if len(month_keys) < self.window_months + 1:
            return {
                "status": "insufficient_history",
                "n_months": len(month_keys),
                "min_required": self.window_months + 1,
            }

        # Rolling IC computation
        ic_series = []
        for i in range(0, len(month_keys) - self.window_months + 1, self.step_months):
            window_keys = month_keys[i:i + self.window_months]
            window_records = []
            for k in window_keys:
                window_records.extend(monthly_buckets[k])

            if len(window_records) < self.min_signals:
                continue

            scores = np.array([float(r[score_field]) for r in window_records])
            returns = np.array([float(r[return_field]) for r in window_records])

            corr, p_value = spearmanr(scores, returns)
            if np.isnan(corr):
                continue

            ic_series.append({
                "window_start": window_keys[0],
                "window_end": window_keys[-1],
                "n_signals": len(window_records),
                "ic": round(float(corr), 4),
                "p_value": round(float(p_value), 4),
                "is_significant": p_value < 0.05,
            })

        if len(ic_series) < 3:
            return {
                "status": "insufficient_windows",
                "n_windows": len(ic_series),
            }

        # Compute ICIR
        ic_values = np.array([w["ic"] for w in ic_series])
        ic_mean = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0

        # Trend analysis: is IC declining?
        n_windows = len(ic_values)
        first_half = ic_values[:n_windows // 2]
        second_half = ic_values[n_windows // 2:]
        ic_trend = float(np.mean(second_half) - np.mean(first_half))

        # Recent IC (last window)
        recent_ic = ic_series[-1]["ic"]

        # Degradation detection
        is_degrading = icir < self.icir_threshold
        is_recent_weak = recent_ic < 0.02  # near-zero recent IC

        if is_degrading and is_recent_weak:
            alert = "CRITICAL — signal source appears to have lost predictive power"
        elif is_degrading:
            alert = "WARNING — ICIR below threshold, signal reliability declining"
        elif ic_trend < -0.05:
            alert = "WATCH — IC trending downward, monitor closely"
        else:
            alert = "OK — signal source is performing within expectations"

        result = {
            "status": "complete",
            "source": f"{source_field}={source_value}" if source_field else "all",
            "n_records": len(valid),
            "n_windows": len(ic_series),
            "ic_mean": round(ic_mean, 4),
            "ic_std": round(ic_std, 4),
            "icir": round(icir, 4),
            "icir_threshold": self.icir_threshold,
            "ic_trend": round(ic_trend, 4),
            "recent_ic": round(recent_ic, 4),
            "is_degrading": is_degrading,
            "alert": alert,
            "ic_series": ic_series,
        }

        logger.info(
            "ic_monitor.computed",
            source=source_value or "all",
            icir=round(icir, 4),
            ic_mean=round(ic_mean, 4),
            recent_ic=round(recent_ic, 4),
            alert=alert,
        )
        return result

    def _bucket_by_month(self, records: list[dict], date_field: str) -> dict:
        buckets: dict[str, list[dict]] = {}
        for r in records:
            d = r.get(date_field, "")
            month_key = str(d)[:7]  # "YYYY-MM"
            buckets.setdefault(month_key, []).append(r)
        return buckets


def format_ic_report(result: dict) -> str:
    """Human-readable IC monitor report."""
    if result.get("status") != "complete":
        return f"IC Monitor: {result.get('status', 'unknown')}"

    lines = [
        f"INFORMATION COEFFICIENT MONITOR — {result['source']}",
        "=" * 60,
        f"Records analyzed: {result['n_records']}",
        f"Rolling windows: {result['n_windows']}",
        "",
        f"IC Mean:     {result['ic_mean']:>8.4f}",
        f"IC Std:      {result['ic_std']:>8.4f}",
        f"ICIR:        {result['icir']:>8.4f}  (threshold: {result['icir_threshold']})",
        f"IC Trend:    {result['ic_trend']:>8.4f}  ({'declining' if result['ic_trend'] < 0 else 'improving'})",
        f"Recent IC:   {result['recent_ic']:>8.4f}",
        "",
        f"Alert: {result['alert']}",
        "",
        f"{'Window':<20} {'N':>6} {'IC':>8} {'p-value':>10} {'Sig?':>6}",
        "-" * 52,
    ]
    for w in result["ic_series"]:
        sig = "YES" if w["is_significant"] else "no"
        lines.append(
            f"{w['window_start']}→{w['window_end']}  {w['n_signals']:>4} "
            f"{w['ic']:>8.4f} {w['p_value']:>10.4f} {sig:>6}"
        )
    return "\n".join(lines)
