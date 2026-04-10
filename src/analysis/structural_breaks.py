"""
Structural Break Detection — detects when market regimes change in real-time.

Complements HMM (which classifies *what* regime we're in) by detecting *when*
a regime change occurs. Uses CUSUM and Bai-Perron tests.

When a break is detected within the last N days:
- Raise conviction threshold (historical patterns less reliable)
- Shorten signal half-life (signals decay faster during transitions)
- Flag signals for manual review
"""

import numpy as np
import structlog
from datetime import datetime

logger = structlog.get_logger()


class StructuralBreakDetector:
    """
    Detects structural breaks in return series using CUSUM and variance
    ratio tests. Lighter-weight alternative to ruptures library for
    real-time monitoring.
    """

    def __init__(
        self,
        cusum_threshold: float = 2.0,
        min_segment: int = 20,
        lookback_days: int = 252,
        recent_break_window: int = 10,
    ):
        """
        Args:
            cusum_threshold: Sigma multiplier for CUSUM break detection.
            min_segment: Minimum segment length between breaks.
            lookback_days: How far back to look for breaks.
            recent_break_window: Days within which a break is "recent".
        """
        self.cusum_threshold = cusum_threshold
        self.min_segment = min_segment
        self.lookback_days = lookback_days
        self.recent_break_window = recent_break_window

    def detect_breaks(
        self,
        returns: np.ndarray | list[float],
        dates: list | None = None,
    ) -> dict:
        """
        Detect structural breaks in a return series.

        Args:
            returns: Daily returns (most recent last).
            dates: Optional aligned dates for labeling breaks.

        Returns:
            Dict with detected breaks, recent break flag, and regime stats.
        """
        returns = np.array(returns)
        if len(returns) < self.min_segment * 2:
            return {"status": "insufficient_data", "n": len(returns)}

        # Use only lookback window
        if len(returns) > self.lookback_days:
            returns = returns[-self.lookback_days:]
            if dates:
                dates = dates[-self.lookback_days:]

        # CUSUM test for mean shifts
        cusum_breaks = self._cusum_test(returns)

        # Variance ratio test for volatility regime changes
        var_breaks = self._variance_ratio_test(returns)

        # Combine and deduplicate breaks
        all_breaks = self._merge_breaks(cusum_breaks, var_breaks, dates)

        # Check for recent breaks
        n = len(returns)
        recent_breaks = [
            b for b in all_breaks
            if b["index"] >= n - self.recent_break_window
        ]
        has_recent_break = len(recent_breaks) > 0

        # Compute segment statistics
        segments = self._segment_stats(returns, all_breaks)

        # Threshold adjustment recommendation
        if has_recent_break:
            threshold_adj = 0.10  # raise conviction threshold by 10pp
            halflife_adj = 0.5   # shorten half-life by 50%
        elif len(all_breaks) > 3:
            threshold_adj = 0.05  # volatile regime, moderate adjustment
            halflife_adj = 0.75
        else:
            threshold_adj = 0.0
            halflife_adj = 1.0

        result = {
            "status": "complete",
            "n_observations": len(returns),
            "n_breaks_detected": len(all_breaks),
            "breaks": all_breaks,
            "has_recent_break": has_recent_break,
            "recent_breaks": recent_breaks,
            "segments": segments,
            "threshold_adjustment": threshold_adj,
            "halflife_multiplier": halflife_adj,
        }

        logger.info(
            "structural_breaks.detected",
            n_breaks=len(all_breaks),
            has_recent=has_recent_break,
            threshold_adj=threshold_adj,
        )
        return result

    def _cusum_test(self, returns: np.ndarray) -> list[int]:
        """
        CUSUM test for mean shifts.

        Detects points where cumulative deviations from the mean exceed
        a threshold, indicating a structural change in the mean return.
        """
        n = len(returns)
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)

        if std == 0:
            return []

        # Forward CUSUM
        cusum = np.zeros(n)
        cusum[0] = 0
        for i in range(1, n):
            cusum[i] = cusum[i - 1] + (returns[i] - mean)

        # Normalize by std
        cusum_normalized = cusum / (std * np.sqrt(n))

        # Detect breaks: points where CUSUM exceeds threshold
        breaks = []
        last_break = -self.min_segment
        for i in range(self.min_segment, n - self.min_segment):
            if abs(cusum_normalized[i]) > self.cusum_threshold:
                if i - last_break >= self.min_segment:
                    breaks.append(i)
                    last_break = i

        return breaks

    def _variance_ratio_test(self, returns: np.ndarray) -> list[int]:
        """
        Detects volatility regime changes by comparing rolling variance
        in adjacent windows.
        """
        n = len(returns)
        window = self.min_segment
        breaks = []
        last_break = -window

        for i in range(window, n - window):
            pre_var = np.var(returns[i - window:i], ddof=1)
            post_var = np.var(returns[i:i + window], ddof=1)

            if pre_var == 0 or post_var == 0:
                continue

            ratio = max(pre_var, post_var) / min(pre_var, post_var)

            # F-test threshold (approximate): ratio > 2.0 suggests vol regime change
            if ratio > 2.0 and i - last_break >= window:
                breaks.append(i)
                last_break = i

        return breaks

    def _merge_breaks(
        self,
        cusum_breaks: list[int],
        var_breaks: list[int],
        dates: list | None,
    ) -> list[dict]:
        """Merge and deduplicate breaks from different tests."""
        all_indices = set(cusum_breaks) | set(var_breaks)
        # Merge breaks within min_segment of each other
        sorted_indices = sorted(all_indices)
        merged = []
        for idx in sorted_indices:
            if not merged or idx - merged[-1] >= self.min_segment:
                merged.append(idx)

        breaks = []
        for idx in merged:
            break_type = []
            if idx in cusum_breaks:
                break_type.append("mean_shift")
            if idx in var_breaks:
                break_type.append("volatility_change")

            entry = {
                "index": idx,
                "type": "+".join(break_type),
            }
            if dates and idx < len(dates):
                entry["date"] = str(dates[idx])[:10]

            breaks.append(entry)

        return breaks

    def _segment_stats(
        self, returns: np.ndarray, breaks: list[dict]
    ) -> list[dict]:
        """Compute return statistics for each segment between breaks."""
        break_points = [0] + [b["index"] for b in breaks] + [len(returns)]
        segments = []

        for i in range(len(break_points) - 1):
            start = break_points[i]
            end = break_points[i + 1]
            seg_returns = returns[start:end]

            if len(seg_returns) < 2:
                continue

            segments.append({
                "start_idx": start,
                "end_idx": end,
                "n_days": len(seg_returns),
                "mean_return": round(float(np.mean(seg_returns)), 6),
                "volatility": round(float(np.std(seg_returns, ddof=1)), 6),
                "sharpe": round(
                    float(np.mean(seg_returns) / np.std(seg_returns, ddof=1) * np.sqrt(252))
                    if np.std(seg_returns, ddof=1) > 0 else 0.0,
                    4,
                ),
            })

        return segments


def format_breaks_report(result: dict) -> str:
    """Human-readable structural breaks report."""
    if result.get("status") != "complete":
        return f"Structural Breaks: {result.get('status', 'unknown')}"

    lines = [
        "STRUCTURAL BREAK DETECTION",
        "=" * 55,
        f"Observations: {result['n_observations']}",
        f"Breaks detected: {result['n_breaks_detected']}",
        f"Recent break (<{10} days): {'YES' if result['has_recent_break'] else 'No'}",
        f"Threshold adjustment: +{result['threshold_adjustment']:.0%}",
        f"Half-life multiplier: {result['halflife_multiplier']:.0%}",
        "",
    ]

    if result["breaks"]:
        lines.append(f"{'Index':>8} {'Date':>12} {'Type':<25}")
        lines.append("-" * 47)
        for b in result["breaks"]:
            lines.append(f"{b['index']:>8} {b.get('date', 'N/A'):>12} {b['type']:<25}")

    if result["segments"]:
        lines.append("")
        lines.append(f"{'Segment':>10} {'Days':>6} {'Mean Ret':>10} {'Vol':>10} {'Sharpe':>8}")
        lines.append("-" * 46)
        for i, seg in enumerate(result["segments"]):
            lines.append(
                f"{i + 1:>10} {seg['n_days']:>6} {seg['mean_return']:>10.4%} "
                f"{seg['volatility']:>10.4%} {seg['sharpe']:>8.2f}"
            )

    return "\n".join(lines)
