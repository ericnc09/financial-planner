"""
Cross-Asset Correlation Filter — detects and dampens correlated signal clusters.

When multiple congressional/insider signals hit the same sector within a short
window, they are not independent signals — they represent one sector bet.
This filter:
1. Groups signals by sector + time window
2. Computes pairwise return correlations between recommended tickers
3. Applies a sqrt(n) dampening factor for correlated clusters
4. Flags portfolio-level concentration risk
"""

import numpy as np
import structlog
from collections import defaultdict
from datetime import datetime, timedelta

logger = structlog.get_logger()


class CorrelationFilter:
    """
    Detects and penalizes correlated signal clusters to prevent
    hidden sector concentration bets.
    """

    def __init__(
        self,
        time_window_days: int = 7,
        max_sector_fraction: float = 0.30,
        correlation_threshold: float = 0.60,
    ):
        """
        Args:
            time_window_days: Days within which signals are considered co-temporal.
            max_sector_fraction: Flag when this fraction of signals are in one sector.
            correlation_threshold: Flag pairs above this return correlation.
        """
        self.time_window_days = time_window_days
        self.max_sector_fraction = max_sector_fraction
        self.correlation_threshold = correlation_threshold

    def analyze_cluster_risk(
        self,
        signals: list[dict],
        return_histories: dict[str, list[float]] | None = None,
    ) -> dict:
        """
        Analyze a batch of signals for correlated clustering.

        Args:
            signals: List of signal dicts with "ticker", "sector",
                     "date" or "disclosure_date", "direction", "conviction".
            return_histories: Optional dict of ticker -> daily returns list
                              for computing pairwise correlations.

        Returns:
            Dict with sector concentration, clusters, dampening factors,
            and correlation matrix if return histories provided.
        """
        if len(signals) < 2:
            return {"status": "too_few_signals", "n": len(signals)}

        # Sector concentration analysis
        sector_counts = defaultdict(list)
        for sig in signals:
            sector = sig.get("sector", "Unknown")
            sector_counts[sector].append(sig)

        total = len(signals)
        sector_analysis = {}
        concentration_warnings = []

        for sector, sigs in sector_counts.items():
            fraction = len(sigs) / total
            sector_analysis[sector] = {
                "n_signals": len(sigs),
                "fraction": round(fraction, 4),
                "tickers": list(set(s.get("ticker", "?") for s in sigs)),
            }
            if fraction > self.max_sector_fraction:
                concentration_warnings.append({
                    "sector": sector,
                    "fraction": round(fraction, 4),
                    "n_signals": len(sigs),
                    "threshold": self.max_sector_fraction,
                })

        # Temporal clustering: group by sector + time window
        clusters = self._find_temporal_clusters(signals)

        # Compute dampening factors for clustered signals
        dampened_signals = self._apply_dampening(signals, clusters)

        # Pairwise correlation matrix if return histories provided
        correlation_matrix = None
        high_corr_pairs = []
        if return_histories:
            correlation_matrix, high_corr_pairs = self._compute_correlations(
                signals, return_histories
            )

        result = {
            "status": "complete",
            "n_signals": total,
            "sector_analysis": sector_analysis,
            "concentration_warnings": concentration_warnings,
            "n_clusters": len(clusters),
            "clusters": clusters,
            "dampened_signals": dampened_signals,
            "high_correlation_pairs": high_corr_pairs,
        }

        logger.info(
            "correlation_filter.analyzed",
            n_signals=total,
            n_sectors=len(sector_analysis),
            n_warnings=len(concentration_warnings),
            n_clusters=len(clusters),
            n_high_corr=len(high_corr_pairs),
        )
        return result

    def _find_temporal_clusters(self, signals: list[dict]) -> list[dict]:
        """Group signals by sector + time window."""
        clusters = []
        sector_groups = defaultdict(list)

        for sig in signals:
            sector = sig.get("sector", "Unknown")
            sector_groups[sector].append(sig)

        for sector, sigs in sector_groups.items():
            if len(sigs) < 2:
                continue

            # Sort by date
            dated = sorted(sigs, key=lambda s: str(s.get("date", s.get("disclosure_date", ""))))

            # Sliding window clustering
            current_cluster = [dated[0]]
            for i in range(1, len(dated)):
                d1 = str(dated[i - 1].get("date", dated[i - 1].get("disclosure_date", "")))[:10]
                d2 = str(dated[i].get("date", dated[i].get("disclosure_date", "")))[:10]
                try:
                    dt1 = datetime.strptime(d1, "%Y-%m-%d")
                    dt2 = datetime.strptime(d2, "%Y-%m-%d")
                    if (dt2 - dt1).days <= self.time_window_days:
                        current_cluster.append(dated[i])
                    else:
                        if len(current_cluster) >= 2:
                            clusters.append({
                                "sector": sector,
                                "n_signals": len(current_cluster),
                                "tickers": [s.get("ticker") for s in current_cluster],
                                "dampening": round(1 / np.sqrt(len(current_cluster)), 4),
                            })
                        current_cluster = [dated[i]]
                except ValueError:
                    current_cluster = [dated[i]]

            if len(current_cluster) >= 2:
                clusters.append({
                    "sector": sector,
                    "n_signals": len(current_cluster),
                    "tickers": [s.get("ticker") for s in current_cluster],
                    "dampening": round(1 / np.sqrt(len(current_cluster)), 4),
                })

        return clusters

    def _apply_dampening(
        self, signals: list[dict], clusters: list[dict]
    ) -> list[dict]:
        """Apply 1/sqrt(n) conviction dampening for clustered signals."""
        # Build ticker -> dampening lookup
        ticker_dampening = {}
        for cluster in clusters:
            for ticker in cluster["tickers"]:
                # Use the strongest dampening if ticker appears in multiple clusters
                current = ticker_dampening.get(ticker, 1.0)
                ticker_dampening[ticker] = min(current, cluster["dampening"])

        dampened = []
        for sig in signals:
            ticker = sig.get("ticker")
            dampening = ticker_dampening.get(ticker, 1.0)
            original_conviction = sig.get("conviction", 0)

            dampened.append({
                "ticker": ticker,
                "original_conviction": original_conviction,
                "dampening_factor": dampening,
                "adjusted_conviction": round(original_conviction * dampening, 4),
                "in_cluster": dampening < 1.0,
            })

        return dampened

    def _compute_correlations(
        self,
        signals: list[dict],
        return_histories: dict[str, list[float]],
    ) -> tuple[dict, list[dict]]:
        """Compute pairwise return correlations between signal tickers."""
        tickers = list(set(
            s.get("ticker") for s in signals
            if s.get("ticker") in return_histories
        ))

        if len(tickers) < 2:
            return {}, []

        # Build return matrix
        min_len = min(len(return_histories[t]) for t in tickers)
        if min_len < 20:
            return {}, []

        returns_matrix = np.array([
            return_histories[t][-min_len:] for t in tickers
        ])
        corr_matrix = np.corrcoef(returns_matrix)

        # Find high-correlation pairs
        high_corr = []
        matrix_dict = {}
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                corr = float(corr_matrix[i, j])
                matrix_dict[f"{tickers[i]}-{tickers[j]}"] = round(corr, 4)
                if abs(corr) > self.correlation_threshold:
                    high_corr.append({
                        "ticker_1": tickers[i],
                        "ticker_2": tickers[j],
                        "correlation": round(corr, 4),
                    })

        return matrix_dict, high_corr
