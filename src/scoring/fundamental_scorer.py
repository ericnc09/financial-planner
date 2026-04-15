from datetime import datetime

import structlog

from src.models.schemas import Direction, EnrichmentSchema

logger = structlog.get_logger()

# Last updated date — used to warn when medians may be stale.
# If >6 months old, a warning is logged at scoring time.
_SECTOR_MEDIANS_UPDATED = "2024-01-15"
_staleness_warned = False

# Median short interest ratio by sector (days-to-cover)
# Source: FINRA / Bloomberg estimates
SECTOR_MEDIAN_SHORT_RATIO = {
    "Technology": 2.5,
    "Communication Services": 2.0,
    "Consumer Cyclical": 3.0,
    "Consumer Defensive": 2.0,
    "Healthcare": 2.5,
    "Financial Services": 1.5,
    "Financials": 1.5,
    "Industrials": 2.5,
    "Energy": 3.5,
    "Utilities": 2.0,
    "Real Estate": 4.0,
    "Basic Materials": 3.0,
    "Materials": 3.0,
}
DEFAULT_SHORT_RATIO = 2.5

# Median trailing P/E by sector (approximate, updated periodically)
# Source: S&P 500 sector medians as of early 2024
SECTOR_MEDIAN_PE = {
    "Technology": 30.0,
    "Communication Services": 22.0,
    "Consumer Cyclical": 22.0,
    "Consumer Defensive": 23.0,
    "Healthcare": 25.0,
    "Financial Services": 14.0,
    "Financials": 14.0,
    "Industrials": 22.0,
    "Energy": 11.0,
    "Utilities": 18.0,
    "Real Estate": 35.0,
    "Basic Materials": 15.0,
    "Materials": 15.0,
}
DEFAULT_SECTOR_PE = 20.0  # fallback when sector unknown


def _check_sector_median_staleness():
    """Log a warning if sector medians haven't been updated in >6 months."""
    global _staleness_warned
    if _staleness_warned:
        return
    updated = datetime.strptime(_SECTOR_MEDIANS_UPDATED, "%Y-%m-%d")
    age_days = (datetime.utcnow() - updated).days
    if age_days > 180:
        logger.warning(
            "fundamental_scorer.stale_sector_medians",
            last_updated=_SECTOR_MEDIANS_UPDATED,
            age_days=age_days,
            hint="Sector median P/E and short ratios may be outdated. "
                 "Update SECTOR_MEDIAN_PE and SECTOR_MEDIAN_SHORT_RATIO "
                 "in fundamental_scorer.py with current values.",
        )
        _staleness_warned = True


class FundamentalScorer:
    """Scores fundamental quality and price setup (0-1). Direction-aware."""

    WEIGHTS = {
        "valuation": 0.22,
        "momentum": 0.18,
        "volatility_regime": 0.13,
        "drawdown_opportunity": 0.18,
        "liquidity": 0.19,
        "short_interest": 0.10,
    }

    def score(self, enrichment: EnrichmentSchema, direction: Direction) -> float:
        _check_sector_median_staleness()
        scores = {
            "valuation": self._valuation_score(enrichment.pe_ratio, direction, enrichment.sector),
            "momentum": self._momentum_score(
                enrichment.momentum_30d, enrichment.momentum_90d, direction
            ),
            "volatility_regime": self._volatility_regime_score(enrichment.rsi_14d),
            "drawdown_opportunity": self._drawdown_opportunity_score(
                enrichment.drawdown_from_52w_high, direction
            ),
            "liquidity": self._liquidity_score(enrichment.avg_volume_30d),
            "short_interest": self._short_interest_score(
                enrichment.short_ratio, direction, enrichment.sector
            ),
        }
        composite = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
        return round(min(1.0, max(0.0, composite)), 4)

    def _valuation_score(
        self, pe_ratio: float | None, direction: Direction, sector: str | None = None
    ) -> float:
        if pe_ratio is None:
            return 0.5
        if pe_ratio <= 0:
            # Negative P/E = unprofitable
            return 0.3 if direction == Direction.BUY else 0.6

        # Sector-relative P/E: ratio of stock P/E to sector median
        sector_pe = SECTOR_MEDIAN_PE.get(sector, DEFAULT_SECTOR_PE) if sector else DEFAULT_SECTOR_PE
        relative_pe = pe_ratio / sector_pe  # <1 = cheap vs sector, >1 = expensive

        if direction == Direction.BUY:
            # Cheap relative to sector = good buy
            if relative_pe < 0.6:
                return 1.0
            elif relative_pe < 0.85:
                return 0.7
            elif relative_pe < 1.2:
                return 0.5
            elif relative_pe < 1.6:
                return 0.3
            return 0.2  # very expensive vs sector
        else:  # SELL — expensive relative to sector confirms overvaluation
            if relative_pe > 1.6:
                return 1.0
            elif relative_pe > 1.2:
                return 0.7
            elif relative_pe > 0.85:
                return 0.5
            elif relative_pe > 0.6:
                return 0.3
            return 0.2  # very cheap vs sector, selling is contrarian

    def _momentum_score(
        self,
        mom_30d: float | None,
        mom_90d: float | None,
        direction: Direction,
    ) -> float:
        if mom_30d is None:
            return 0.5
        if direction == Direction.BUY:
            # Positive but not parabolic is ideal
            if 0.0 <= mom_30d <= 0.15:
                return 1.0
            elif 0.15 < mom_30d <= 0.30:
                return 0.7
            elif mom_30d > 0.30:
                return 0.4  # overextended
            else:
                # Negative momentum — contrarian OK but riskier
                return 0.5
        else:  # SELL
            if mom_30d < -0.10:
                return 1.0  # already falling, sell confirmed
            elif mom_30d < 0:
                return 0.7
            elif mom_30d <= 0.15:
                return 0.4
            return 0.3  # selling into a rally is contrarian

    def _volatility_regime_score(self, rsi: float | None) -> float:
        if rsi is None:
            return 0.5
        if 30 <= rsi <= 70:
            return 1.0  # healthy regime
        elif 20 <= rsi < 30 or 70 < rsi <= 80:
            return 0.6  # approaching extreme
        return 0.4  # extreme — higher risk

    def _drawdown_opportunity_score(
        self, drawdown: float | None, direction: Direction
    ) -> float:
        if drawdown is None:
            return 0.5
        if direction == Direction.BUY:
            if drawdown >= 0.20:
                return 1.0  # 20%+ dip buy
            elif drawdown >= 0.10:
                return 0.7
            elif drawdown >= 0.05:
                return 0.5
            return 0.3  # near highs, less opportunity
        else:  # SELL — near highs is better for shorts
            if drawdown < 0.05:
                return 1.0  # near top
            elif drawdown < 0.10:
                return 0.7
            return 0.4  # already down, less sell conviction

    def _short_interest_score(
        self, short_ratio: float | None, direction: Direction, sector: str | None = None
    ) -> float:
        """Score based on short interest ratio (days-to-cover).

        BUY: High short ratio → short squeeze potential → bullish boost.
             But extremely high ratio (>7 days) → crowded short, risky entry.
        SELL: High short ratio → confirms bearish thesis and crowded trade.
        """
        if short_ratio is None:
            return 0.5  # neutral when unknown

        sector_median = SECTOR_MEDIAN_SHORT_RATIO.get(sector, DEFAULT_SHORT_RATIO) if sector else DEFAULT_SHORT_RATIO
        relative_sr = short_ratio / max(sector_median, 0.1)

        if direction == Direction.BUY:
            # Moderate squeeze potential is best; extreme shorts can signal real trouble
            if 1.5 <= relative_sr <= 3.0:
                return 0.85  # elevated short — squeeze potential
            elif 1.0 <= relative_sr < 1.5:
                return 0.65  # mildly elevated
            elif relative_sr < 1.0:
                return 0.45  # lightly shorted — less squeeze potential
            else:  # relative_sr > 3.0 — extreme short, could mean real distress
                return 0.35
        else:  # SELL
            # High short interest confirms bear thesis but crowded = squeeze risk
            if relative_sr >= 2.5:
                return 0.80  # heavily shorted, confirms bearish positioning
            elif relative_sr >= 1.5:
                return 0.65
            elif relative_sr >= 1.0:
                return 0.45
            return 0.30  # lightly shorted, selling against the crowd

    def _liquidity_score(self, avg_volume: float | None) -> float:
        if avg_volume is None:
            return 0.5
        if avg_volume >= 1_000_000:
            return 1.0
        elif avg_volume >= 500_000:
            return 0.8
        elif avg_volume >= 100_000:
            return 0.5
        return 0.2  # illiquid
