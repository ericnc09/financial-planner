import structlog

from src.models.schemas import Direction, EnrichmentSchema

logger = structlog.get_logger()

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


class FundamentalScorer:
    """Scores fundamental quality and price setup (0-1). Direction-aware."""

    WEIGHTS = {
        "valuation": 0.25,
        "momentum": 0.20,
        "volatility_regime": 0.15,
        "drawdown_opportunity": 0.20,
        "liquidity": 0.20,
    }

    def score(self, enrichment: EnrichmentSchema, direction: Direction) -> float:
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
