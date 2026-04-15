import structlog
from fredapi import Fred

from src.models.schemas import MacroRegime, MacroSnapshotSchema

logger = structlog.get_logger()


class FredClient:
    def __init__(self, api_key: str):
        self._fred = Fred(api_key=api_key)

    def _get_latest(self, series_id: str, as_of: str | None = None) -> float | None:
        """Fetch the latest observation from a FRED series.

        Args:
            series_id: FRED series identifier (e.g. "T10Y2Y").
            as_of: Optional date string (YYYY-MM-DD). When provided, only
                   observations on or before this date are considered. This
                   ensures point-in-time correctness when scoring historical
                   signals — without it, backfilling would use future data.
        """
        series = self._fred.get_series(series_id)
        if as_of:
            series = series[series.index <= as_of]
        series = series.dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])

    def get_latest_yield_spread(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("T10Y2Y", as_of)
        except Exception as e:
            logger.warning("fred.yield_spread_failed", error=str(e))
            return None

    def get_latest_unemployment_claims(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("ICSA", as_of)
        except Exception as e:
            logger.warning("fred.unemployment_claims_failed", error=str(e))
            return None

    def get_cpi_yoy(self, as_of: str | None = None) -> float | None:
        try:
            series = self._fred.get_series("CPIAUCSL")
            if as_of:
                series = series[series.index <= as_of]
            series = series.dropna()
            if len(series) < 13:
                return None
            latest = series.iloc[-1]
            year_ago = series.iloc[-13]
            return ((latest - year_ago) / year_ago) * 100
        except Exception as e:
            logger.warning("fred.cpi_yoy_failed", error=str(e))
            return None

    def get_fed_funds_rate(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("FEDFUNDS", as_of)
        except Exception as e:
            logger.warning("fred.fed_funds_failed", error=str(e))
            return None

    def _classify_regime(
        self,
        spread: float | None,
        claims: float | None,
        cpi: float | None,
        fed_rate: float | None,
    ) -> tuple[MacroRegime, float]:
        """
        Returns (regime, regime_score).
        regime_score: 0.0 = strong expansion, 1.0 = deep recession.
        """
        score = 0.0
        factors = 0

        if spread is not None:
            factors += 1
            if spread < -0.2:
                score += 1.0  # inverted yield curve
            elif spread < 0.5:
                score += 0.6  # flat
            else:
                score += 0.1  # healthy spread

        if claims is not None:
            factors += 1
            if claims > 350_000:
                score += 1.0
            elif claims > 250_000:
                score += 0.5
            else:
                score += 0.1

        if cpi is not None:
            factors += 1
            if cpi > 5.0:
                score += 0.8  # high inflation stress
            elif cpi > 3.0:
                score += 0.4
            else:
                score += 0.1

        if fed_rate is not None:
            factors += 1
            if fed_rate > 5.0:
                score += 0.7  # restrictive
            elif fed_rate > 3.0:
                score += 0.4
            else:
                score += 0.1

        regime_score = score / factors if factors > 0 else 0.5

        if regime_score < 0.3:
            regime = MacroRegime.EXPANSION
        elif regime_score > 0.6:
            regime = MacroRegime.RECESSION
        else:
            regime = MacroRegime.TRANSITION

        return regime, round(regime_score, 3)

    def get_vix(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("VIXCLS", as_of)
        except Exception as e:
            logger.warning("fred.vix_failed", error=str(e))
            return None

    def get_consumer_sentiment(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("UMCSENT", as_of)
        except Exception as e:
            logger.warning("fred.consumer_sentiment_failed", error=str(e))
            return None

    def get_money_supply_m2(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("M2SL", as_of)
        except Exception as e:
            logger.warning("fred.m2_failed", error=str(e))
            return None

    def get_housing_starts(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("HOUST", as_of)
        except Exception as e:
            logger.warning("fred.housing_starts_failed", error=str(e))
            return None

    def get_industrial_production(self, as_of: str | None = None) -> float | None:
        try:
            return self._get_latest("INDPRO", as_of)
        except Exception as e:
            logger.warning("fred.industrial_production_failed", error=str(e))
            return None

    def get_extended_macro(self, as_of: str | None = None) -> dict:
        """Fetch all extended macro indicators."""
        return {
            "vix": self.get_vix(as_of),
            "consumer_sentiment": self.get_consumer_sentiment(as_of),
            "money_supply_m2": self.get_money_supply_m2(as_of),
            "housing_starts": self.get_housing_starts(as_of),
            "industrial_production": self.get_industrial_production(as_of),
        }

    def get_macro_snapshot(self, as_of: str | None = None) -> MacroSnapshotSchema:
        from datetime import datetime

        spread = self.get_latest_yield_spread(as_of)
        claims = self.get_latest_unemployment_claims(as_of)
        cpi = self.get_cpi_yoy(as_of)
        fed_rate = self.get_fed_funds_rate(as_of)

        regime, regime_score = self._classify_regime(spread, claims, cpi, fed_rate)

        snapshot = MacroSnapshotSchema(
            snapshot_date=datetime.utcnow(),
            yield_spread_10y2y=spread,
            unemployment_claims=claims,
            cpi_yoy=cpi,
            fed_funds_rate=fed_rate,
            regime=regime,
            regime_score=regime_score,
        )
        logger.info(
            "fred.macro_snapshot",
            regime=regime.value,
            score=regime_score,
            spread=spread,
            claims=claims,
            as_of=as_of,
        )
        return snapshot
