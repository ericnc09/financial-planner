import structlog
from fredapi import Fred

from src.models.schemas import MacroRegime, MacroSnapshotSchema

logger = structlog.get_logger()


class FredClient:
    def __init__(self, api_key: str):
        self._fred = Fred(api_key=api_key)

    def get_latest_yield_spread(self) -> float | None:
        try:
            series = self._fred.get_series("T10Y2Y")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.yield_spread_failed", error=str(e))
            return None

    def get_latest_unemployment_claims(self) -> float | None:
        try:
            series = self._fred.get_series("ICSA")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.unemployment_claims_failed", error=str(e))
            return None

    def get_cpi_yoy(self) -> float | None:
        try:
            series = self._fred.get_series("CPIAUCSL")
            series = series.dropna()
            if len(series) < 13:
                return None
            latest = series.iloc[-1]
            year_ago = series.iloc[-13]
            return ((latest - year_ago) / year_ago) * 100
        except Exception as e:
            logger.warning("fred.cpi_yoy_failed", error=str(e))
            return None

    def get_fed_funds_rate(self) -> float | None:
        try:
            series = self._fred.get_series("FEDFUNDS")
            return float(series.dropna().iloc[-1])
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

    def get_vix(self) -> float | None:
        try:
            series = self._fred.get_series("VIXCLS")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.vix_failed", error=str(e))
            return None

    def get_consumer_sentiment(self) -> float | None:
        try:
            series = self._fred.get_series("UMCSENT")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.consumer_sentiment_failed", error=str(e))
            return None

    def get_money_supply_m2(self) -> float | None:
        try:
            series = self._fred.get_series("M2SL")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.m2_failed", error=str(e))
            return None

    def get_housing_starts(self) -> float | None:
        try:
            series = self._fred.get_series("HOUST")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.housing_starts_failed", error=str(e))
            return None

    def get_industrial_production(self) -> float | None:
        try:
            series = self._fred.get_series("INDPRO")
            return float(series.dropna().iloc[-1])
        except Exception as e:
            logger.warning("fred.industrial_production_failed", error=str(e))
            return None

    def get_extended_macro(self) -> dict:
        """Fetch all extended macro indicators."""
        return {
            "vix": self.get_vix(),
            "consumer_sentiment": self.get_consumer_sentiment(),
            "money_supply_m2": self.get_money_supply_m2(),
            "housing_starts": self.get_housing_starts(),
            "industrial_production": self.get_industrial_production(),
        }

    def get_macro_snapshot(self) -> MacroSnapshotSchema:
        from datetime import datetime

        spread = self.get_latest_yield_spread()
        claims = self.get_latest_unemployment_claims()
        cpi = self.get_cpi_yoy()
        fed_rate = self.get_fed_funds_rate()

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
        )
        return snapshot
