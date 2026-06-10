"""
Weather Overlay — scores weather impact for the 5 weather-sensitive sectors.

Channels (sector → mechanism):
  Energy             : degree-day delta → heating/cooling fuel demand
  Utilities          : |degree-day delta| → electricity/gas demand either way
  Financial Services : active severe-weather alerts → insurance loss risk
  Consumer Defensive : |temperature anomaly| → crop/food input cost stress
  Industrials        : active severe-weather alerts → transport disruption

All other sectors return None and are excluded from the ensemble (the
ensemble re-normalizes over available components). Sector labels are
yfinance-coarse (e.g. insurers inside Financial Services), so magnitudes
are deliberately modest: max ±15 points from neutral 50, weighted 0.045.

Data (both free, keyless):
  - Open-Meteo forecast API: next-7d + past-31d daily mean temperature for
    6 large US metros → forecast anomaly + degree-day delta
  - NWS active alerts API: count of Severe/Extreme alerts
"""

import asyncio
from datetime import datetime, timedelta

import httpx
import numpy as np
import structlog

logger = structlog.get_logger()

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"

# Population-center proxies for US weather-driven demand
CITIES = {
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Houston": (29.76, -95.37),
    "Atlanta": (33.75, -84.39),
    "Los Angeles": (34.05, -118.24),
    "Minneapolis": (44.98, -93.27),
}

BASE_TEMP_C = 18.0  # degree-day base temperature

# yfinance sector → weather channel
SECTOR_CHANNELS = {
    "Energy": "energy",
    "Utilities": "utilities",
    "Financial Services": "insurance",
    "Financials": "insurance",
    "Consumer Defensive": "agriculture",
    "Consumer Staples": "agriculture",
    "Industrials": "transport",
}

CACHE_TTL_HOURS = 6


class WeatherOverlay:
    """Fetches a national weather context and scores sector impact 0-100."""

    def __init__(self):
        self._context: dict | None = None
        self._context_at: datetime | None = None
        self._lock = asyncio.Lock()

    async def get_context(self) -> dict | None:
        """National weather context, cached for CACHE_TTL_HOURS."""
        async with self._lock:
            if (
                self._context is not None
                and self._context_at is not None
                and datetime.utcnow() - self._context_at < timedelta(hours=CACHE_TTL_HOURS)
            ):
                return self._context

            try:
                context = await self._fetch_context()
                self._context = context
                self._context_at = datetime.utcnow()
                return context
            except Exception as e:
                logger.warning("weather.context_failed", error=str(e))
                return self._context  # serve stale if we have one

    async def _fetch_context(self) -> dict:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=20.0
        ) as client:
            lats = ",".join(str(lat) for lat, _ in CITIES.values())
            lons = ",".join(str(lon) for _, lon in CITIES.values())
            meteo_task = client.get(OPEN_METEO_URL, params={
                "latitude": lats,
                "longitude": lons,
                "daily": "temperature_2m_mean",
                "forecast_days": 7,
                "past_days": 31,
                "timezone": "UTC",
            })
            alerts_task = client.get(
                NWS_ALERTS_URL,
                params={"severity": "Severe,Extreme", "status": "actual"},
                headers={"Accept": "application/geo+json"},
            )
            meteo_resp, alerts_resp = await asyncio.gather(
                meteo_task, alerts_task, return_exceptions=True
            )

        anomalies, dd_deltas = [], []
        if not isinstance(meteo_resp, Exception) and meteo_resp.status_code == 200:
            payload = meteo_resp.json()
            locations = payload if isinstance(payload, list) else [payload]
            for loc in locations:
                temps = (loc.get("daily") or {}).get("temperature_2m_mean") or []
                temps = [t for t in temps if t is not None]
                if len(temps) < 38:
                    continue
                past, forecast = np.array(temps[:31]), np.array(temps[31:38])
                anomalies.append(float(np.mean(forecast) - np.mean(past)))
                dd = lambda arr: np.maximum(0, BASE_TEMP_C - arr) + np.maximum(0, arr - BASE_TEMP_C)
                dd_deltas.append(float(np.mean(dd(forecast)) - np.mean(dd(past))))

        n_alerts = None
        if not isinstance(alerts_resp, Exception) and alerts_resp.status_code == 200:
            features = alerts_resp.json().get("features") or []
            n_alerts = len(features)

        context = {
            "temp_anomaly_c": round(float(np.mean(anomalies)), 2) if anomalies else None,
            "degree_day_delta": round(float(np.mean(dd_deltas)), 2) if dd_deltas else None,
            "n_severe_alerts": n_alerts,
            "n_cities": len(anomalies),
            "as_of": datetime.utcnow().isoformat(),
        }
        logger.info("weather.context", **{k: v for k, v in context.items() if k != "as_of"})
        return context

    @staticmethod
    def score(sector: str | None, direction: str, context: dict | None) -> dict | None:
        """
        Score weather impact 0-100 for a sector, or None when the sector is
        not weather-sensitive / no context is available.
        """
        if not sector or not context:
            return None
        channel = SECTOR_CHANNELS.get(sector)
        if channel is None:
            return None

        dd_delta = context.get("degree_day_delta")
        anomaly = context.get("temp_anomaly_c")
        n_alerts = context.get("n_severe_alerts")

        delta = 0.0  # buy-direction score delta from neutral 50
        rationale = ""

        if channel == "energy" and dd_delta is not None:
            # More degree-days ahead = more heating/cooling fuel demand
            delta = float(np.clip(dd_delta * 2.0, -15, 15))
            rationale = f"degree-day delta {dd_delta:+.1f} → fuel demand"
        elif channel == "utilities" and dd_delta is not None:
            # Demand rises with temperature extremes in either direction
            delta = float(np.clip(abs(dd_delta) * 1.5, 0, 12))
            rationale = f"|degree-day delta| {abs(dd_delta):.1f} → power demand"
        elif channel == "insurance" and n_alerts is not None:
            # Active severe weather = near-term claims/loss pressure
            delta = -float(np.clip(n_alerts * 0.5, 0, 15))
            rationale = f"{n_alerts} severe alerts → claims risk"
        elif channel == "agriculture" and anomaly is not None:
            # Large temperature anomalies stress crops / input costs
            delta = -float(np.clip(abs(anomaly) * 2.0, 0, 10))
            rationale = f"temp anomaly {anomaly:+.1f}°C → crop/input stress"
        elif channel == "transport" and n_alerts is not None:
            delta = -float(np.clip(n_alerts * 0.4, 0, 12))
            rationale = f"{n_alerts} severe alerts → disruption risk"
        else:
            return None

        if direction == "sell":
            delta = -delta
        score = float(np.clip(50.0 + delta, 0.0, 100.0))

        return {
            "score": round(score, 1),
            "channel": channel,
            "sector": sector,
            "direction": direction,
            "rationale": rationale,
            "context": {
                "temp_anomaly_c": anomaly,
                "degree_day_delta": dd_delta,
                "n_severe_alerts": n_alerts,
            },
        }
