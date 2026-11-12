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

Every reading is scored against its own normal, never as a raw level:
  - Temperature / degree days: the 7-day forecast vs the SAME calendar week
    in each of the previous five years. (Comparing with the trailing month
    measured the season — every autumn looked like a cold anomaly.)
  - Severe alerts: today's count as a z-score against the trailing 30 days
    of counts stored by the pipeline. (There are always dozens of active
    alerts somewhere in the US, so the raw count pinned insurers and
    transport at the maximum penalty permanently.) Until enough history
    exists the alert channels return no score.

Data (all free, keyless):
  - Open-Meteo forecast API: next-7d daily max/min temperature, 6 US metros
  - Open-Meteo historical archive API: the same weeks in prior years
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
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
NWS_ALERTS_URL = "https://api.weather.gov/alerts/active"

FORECAST_DAYS = 7
NORMAL_YEARS = 5          # prior years used for the calendar-week normal
MIN_NORMAL_YEARS = 3      # fewer usable years than this -> no temperature score
MIN_BASELINE_DAYS = 10    # days of stored alert counts needed before scoring
# Week-ahead readings this close to the calendar normal carry no information
# for the channels that react to the SIZE of a deviation (either direction).
NORMAL_TEMP_BAND_C = 2.0
NORMAL_DEGREE_DAY_BAND = 1.5
BASELINE_WINDOW_DAYS = 30

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
        today = datetime.utcnow().date()
        daily_vars = "temperature_2m_max,temperature_2m_min"
        lats = ",".join(str(lat) for lat, _ in CITIES.values())
        lons = ",".join(str(lon) for _, lon in CITIES.values())

        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=30.0
        ) as client:
            meteo_task = client.get(OPEN_METEO_URL, params={
                "latitude": lats,
                "longitude": lons,
                "daily": daily_vars,
                "forecast_days": FORECAST_DAYS,
                "timezone": "UTC",
            })
            # One request spanning the same calendar week in each prior year
            archive_task = client.get(OPEN_METEO_ARCHIVE_URL, params={
                "latitude": lats,
                "longitude": lons,
                "daily": daily_vars,
                "start_date": (today - timedelta(days=366 * NORMAL_YEARS + 7)).isoformat(),
                "end_date": (today - timedelta(days=350)).isoformat(),
                "timezone": "UTC",
            })
            alerts_task = client.get(
                NWS_ALERTS_URL,
                params={"severity": "Severe,Extreme", "status": "actual"},
                headers={"Accept": "application/geo+json"},
            )
            meteo_resp, archive_resp, alerts_resp = await asyncio.gather(
                meteo_task, archive_task, alerts_task, return_exceptions=True
            )

        anomalies, dd_deltas = [], []
        forecasts = self._locations(meteo_resp)
        archives = self._locations(archive_resp)
        if forecasts and len(archives) == len(forecasts):
            for forecast_loc, archive_loc in zip(forecasts, archives):
                reading = self._city_anomaly(
                    self._daily_means(forecast_loc), self._daily_means(archive_loc)
                )
                if reading is not None:
                    anomalies.append(reading[0])
                    dd_deltas.append(reading[1])

        n_alerts = None
        if not isinstance(alerts_resp, Exception) and alerts_resp.status_code == 200:
            features = alerts_resp.json().get("features") or []
            n_alerts = len(features)

        context = {
            "temp_anomaly_c": round(float(np.mean(anomalies)), 2) if anomalies else None,
            "degree_day_delta": round(float(np.mean(dd_deltas)), 2) if dd_deltas else None,
            "n_severe_alerts": n_alerts,
            "n_cities": len(anomalies),
            "temperature_baseline": f"same calendar week, prior {NORMAL_YEARS} years",
            "as_of": datetime.utcnow().isoformat(),
        }
        logger.info("weather.context", **{k: v for k, v in context.items() if k != "as_of"})
        return context

    @staticmethod
    def _locations(resp) -> list[dict]:
        """Per-location payloads of an Open-Meteo response ([] on failure)."""
        if isinstance(resp, Exception) or resp.status_code != 200:
            return []
        payload = resp.json()
        return payload if isinstance(payload, list) else [payload]

    @staticmethod
    def _daily_means(location: dict) -> dict[str, float]:
        """ISO date -> daily mean temperature, as (max + min) / 2."""
        daily = location.get("daily") or {}
        out = {}
        for day, hi, lo in zip(
            daily.get("time") or [],
            daily.get("temperature_2m_max") or [],
            daily.get("temperature_2m_min") or [],
        ):
            if hi is not None and lo is not None:
                out[day] = (hi + lo) / 2.0
        return out

    @staticmethod
    def _degree_days(temps: np.ndarray) -> np.ndarray:
        return np.maximum(0, BASE_TEMP_C - temps) + np.maximum(0, temps - BASE_TEMP_C)

    @classmethod
    def _city_anomaly(
        cls, forecast: dict[str, float], archive: dict[str, float]
    ) -> tuple[float, float] | None:
        """(temperature anomaly in C, degree-day delta) of the forecast week
        against the same calendar week in prior years, or None without
        enough data."""
        if len(forecast) < FORECAST_DAYS - 2:
            return None
        days = sorted(forecast)
        forecast_temps = np.array([forecast[d] for d in days])

        normal_means, normal_dds = [], []
        for years_back in range(1, NORMAL_YEARS + 1):
            window = []
            for day in days:
                d = datetime.strptime(day, "%Y-%m-%d").date()
                try:
                    past = d.replace(year=d.year - years_back)
                except ValueError:  # 29 February
                    past = d.replace(year=d.year - years_back, day=28)
                value = archive.get(past.isoformat())
                if value is not None:
                    window.append(value)
            if len(window) >= len(days) - 2:
                temps = np.array(window)
                normal_means.append(float(np.mean(temps)))
                normal_dds.append(float(np.mean(cls._degree_days(temps))))

        if len(normal_means) < MIN_NORMAL_YEARS:
            return None
        anomaly = float(np.mean(forecast_temps)) - float(np.mean(normal_means))
        dd_delta = float(np.mean(cls._degree_days(forecast_temps))) - float(np.mean(normal_dds))
        return anomaly, dd_delta

    @staticmethod
    def with_alert_baseline(context: dict | None, daily_counts: list[float]) -> dict | None:
        """Attach how unusual today's alert count is.

        Args:
            context: Output of get_context().
            daily_counts: Severe-alert counts for previous days (one per day,
                          today excluded), e.g. the trailing 30 days.

        Adds alert_z (None until MIN_BASELINE_DAYS of history exist).
        """
        if not context:
            return context
        out = dict(context)
        counts = [float(c) for c in daily_counts if c is not None]
        out["alert_baseline_days"] = len(counts)
        out["alert_z"] = None
        n_alerts = context.get("n_severe_alerts")
        if n_alerts is not None and len(counts) >= MIN_BASELINE_DAYS:
            mean = float(np.mean(counts))
            # Floor the spread so a quiet, flat month cannot turn one extra
            # alert into a five-sigma event.
            spread = max(float(np.std(counts, ddof=1)), 0.15 * mean, 1.0)
            out["alert_baseline_mean"] = round(mean, 1)
            out["alert_z"] = round((float(n_alerts) - mean) / spread, 2)
        return out

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
        alert_z = context.get("alert_z")

        delta = 0.0  # buy-direction score delta from neutral 50
        rationale = ""

        if channel == "energy" and dd_delta is not None:
            # More degree-days ahead = more heating/cooling fuel demand
            delta = float(np.clip(dd_delta * 2.0, -15, 15))
            rationale = f"degree-day delta {dd_delta:+.1f} vs normal → fuel demand"
        elif channel == "utilities" and dd_delta is not None:
            # Demand rises with temperature extremes in either direction.
            # Inside the normal band this says nothing, so it stays at 50
            # rather than nudging every utility the same way.
            excess = max(0.0, abs(dd_delta) - NORMAL_DEGREE_DAY_BAND)
            delta = float(np.clip(excess * 2.0, 0, 12))
            rationale = f"|degree-day delta| {abs(dd_delta):.1f} vs normal → power demand"
        elif channel == "insurance" and alert_z is not None:
            # More severe weather than usual = near-term claims/loss pressure
            delta = -float(np.clip(alert_z * 5.0, -15, 15))
            rationale = f"{n_alerts} severe alerts ({alert_z:+.1f}σ vs 30d) → claims risk"
        elif channel == "agriculture" and anomaly is not None:
            # Large temperature anomalies stress crops / input costs
            excess = max(0.0, abs(anomaly) - NORMAL_TEMP_BAND_C)
            delta = -float(np.clip(excess * 2.5, 0, 10))
            rationale = f"temp anomaly {anomaly:+.1f}°C vs normal → crop/input stress"
        elif channel == "transport" and alert_z is not None:
            delta = -float(np.clip(alert_z * 4.0, -12, 12))
            rationale = f"{n_alerts} severe alerts ({alert_z:+.1f}σ vs 30d) → disruption risk"
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
                "alert_z": alert_z,
            },
        }
