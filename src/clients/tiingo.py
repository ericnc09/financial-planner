import asyncio
from datetime import datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import EnrichmentSchema

logger = structlog.get_logger()


def _compute_rsi(closes: list[float], period: int = 14) -> float | None:
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas[-period:]]
    losses = [-d if d < 0 else 0 for d in deltas[-period:]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


class TiingoClient:
    BASE_URL = "https://api.tiingo.com"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=30),
        retry=lambda retry_state: (
            retry_state.outcome is not None
            and retry_state.outcome.failed
            and not isinstance(retry_state.outcome.exception(), httpx.HTTPStatusError)
        ),
    )
    async def _get(self, path: str, params: dict | None = None) -> dict | list:
        resp = await self._client.get(path, params=params)
        resp.raise_for_status()
        return resp.json()

    async def get_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        frequency: str = "daily",
    ) -> list[dict]:
        return await self._get(
            f"/tiingo/daily/{ticker}/prices",
            params={
                "startDate": start_date,
                "endDate": end_date,
                "resampleFreq": frequency,
            },
        )

    async def get_current_price(self, ticker: str) -> float | None:
        data = await self._get(f"/iex/{ticker}")
        if data and isinstance(data, list) and len(data) > 0:
            return data[0].get("last") or data[0].get("tngoLast")
        return None

    async def get_fundamentals_daily(self, ticker: str) -> dict | None:
        try:
            data = await self._get(
                f"/tiingo/fundamentals/{ticker}/daily",
                params={"limit": 1},
            )
            if data and isinstance(data, list) and len(data) > 0:
                return data[0]
        except httpx.HTTPStatusError:
            logger.warning("tiingo.fundamentals_daily_failed", ticker=ticker)
        return None

    async def get_fundamentals_statements(self, ticker: str) -> list[dict]:
        try:
            data = await self._get(
                f"/tiingo/fundamentals/{ticker}/statements",
                params={"limit": 4},
            )
            return data if isinstance(data, list) else []
        except httpx.HTTPStatusError:
            logger.warning("tiingo.fundamentals_statements_failed", ticker=ticker)
            return []

    async def get_metadata(self, ticker: str) -> dict | None:
        try:
            data = await self._get(f"/tiingo/daily/{ticker}")
            return data if isinstance(data, dict) else None
        except httpx.HTTPStatusError:
            logger.warning("tiingo.metadata_failed", ticker=ticker)
            return None

    async def enrich_ticker(self, ticker: str) -> EnrichmentSchema:
        today = datetime.utcnow()
        start_90d = (today - timedelta(days=120)).strftime("%Y-%m-%d")
        start_1y = (today - timedelta(days=400)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")

        # Sequential with delay to respect free-tier rate limits
        prices_90d = await self.get_price_history(ticker, start_90d, end)
        await asyncio.sleep(1.0)
        prices_1y = await self.get_price_history(ticker, start_1y, end)
        await asyncio.sleep(1.0)
        fundamentals = await self.get_fundamentals_daily(ticker)
        await asyncio.sleep(1.0)
        metadata = await self.get_metadata(ticker)

        # Current price
        closes_90d = [p["adjClose"] for p in prices_90d if p.get("adjClose")]
        closes_1y = [p["adjClose"] for p in prices_1y if p.get("adjClose")]
        volumes = [p["volume"] for p in prices_90d if p.get("volume")]

        current_price = closes_90d[-1] if closes_90d else None

        # Momentum
        momentum_30d = None
        if len(closes_90d) >= 22:
            momentum_30d = (closes_90d[-1] - closes_90d[-22]) / closes_90d[-22]

        momentum_90d = None
        if len(closes_90d) >= 63:
            momentum_90d = (closes_90d[-1] - closes_90d[-63]) / closes_90d[-63]

        # RSI
        rsi = _compute_rsi(closes_90d)

        # Drawdown from 52-week high
        drawdown = None
        if closes_1y:
            high_52w = max(closes_1y)
            if high_52w > 0 and current_price:
                drawdown = (high_52w - current_price) / high_52w

        # Average volume
        avg_volume = sum(volumes[-30:]) / len(volumes[-30:]) if volumes else None

        # Fundamentals
        pe_ratio = fundamentals.get("peRatio") if fundamentals else None
        market_cap = fundamentals.get("marketCap") if fundamentals else None

        # EPS from fundamentals daily endpoint
        eps_latest = fundamentals.get("epsTTM") if fundamentals else None
        eps_growth_yoy = None
        revenue_growth_yoy = None

        # Sector from metadata
        sector = None
        if metadata:
            sector = metadata.get("sector") or metadata.get("industry")

        return EnrichmentSchema(
            pe_ratio=pe_ratio,
            market_cap=market_cap,
            revenue_growth_yoy=revenue_growth_yoy,
            eps_latest=eps_latest,
            eps_growth_yoy=eps_growth_yoy,
            price_at_signal=current_price,
            momentum_30d=momentum_30d,
            momentum_90d=momentum_90d,
            rsi_14d=rsi,
            drawdown_from_52w_high=drawdown,
            avg_volume_30d=avg_volume,
            sector=sector,
        )

    async def close(self):
        await self._client.aclose()
