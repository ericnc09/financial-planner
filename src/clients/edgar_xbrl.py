"""
SEC XBRL company facts client — point-in-time fundamentals from
data.sec.gov (free, no key, same UA/rate-limit etiquette as EDGAR).

Replaces the flaky yfinance .info fundamentals with filings-derived values:
  - eps_ttm / eps_growth_yoy      (EarningsPerShareDiluted, quarterly)
  - revenue_growth_yoy            (quarterly revenue vs same quarter LY)

Point-in-time correctness: pass as_of and only facts with filed <= as_of
are used — a backtest can never see a quarter before it was actually filed.
"""

import asyncio
import json
from datetime import datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# Revenue tag priority — US GAAP filers use different concepts
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
]
EPS_TAGS = ["EarningsPerShareDiluted", "EarningsPerShareBasic"]


class EdgarXbrlClient:
    """Fetches point-in-time fundamentals from SEC XBRL company facts."""

    _ticker_map: dict[str, int] | None = None  # TICKER -> CIK (class cache)

    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=30.0,
            follow_redirects=True,
        )
        self._rate_limiter = asyncio.Semaphore(8)
        self._facts_cache: dict[str, tuple[datetime, dict]] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _get(self, url: str) -> httpx.Response:
        async with self._rate_limiter:
            await asyncio.sleep(0.12)  # ~8 req/s max, SEC etiquette
            resp = await self._client.get(url)
            if resp.status_code == 429:
                await asyncio.sleep(5)
                raise Exception("Rate limited by SEC")
            resp.raise_for_status()
            return resp

    async def _load_ticker_map(self) -> dict[str, int]:
        if EdgarXbrlClient._ticker_map is None:
            resp = await self._get(TICKER_MAP_URL)
            raw = resp.json()
            EdgarXbrlClient._ticker_map = {
                v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()
            }
            logger.info("edgar_xbrl.ticker_map_loaded", n=len(EdgarXbrlClient._ticker_map))
        return EdgarXbrlClient._ticker_map

    async def get_cik(self, ticker: str) -> int | None:
        ticker_map = await self._load_ticker_map()
        return ticker_map.get(ticker.upper())

    async def get_fundamentals(
        self, ticker: str, as_of: datetime | None = None
    ) -> dict | None:
        """
        Point-in-time fundamentals for a ticker.

        Args:
            ticker: Stock ticker.
            as_of: Only use facts filed on or before this date.
                   Defaults to now (live mode).

        Returns:
            Dict with eps/revenue metrics or None when unavailable.
        """
        as_of = as_of or datetime.utcnow()
        try:
            cik = await self.get_cik(ticker)
            if cik is None:
                logger.debug("edgar_xbrl.unknown_ticker", ticker=ticker)
                return None

            cache_key = ticker.upper()
            cached = self._facts_cache.get(cache_key)
            if cached and (datetime.utcnow() - cached[0]) < timedelta(hours=12):
                facts = cached[1]
            else:
                resp = await self._get(COMPANY_FACTS_URL.format(cik=cik))
                facts = resp.json()
                self._facts_cache[cache_key] = (datetime.utcnow(), facts)

            gaap = (facts.get("facts") or {}).get("us-gaap") or {}

            eps_q = self._quarterly_series(gaap, EPS_TAGS, as_of)
            rev_q = self._quarterly_series(gaap, REVENUE_TAGS, as_of)

            result: dict = {
                "ticker": ticker.upper(),
                "cik": cik,
                "as_of": as_of.isoformat(),
                "source": "sec_xbrl",
            }

            # Q4 rows are often absent (10-Ks report full-year durations), so
            # "last 4 rows" can silently skip a quarter — only sum a TTM when
            # the 4 quarter-ends are actually contiguous (~3 quarters apart).
            def _contiguous(quarters: list[tuple[datetime, float]]) -> bool:
                if len(quarters) < 4:
                    return False
                span = (quarters[-1][0] - quarters[0][0]).days
                return 250 <= span <= 290

            if eps_q:
                result["eps_latest"] = round(eps_q[-1][1], 4)
            if len(eps_q) >= 4 and _contiguous(eps_q[-4:]):
                eps_ttm = sum(v for _, v in eps_q[-4:])
                result["eps_ttm"] = round(eps_ttm, 4)
                if len(eps_q) >= 8 and _contiguous(eps_q[-8:-4]):
                    eps_ttm_prior = sum(v for _, v in eps_q[-8:-4])
                    if abs(eps_ttm_prior) > 1e-9:
                        result["eps_growth_yoy"] = round(
                            (eps_ttm - eps_ttm_prior) / abs(eps_ttm_prior), 4
                        )

            if rev_q:
                latest_end, latest_rev = rev_q[-1]
                result["revenue_latest_q"] = latest_rev
                # Same quarter last year: end date ~365d earlier (±45d window)
                target = latest_end - timedelta(days=365)
                candidates = [
                    (abs((end - target).days), val)
                    for end, val in rev_q
                    if abs((end - target).days) <= 45
                ]
                if candidates:
                    _, rev_ly = min(candidates)
                    if abs(rev_ly) > 1e-9:
                        result["revenue_growth_yoy"] = round(
                            (latest_rev - rev_ly) / abs(rev_ly), 4
                        )

            has_data = any(
                k in result
                for k in ("eps_ttm", "eps_latest", "revenue_growth_yoy")
            )
            if not has_data:
                return None

            logger.info(
                "edgar_xbrl.fundamentals",
                ticker=ticker,
                eps_ttm=result.get("eps_ttm"),
                eps_growth=result.get("eps_growth_yoy"),
                rev_growth=result.get("revenue_growth_yoy"),
            )
            return result

        except Exception as e:
            logger.warning("edgar_xbrl.failed", ticker=ticker, error=str(e))
            return None

    @staticmethod
    def _quarterly_series(
        gaap: dict, tags: list[str], as_of: datetime
    ) -> list[tuple[datetime, float]]:
        """
        Extract a deduplicated quarterly series (end_date, value) for the
        first tag that yields data. Only entries from 10-Q/10-K filings with
        ~quarter-length duration and filed <= as_of are used (point-in-time).
        """
        for tag in tags:
            units = (gaap.get(tag) or {}).get("units") or {}
            # USD for revenue, USD/shares for EPS
            entries = units.get("USD") or units.get("USD/shares") or []
            by_end: dict[str, tuple[datetime, float, datetime]] = {}
            annual: list[tuple[datetime, datetime, float, datetime]] = []  # (start, end, val, filed)
            for e in entries:
                form = e.get("form") or ""
                if form not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                    continue
                start_s, end_s, filed_s = e.get("start"), e.get("end"), e.get("filed")
                val = e.get("val")
                if not (start_s and end_s and filed_s) or val is None:
                    continue
                try:
                    start = datetime.strptime(start_s, "%Y-%m-%d")
                    end = datetime.strptime(end_s, "%Y-%m-%d")
                    filed = datetime.strptime(filed_s, "%Y-%m-%d")
                except ValueError:
                    continue
                # Point-in-time guard
                if filed > as_of:
                    continue
                duration = (end - start).days
                if 80 <= duration <= 100:
                    # Keep earliest filing per quarter-end (original disclosure)
                    existing = by_end.get(end_s)
                    if existing is None or filed < existing[2]:
                        by_end[end_s] = (end, float(val), filed)
                elif 340 <= duration <= 380:
                    annual.append((start, end, float(val), filed))

            # Q4 is rarely tagged with a quarterly duration — the 10-K carries
            # the full-year value instead. Synthesize Q4 = FY − (Q1+Q2+Q3)
            # when the three interim quarters of that fiscal year are present.
            for a_start, a_end, a_val, a_filed in annual:
                if a_end.strftime("%Y-%m-%d") in by_end:
                    continue  # explicit Q4 row exists
                interim = [
                    (end, val) for end, val, _ in by_end.values()
                    if a_start <= end < a_end
                ]
                if len(interim) == 3:
                    q4_val = a_val - sum(v for _, v in interim)
                    by_end[a_end.strftime("%Y-%m-%d")] = (a_end, q4_val, a_filed)

            if len(by_end) >= 1:
                series = sorted(
                    [(end, val) for end, val, _ in by_end.values()],
                    key=lambda x: x[0],
                )
                return series
        return []

    async def close(self):
        await self._client.aclose()
