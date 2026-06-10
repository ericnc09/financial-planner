"""
News client — per-ticker business headlines from Finnhub (primary, requires
free API key) and GDELT 2.0 DOC API (breadth, no key needed).

Every article carries its publication timestamp. Downstream consumers MUST
filter on published_at <= as_of for any backtest (bias guard) — the
NewsSentimentAnalyzer does this automatically.
"""

import asyncio
import re
from datetime import datetime, timedelta

import httpx
import structlog

logger = structlog.get_logger()

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"


class NewsClient:
    """Fetches and normalizes per-ticker news articles.

    Normalized article dict:
        {headline, summary, source, url, published_at: datetime (UTC naive),
         provider: "finnhub" | "gdelt"}
    """

    # GDELT enforces ~1 request / 5 seconds per IP and puts violators in a
    # multi-minute penalty box — throttle globally across all instances.
    _gdelt_lock: asyncio.Lock = asyncio.Lock()
    _gdelt_last_request: float = 0.0
    GDELT_MIN_INTERVAL = 6.0

    def __init__(self, finnhub_api_key: str | None = None):
        self.finnhub_api_key = finnhub_api_key
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
            follow_redirects=True,
        )

    async def get_company_news(
        self,
        ticker: str,
        days: int = 14,
        company_name: str | None = None,
        max_articles: int = 120,
    ) -> list[dict]:
        """Fetch news for a ticker from all available providers, deduplicated
        by normalized headline and sorted by published_at ascending."""
        results = await asyncio.gather(
            self._finnhub_news(ticker, days),
            self._gdelt_news(ticker, days, company_name),
            return_exceptions=True,
        )
        articles: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                logger.debug("news.provider_failed", error=str(r))
                continue
            articles.extend(r)

        # Dedupe by normalized headline
        seen: set[str] = set()
        deduped = []
        for a in sorted(articles, key=lambda x: x["published_at"]):
            key = re.sub(r"[^a-z0-9]", "", a["headline"].lower())[:80]
            if key and key not in seen:
                seen.add(key)
                deduped.append(a)

        deduped = deduped[-max_articles:]
        logger.info(
            "news.fetched", ticker=ticker, n=len(deduped),
            finnhub=bool(self.finnhub_api_key),
        )
        return deduped

    # ── Finnhub (primary — ticker-tagged, clean) ─────────────────────────

    async def _finnhub_news(self, ticker: str, days: int) -> list[dict]:
        if not self.finnhub_api_key:
            return []
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=days)
        resp = await self._client.get(FINNHUB_URL, params={
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": self.finnhub_api_key,
        })
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            return []

        articles = []
        for it in items:
            ts = it.get("datetime")
            headline = (it.get("headline") or "").strip()
            if not ts or not headline:
                continue
            articles.append({
                "headline": headline,
                "summary": (it.get("summary") or "").strip(),
                "source": it.get("source") or "finnhub",
                "url": it.get("url") or "",
                "published_at": datetime.utcfromtimestamp(int(ts)),
                "provider": "finnhub",
            })
        return articles

    # ── GDELT (breadth — keyless global firehose) ────────────────────────

    async def _gdelt_news(
        self, ticker: str, days: int, company_name: str | None
    ) -> list[dict]:
        # Query by company name when we have one (ticker strings alone are
        # noisy in general news), restricted to English business coverage.
        name = (company_name or "").strip()
        if name:
            # Strip corporate suffixes that hurt matching
            name = re.sub(
                r"\b(incorporated|corporation|company|holdings?|group|inc|corp|co|ltd|plc|sa|nv|lp)\.?$",
                "", name, flags=re.I,
            ).strip().rstrip(",")
        subject = name if len(name) >= 4 else ticker
        query = f"{subject} stock sourcelang:english"

        # Global throttle: one GDELT request per GDELT_MIN_INTERVAL seconds
        async with NewsClient._gdelt_lock:
            now = asyncio.get_event_loop().time()
            wait = NewsClient.GDELT_MIN_INTERVAL - (now - NewsClient._gdelt_last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            resp = await self._client.get(GDELT_URL, params={
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": "75",
                "timespan": f"{min(days, 30)}d",
            })
            NewsClient._gdelt_last_request = asyncio.get_event_loop().time()

        if resp.status_code == 429 or "limit requests" in resp.text[:120]:
            # GDELT's penalty box clears only after a quiet period — degrade
            # gracefully rather than retrying (retries extend the penalty).
            logger.warning("news.gdelt_rate_limited", ticker=ticker)
            return []
        resp.raise_for_status()
        try:
            data = resp.json()
        except Exception:
            return []  # GDELT returns plain-text/HTML error pages on bad queries

        articles = []
        for it in data.get("articles", []) or []:
            headline = (it.get("title") or "").strip()
            seendate = it.get("seendate") or ""  # "20260610T120000Z"
            if not headline or not seendate:
                continue
            try:
                published = datetime.strptime(seendate, "%Y%m%dT%H%M%SZ")
            except ValueError:
                continue
            articles.append({
                "headline": headline,
                "summary": "",
                "source": it.get("domain") or "gdelt",
                "url": it.get("url") or "",
                "published_at": published,
                "provider": "gdelt",
            })
        return articles

    async def close(self):
        await self._client.aclose()
