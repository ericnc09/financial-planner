"""
Congressional trading data client.

Scrapes Capitol Trades (capitoltrades.com) for both House and Senate trades.
Data is embedded in Next.js RSC payloads — parsed from the HTML response.

Zero cost, no API key required.
"""

import json
import re
from datetime import datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import Direction, SmartMoneyEventSchema, SourceType

logger = structlog.get_logger()


class CongressClient:
    """Fetches congressional trading data from Capitol Trades."""

    BASE_URL = "https://www.capitoltrades.com/trades"

    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30.0,
            follow_redirects=True,
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _fetch_page(self, page: int = 1) -> list[dict]:
        """
        Fetch a page from Capitol Trades and extract trade objects
        from the Next.js RSC payload embedded in the HTML.
        """
        url = f"{self.BASE_URL}?page={page}"
        resp = await self._client.get(url)
        resp.raise_for_status()

        trades = self._extract_trades_from_rsc(resp.text)
        return trades

    def _extract_trades_from_rsc(self, html: str) -> list[dict]:
        """
        Capitol Trades uses Next.js RSC streaming. Trade data lives in
        self.__next_f.push([1,"..."]) script chunks as escaped JSON.
        """
        chunks = re.findall(
            r'self\.__next_f\.push\(\[1,"(.*?)"\]\)', html, re.DOTALL
        )

        trades = []
        for chunk in chunks:
            if "txType" not in chunk:
                continue

            # Decode the escaped RSC string
            try:
                decoded = json.loads('"' + chunk + '"')
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            # Extract trade objects: {_issuerId, _politicianId, _txId, ...value}
            trade_pattern = (
                r'\{"_issuerId":\d+,"_politicianId":"[A-Z]\d+",'
                r'"_txId":\d+,.*?"value":\d+\}'
            )
            raw_matches = re.findall(trade_pattern, decoded)

            for raw in raw_matches:
                try:
                    trade = json.loads(raw)
                    trades.append(trade)
                except json.JSONDecodeError:
                    continue

        return trades

    def _parse_trade(self, trade: dict) -> SmartMoneyEventSchema | None:
        """Convert a Capitol Trades object to SmartMoneyEventSchema."""
        try:
            issuer = trade.get("issuer", {})
            ticker = (issuer.get("issuerTicker") or "").strip().upper()

            # Skip if no ticker (bonds, funds without tickers, etc.)
            if not ticker or len(ticker) > 10 or ":" in ticker:
                # Capitol Trades sometimes uses "TICKER:EXCHANGE" format
                if ":" in ticker:
                    ticker = ticker.split(":")[0]
                if not ticker:
                    return None

            tx_type = (trade.get("txType") or "").lower()
            if tx_type in ("buy", "purchase"):
                direction = Direction.BUY
            elif tx_type in ("sell", "sale", "sale_full", "sale_partial"):
                direction = Direction.SELL
            else:
                return None

            tx_date_str = trade.get("txDate")
            if not tx_date_str:
                return None
            trade_date = datetime.strptime(tx_date_str, "%Y-%m-%d")

            pub_date_str = trade.get("pubDate")
            disclosure_date = None
            if pub_date_str:
                try:
                    disclosure_date = datetime.fromisoformat(
                        pub_date_str.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except ValueError:
                    pass

            value = trade.get("value")
            size_lower = float(value) if value else None
            size_upper = size_lower

            politician = trade.get("politician", {})
            chamber = trade.get("chamber", "")
            prefix = "Sen." if chamber == "senate" else "Rep."
            first = politician.get("firstName", "")
            last = politician.get("lastName", "")
            party = (politician.get("party") or "")[:1].upper()
            state = (trade.get("_stateId") or politician.get("_stateId") or "").upper()

            actor_parts = [prefix, first, last]
            if party or state:
                actor_parts.append(f"({party}-{state})" if party and state else f"({party or state})")
            actor = " ".join(actor_parts).strip()

            return SmartMoneyEventSchema(
                ticker=ticker,
                actor=actor,
                direction=direction,
                size_lower=size_lower,
                size_upper=size_upper,
                trade_date=trade_date,
                disclosure_date=disclosure_date,
                source_type=SourceType.CONGRESSIONAL,
                raw_payload=json.dumps(trade, default=str),
            )
        except Exception as e:
            logger.debug("congress.parse_trade_failed", error=str(e))
            return None

    async def get_all_congressional_trades(
        self, since_days: int = 14, max_pages: int = 5
    ) -> list[SmartMoneyEventSchema]:
        """
        Fetch congressional trades from Capitol Trades.
        Paginates up to max_pages to get recent data.
        """
        cutoff = datetime.utcnow() - timedelta(days=since_days)
        all_events = []

        for page in range(1, max_pages + 1):
            try:
                trades = await self._fetch_page(page)
                if not trades:
                    break

                page_events = []
                for trade in trades:
                    event = self._parse_trade(trade)
                    if event and event.trade_date >= cutoff:
                        page_events.append(event)

                all_events.extend(page_events)
                logger.debug(
                    "congress.page_fetched",
                    page=page,
                    raw=len(trades),
                    parsed=len(page_events),
                )

                # If all trades on this page are before cutoff, stop paginating
                if trades and not page_events:
                    break

            except Exception as e:
                logger.warning(
                    "congress.page_fetch_failed", page=page, error=str(e)
                )
                break

        # Deduplicate
        seen = set()
        deduped = []
        for e in all_events:
            key = (e.ticker, e.actor, e.trade_date.date(), e.direction)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        deduped.sort(key=lambda e: e.trade_date, reverse=True)
        logger.info("congress.all_trades", total=len(deduped))
        return deduped

    async def close(self):
        await self._client.aclose()
