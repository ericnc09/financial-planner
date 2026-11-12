"""
SEC EDGAR client for insider trading data (Form 3/4/5).

Uses the EDGAR daily form indexes to list every Form 4 filed in the look-back
window, then parses each filing's XML for transaction details. Each day's index
is published after that day ends, so same-day filings arrive on the next run.

Free, no API key. Requires User-Agent header and respects 10 req/s rate limit.
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.clients.symbols import normalize_symbol
from src.models.schemas import Direction, SmartMoneyEventSchema, SourceType

logger = structlog.get_logger()

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"

# Open-market transactions only. Awards, option exercises, gifts and tax
# withholding don't reflect a decision to trade at the market price.
BUY_CODES = {"P"}
SELL_CODES = {"S"}

# Filers of unlisted companies put these in the trading symbol field
PLACEHOLDER_TICKERS = {"NONE", "N/A", "NA"}

ARCHIVES_URL = "https://www.sec.gov/Archives"
MIN_REQUEST_INTERVAL = 0.125  # 8 req/s; SEC allows at most 10
MAX_CONCURRENT_FILINGS = 8


class EdgarClient:
    """Fetches insider trades from SEC EDGAR Form 4 filings via daily index + XML parsing."""

    def __init__(self):
        self._client = httpx.AsyncClient(
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/xml, application/json, text/html",
            },
            timeout=30.0,
            follow_redirects=True,
        )
        self._slot_lock = asyncio.Lock()
        self._next_slot = 0.0

    async def _wait_for_request_slot(self):
        """Space requests across all concurrent tasks to stay under SEC's limit."""
        async with self._slot_lock:
            now = asyncio.get_running_loop().time()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + MIN_REQUEST_INTERVAL
        if wait > 0:
            await asyncio.sleep(wait)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        await self._wait_for_request_slot()
        resp = await self._client.get(url, params=params)
        if resp.status_code == 429:
            await asyncio.sleep(5)
            raise Exception("Rate limited by SEC")
        resp.raise_for_status()
        return resp

    @staticmethod
    def _parse_daily_index(text: str) -> dict[str, datetime]:
        """Map each Form 4's archive path to its filing date."""
        filings = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[0] != "4":
                continue
            try:
                filings[parts[-1]] = datetime.strptime(parts[-2], "%Y%m%d")
            except ValueError:
                continue
        return filings

    async def _get_daily_form4s(self, day: date) -> dict[str, datetime]:
        """List one day's Form 4 filings. Days without an index yield nothing."""
        quarter = (day.month - 1) // 3 + 1
        url = (
            f"{ARCHIVES_URL}/edgar/daily-index/{day.year}/QTR{quarter}/"
            f"form.{day:%Y%m%d}.idx"
        )
        try:
            resp = await self._get(url)
        except Exception as e:
            # Expected for holidays and for today, whose index isn't out yet
            logger.warning("edgar.daily_index_unavailable", day=str(day), error=str(e))
            return {}
        return self._parse_daily_index(resp.text)

    async def _get_filing_events(
        self, path: str, filing_date: datetime
    ) -> list[SmartMoneyEventSchema] | None:
        """Fetch one filing and parse it. Returns None if it can't be fetched."""
        try:
            resp = await self._get(f"{ARCHIVES_URL}/{path}")
        except Exception as e:
            logger.debug("edgar.filing_fetch_failed", path=path, error=str(e))
            return None
        return self._parse_form4_xml(resp.text, filing_date)

    def _parse_form4_xml(
        self, content: str, filing_date: datetime
    ) -> list[SmartMoneyEventSchema]:
        """Parse a Form 4 submission for transaction details."""
        try:
            document = re.search(
                r"<ownershipDocument>.*</ownershipDocument>", content, re.DOTALL
            )
            if not document:
                return []

            root = ET.fromstring(document.group(0))
            events = []

            # Issuer info
            issuer = root.find(".//issuer")
            ticker = ""
            if issuer is not None:
                ticker_elem = issuer.find("issuerTradingSymbol")
                if ticker_elem is not None and ticker_elem.text:
                    ticker = ticker_elem.text.strip().upper()
            if not ticker or ticker in PLACEHOLDER_TICKERS:
                return []
            # "NYSE: VTEX" / "BRK/B" -> one canonical spelling, so the same
            # security always lands under the same ticker.
            ticker = normalize_symbol(ticker)
            if not ticker:
                return []

            # Reporting owner
            owner_name = "Unknown"
            owner_title = ""
            owner_elem = root.find(".//reportingOwner")
            if owner_elem is not None:
                name_elem = owner_elem.find(".//rptOwnerName")
                if name_elem is not None and name_elem.text:
                    owner_name = name_elem.text.strip()
                title_elem = owner_elem.find(".//officerTitle")
                if title_elem is not None and title_elem.text:
                    owner_title = title_elem.text.strip()

            actor = f"{owner_name} ({owner_title})" if owner_title else owner_name

            # Parse non-derivative transactions
            for tx in root.findall(".//nonDerivativeTransaction"):
                event = self._parse_transaction(tx, ticker, actor, filing_date)
                if event:
                    events.append(event)

            # Parse derivative transactions
            for tx in root.findall(".//derivativeTransaction"):
                event = self._parse_transaction(tx, ticker, actor, filing_date)
                if event:
                    events.append(event)

            return events

        except ET.ParseError:
            return []
        except Exception as e:
            logger.debug("edgar.form4_parse_failed", error=str(e))
            return []

    def _parse_transaction(
        self,
        tx_elem,
        ticker: str,
        actor: str,
        filing_date: datetime,
    ) -> SmartMoneyEventSchema | None:
        try:
            code_elem = tx_elem.find(".//transactionCode")
            if code_elem is None or not code_elem.text:
                return None
            code = code_elem.text.strip().upper()

            if code in BUY_CODES:
                direction = Direction.BUY
            elif code in SELL_CODES:
                direction = Direction.SELL
            else:
                return None

            # Transaction date — MUST come from the filing, never fall back
            # to filing_date (which introduces look-ahead bias since the
            # filing is published days after the actual trade).
            date_elem = tx_elem.find(".//transactionDate/value")
            if date_elem is None or not date_elem.text:
                logger.debug("edgar.missing_trade_date", filing_date=filing_date)
                return None
            try:
                trade_date = datetime.strptime(
                    date_elem.text.strip(), "%Y-%m-%d"
                )
            except ValueError:
                logger.debug("edgar.invalid_trade_date", raw=date_elem.text)
                return None

            # Shares
            shares_elem = tx_elem.find(".//transactionShares/value")
            shares = 0.0
            if shares_elem is not None and shares_elem.text:
                try:
                    shares = float(shares_elem.text)
                except ValueError:
                    pass

            # Price
            price_elem = tx_elem.find(".//transactionPricePerShare/value")
            price = 0.0
            if price_elem is not None and price_elem.text:
                try:
                    price = float(price_elem.text)
                except ValueError:
                    pass

            dollar_value = shares * price if shares and price else None

            return SmartMoneyEventSchema(
                ticker=ticker,
                actor=actor,
                direction=direction,
                size_lower=dollar_value,
                size_upper=dollar_value,
                trade_date=trade_date,
                disclosure_date=filing_date,
                source_type=SourceType.INSIDER,
                raw_payload=json.dumps(
                    {
                        "transaction_code": code,
                        "shares": shares,
                        "price_per_share": price,
                        "dollar_value": dollar_value,
                        "source": "sec_edgar_form4",
                    }
                ),
            )
        except Exception:
            return None

    async def get_bulk_insider_trades(
        self, since_days: int = 14
    ) -> list[SmartMoneyEventSchema]:
        """
        Get insider trades from every Form 4 filed in the last since_days:
        1. Listing filings from the EDGAR daily form indexes
        2. Fetching each filing
        3. Parsing its XML for open-market transactions
        """
        today = datetime.utcnow().date()
        days = [today - timedelta(days=n) for n in range(since_days + 1)]
        indexes = await asyncio.gather(
            *(self._get_daily_form4s(d) for d in days if d.weekday() < 5)
        )
        filings = {path: filed for index in indexes for path, filed in index.items()}

        limit = asyncio.Semaphore(MAX_CONCURRENT_FILINGS)

        async def fetch(path: str, filed: datetime):
            async with limit:
                return await self._get_filing_events(path, filed)

        results = await asyncio.gather(*(fetch(p, f) for p, f in filings.items()))
        all_events = [e for events in results if events for e in events]
        failed = sum(1 for events in results if events is None)

        # Deduplicate
        seen = set()
        deduped = []
        for e in all_events:
            key = (e.ticker, e.actor, e.trade_date.date(), e.direction)
            if key not in seen:
                seen.add(key)
                deduped.append(e)

        deduped.sort(key=lambda e: e.trade_date, reverse=True)
        logger.info(
            "edgar.bulk_insider_trades",
            total=len(deduped),
            filings=len(filings),
            failed=failed,
        )
        return deduped

    async def close(self):
        await self._client.aclose()
