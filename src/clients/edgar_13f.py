"""
SEC 13F-HR institutional holdings client — a quarterly "smart money" signal
source from the filings of curated, well-known funds.

For each tracked fund:
  1. Find its two most recent 13F-HR filings (data.sec.gov submissions API)
  2. Parse both information tables (holdings XML)
  3. Diff: new positions and large adds (≥25% share increase) → BUY events;
     full exits → SELL events

Timing semantics (no lookahead): trade_date = report period end (the quarter
during which the fund actually traded), disclosure_date = filing date (when
the information became public — up to 45 days later). Anything consuming
these events for "followable alpha" must anchor on disclosure_date.

Free, no key. Same SEC UA/rate-limit etiquette as the other EDGAR clients.
"""

import asyncio
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.schemas import Direction, SmartMoneyEventSchema, SourceType

logger = structlog.get_logger()

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"

# Curated funds worth following (verified against SEC submissions API)
TRACKED_FUNDS: dict[str, int] = {
    "Berkshire Hathaway": 1067983,
    "Pershing Square Capital": 1336528,
    "Scion Asset Management": 1649339,
    "Bridgewater Associates": 1350694,
    "Renaissance Technologies": 1037389,
    "Tiger Global Management": 1167483,
    "Appaloosa Management": 1656456,
    "Third Point": 1040273,
}

# Share increase ratio that counts as a meaningful ADD
ADD_THRESHOLD = 1.25

_CORP_SUFFIX_RE = re.compile(
    r"\b(incorporated|corporation|company|holdings?|group|industries|"
    r"international|technologies|inc|corp|co|ltd|plc|sa|nv|lp|llc|"
    r"cl\s*[abc]|class\s*[abc]|com|new|del)\b\.?",
    re.I,
)


def _normalize_issuer(name: str) -> str:
    """Normalize an issuer/company name for ticker matching."""
    n = name.upper()
    n = re.sub(r"[^A-Z0-9 ]", " ", n)
    n = _CORP_SUFFIX_RE.sub(" ", n)
    return re.sub(r"\s+", " ", n).strip()


class Edgar13FClient:
    """Emits SmartMoneyEventSchema events from curated funds' 13F diffs."""

    _name_to_ticker: dict[str, str] | None = None  # class-level cache

    def __init__(self, funds: dict[str, int] | None = None):
        self.funds = funds or TRACKED_FUNDS
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
            follow_redirects=True,
        )
        self._rate_limiter = asyncio.Semaphore(8)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=15))
    async def _get(self, url: str) -> httpx.Response:
        async with self._rate_limiter:
            await asyncio.sleep(0.12)
            resp = await self._client.get(url)
            if resp.status_code == 429:
                await asyncio.sleep(5)
                raise Exception("Rate limited by SEC")
            resp.raise_for_status()
            return resp

    async def _load_name_map(self) -> dict[str, str]:
        """Normalized company title -> ticker (for issuer-name matching)."""
        if Edgar13FClient._name_to_ticker is None:
            resp = await self._get(TICKER_MAP_URL)
            raw = resp.json()
            mapping: dict[str, str] = {}
            for v in raw.values():
                norm = _normalize_issuer(v["title"])
                # Prefer the first (lowest file-number = usually primary) listing
                if norm and norm not in mapping:
                    mapping[norm] = v["ticker"].upper()
            Edgar13FClient._name_to_ticker = mapping
            logger.info("edgar_13f.name_map_loaded", n=len(mapping))
        return Edgar13FClient._name_to_ticker

    async def get_recent_13f_events(self) -> list[SmartMoneyEventSchema]:
        """Diff the two latest 13F-HR filings of every tracked fund."""
        name_map = await self._load_name_map()
        all_events: list[SmartMoneyEventSchema] = []

        for fund_name, cik in self.funds.items():
            try:
                events = await self._fund_events(fund_name, cik, name_map)
                all_events.extend(events)
            except Exception as e:
                logger.warning("edgar_13f.fund_failed", fund=fund_name, error=str(e))

        logger.info("edgar_13f.events", n=len(all_events))
        return all_events

    async def _fund_events(
        self, fund_name: str, cik: int, name_map: dict[str, str]
    ) -> list[SmartMoneyEventSchema]:
        resp = await self._get(SUBMISSIONS_URL.format(cik=cik))
        sub = resp.json()
        entity_name = sub.get("name", fund_name)

        recent = sub.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        report_dates = recent.get("reportDate", [])
        filing_dates = recent.get("filingDate", [])

        filings_13f = [
            (accessions[i], report_dates[i], filing_dates[i])
            for i, f in enumerate(forms)
            if f == "13F-HR"
        ][:2]

        if len(filings_13f) < 2:
            logger.debug("edgar_13f.not_enough_filings", fund=fund_name)
            return []

        (acc_new, period_new, filed_new), (acc_old, _, _) = filings_13f

        holdings_new = await self._holdings(cik, acc_new)
        holdings_old = await self._holdings(cik, acc_old)
        if not holdings_new or not holdings_old:
            return []

        period_dt = datetime.strptime(period_new, "%Y-%m-%d")
        filed_dt = datetime.strptime(filed_new, "%Y-%m-%d")

        events = []
        # New positions + large adds → BUY
        for cusip, pos in holdings_new.items():
            old = holdings_old.get(cusip)
            if old is None:
                action = "new_position"
            elif old["shares"] > 0 and pos["shares"] / old["shares"] >= ADD_THRESHOLD:
                action = "added"
            else:
                continue
            ev = self._make_event(
                pos, entity_name, Direction.BUY, action,
                period_dt, filed_dt, name_map,
            )
            if ev:
                events.append(ev)

        # Full exits → SELL
        for cusip, pos in holdings_old.items():
            if cusip not in holdings_new:
                ev = self._make_event(
                    pos, entity_name, Direction.SELL, "exited",
                    period_dt, filed_dt, name_map,
                )
                if ev:
                    events.append(ev)

        logger.info(
            "edgar_13f.fund_diffed",
            fund=entity_name, period=period_new,
            n_events=len(events),
            n_holdings=len(holdings_new),
        )
        return events

    def _make_event(
        self,
        pos: dict,
        fund_name: str,
        direction: Direction,
        action: str,
        period_dt: datetime,
        filed_dt: datetime,
        name_map: dict[str, str],
    ) -> SmartMoneyEventSchema | None:
        ticker = name_map.get(_normalize_issuer(pos["issuer"]))
        if not ticker:
            return None  # unmappable issuer (foreign, OTC, renamed) — skip
        value = pos["value"]
        return SmartMoneyEventSchema(
            ticker=ticker,
            actor=fund_name,
            direction=direction,
            size_lower=value,
            size_upper=value,
            trade_date=period_dt,
            disclosure_date=filed_dt,
            source_type=SourceType.INSTITUTIONAL,
            raw_payload=json.dumps({
                "source": "sec_13f",
                "action": action,
                "issuer": pos["issuer"],
                "cusip": pos["cusip"],
                "shares": pos["shares"],
                "value_usd": value,
            }),
        )

    async def _holdings(self, cik: int, accession: str) -> dict[str, dict]:
        """Fetch and parse a 13F information table → {cusip: position}."""
        acc_nodash = accession.replace("-", "")
        index_url = ARCHIVES_BASE.format(cik=cik, accession=acc_nodash) + "/"
        resp = await self._get(index_url)

        # The info table is an XML doc, usually named *infotable*.xml or
        # similar; exclude the primary_doc.xml (cover page).
        xml_files = re.findall(r'href="([^"]+\.xml)"', resp.text, flags=re.I)
        candidates = [
            f for f in xml_files
            if "primary_doc" not in f.lower() and "xsl" not in f.lower()
        ]
        if not candidates:
            return {}
        info_url = candidates[0]
        if info_url.startswith("/"):
            info_url = f"https://www.sec.gov{info_url}"
        elif not info_url.startswith("http"):
            info_url = index_url + info_url

        resp = await self._get(info_url)
        return self._parse_info_table(resp.text)

    @staticmethod
    def _parse_info_table(xml_text: str) -> dict[str, dict]:
        """Parse 13F info table XML. Values are USD (whole dollars post-2023)."""
        try:
            # Strip namespaces for resilient parsing across filers
            xml_clean = re.sub(r'xmlns(:\w+)?="[^"]*"', "", xml_text)
            xml_clean = re.sub(r"<(/?)\w+:", r"<\1", xml_clean)
            root = ET.fromstring(xml_clean)
        except ET.ParseError:
            return {}

        holdings: dict[str, dict] = {}
        for info in root.iter("infoTable"):
            issuer = (info.findtext("nameOfIssuer") or "").strip()
            cusip = (info.findtext("cusip") or "").strip()
            value_s = (info.findtext("value") or "0").strip()
            shares_s = (info.findtext("shrsOrPrnAmt/sshPrnamt") or "0").strip()
            put_call = (info.findtext("putCall") or "").strip()
            if not issuer or not cusip:
                continue
            if put_call:
                continue  # skip options positions — equity holdings only
            try:
                value = float(value_s)
                shares = float(shares_s)
            except ValueError:
                continue
            # Aggregate duplicate CUSIP rows (multiple discretion buckets)
            if cusip in holdings:
                holdings[cusip]["value"] += value
                holdings[cusip]["shares"] += shares
            else:
                holdings[cusip] = {
                    "issuer": issuer, "cusip": cusip,
                    "value": value, "shares": shares,
                }
        return holdings

    async def close(self):
        await self._client.aclose()
