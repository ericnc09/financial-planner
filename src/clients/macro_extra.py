"""
BLS + Treasury FiscalData clients — enrich the FRED macro layer with
primary-source labor and fiscal data. Both are free; BLS optionally takes
a registration key (raises the daily quota 25 → 500).

BLS series:
  CES0000000001 — total nonfarm payrolls (thousands, SA) → YoY growth %
  LNS14000000   — unemployment rate (%, SA)

Treasury FiscalData:
  avg_interest_rates — average rate on marketable Treasury debt
  debt_to_penny      — total public debt outstanding (USD)
"""

import asyncio
from datetime import datetime

import httpx
import structlog

logger = structlog.get_logger()

USER_AGENT = "SmartMoneyFollows research@smartmoneyfollows.dev"

BLS_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
TREASURY_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"

SERIES_PAYROLLS = "CES0000000001"
SERIES_UNEMPLOYMENT = "LNS14000000"


class MacroExtraClient:
    """Fetches BLS labor data and Treasury fiscal data."""

    def __init__(self, bls_api_key: str | None = None):
        self.bls_api_key = bls_api_key
        self._client = httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT},
            timeout=30.0,
        )

    async def get_snapshot(self) -> dict:
        """All extra macro indicators in one dict (None on per-source failure)."""
        bls_task = self._bls_data()
        rates_task = self._treasury_avg_rate()
        debt_task = self._treasury_debt()
        bls, avg_rate, debt = await asyncio.gather(
            bls_task, rates_task, debt_task, return_exceptions=True
        )

        if isinstance(bls, Exception):
            logger.warning("macro_extra.bls_failed", error=str(bls))
            bls = {}
        if isinstance(avg_rate, Exception):
            logger.warning("macro_extra.treasury_rate_failed", error=str(avg_rate))
            avg_rate = None
        if isinstance(debt, Exception):
            logger.warning("macro_extra.treasury_debt_failed", error=str(debt))
            debt = None

        snapshot = {
            "payrolls_yoy_pct": bls.get("payrolls_yoy_pct"),
            "unemployment_rate": bls.get("unemployment_rate"),
            "treasury_avg_rate": avg_rate,
            "total_public_debt": debt,
            "detail": {
                "payrolls_latest_thousands": bls.get("payrolls_latest"),
                "payrolls_period": bls.get("payrolls_period"),
                "unemployment_period": bls.get("unemployment_period"),
            },
            "as_of": datetime.utcnow().isoformat(),
        }
        logger.info(
            "macro_extra.snapshot",
            payrolls_yoy=snapshot["payrolls_yoy_pct"],
            unemployment=snapshot["unemployment_rate"],
            avg_rate=snapshot["treasury_avg_rate"],
        )
        return snapshot

    # ── BLS ──────────────────────────────────────────────────────────────

    async def _bls_data(self) -> dict:
        year = datetime.utcnow().year
        payload: dict = {
            "seriesid": [SERIES_PAYROLLS, SERIES_UNEMPLOYMENT],
            "startyear": str(year - 2),
            "endyear": str(year),
        }
        if self.bls_api_key:
            payload["registrationkey"] = self.bls_api_key

        resp = await self._client.post(BLS_URL, json=payload)
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "REQUEST_SUCCEEDED":
            logger.warning("macro_extra.bls_status", status=body.get("status"),
                           message=body.get("message"))
            return {}

        out: dict = {}
        for series in body.get("Results", {}).get("series", []):
            sid = series.get("seriesID")
            # Monthly data points, most recent first; skip annual averages (M13)
            points = [
                p for p in series.get("data", [])
                if p.get("period", "").startswith("M") and p["period"] != "M13"
            ]
            if not points:
                continue
            latest = points[0]
            if sid == SERIES_UNEMPLOYMENT:
                out["unemployment_rate"] = float(latest["value"])
                out["unemployment_period"] = f"{latest['year']}-{latest['period']}"
            elif sid == SERIES_PAYROLLS:
                latest_val = float(latest["value"])
                out["payrolls_latest"] = latest_val
                out["payrolls_period"] = f"{latest['year']}-{latest['period']}"
                # Same month one year earlier
                prior = next(
                    (p for p in points
                     if p["period"] == latest["period"]
                     and int(p["year"]) == int(latest["year"]) - 1),
                    None,
                )
                if prior:
                    prior_val = float(prior["value"])
                    if prior_val > 0:
                        out["payrolls_yoy_pct"] = round(
                            (latest_val - prior_val) / prior_val * 100, 3
                        )
        return out

    # ── Treasury FiscalData ──────────────────────────────────────────────

    async def _treasury_avg_rate(self) -> float | None:
        resp = await self._client.get(
            f"{TREASURY_BASE}/v2/accounting/od/avg_interest_rates",
            params={
                "filter": "security_desc:eq:Total Marketable",
                "sort": "-record_date",
                "page[size]": "1",
            },
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            return None
        return float(data[0]["avg_interest_rate_amt"])

    async def _treasury_debt(self) -> float | None:
        resp = await self._client.get(
            f"{TREASURY_BASE}/v2/accounting/od/debt_to_penny",
            params={"sort": "-record_date", "page[size]": "1"},
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        if not data:
            return None
        return float(data[0]["tot_pub_debt_out_amt"])

    async def close(self):
        await self._client.aclose()
