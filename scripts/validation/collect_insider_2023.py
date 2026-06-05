"""
Point-in-time data collector for the 2023 backtest.

Pulls EVERY open-market insider purchase (Form 4, transaction code "P")
filed in 2023 for a fixed, recognized universe (S&P SmallCap 600), straight
from SEC EDGAR. This is the authoritative, public, no-cherry-picking source:
we take the whole universe and the whole year, then let the scoring engine
rank — we never hand-pick tickers.

Why code "P" only: it is a *cash, open-market purchase* — the only Form 4
code that represents a genuine conviction bet with the insider's own money.
Grants (A), option exercises (M) and tax-withholding (F) are compensation,
not signals, so they are excluded.

Why disclosure_date = SEC acceptance date: the public (and therefore our
strategy) cannot act until the filing is actually published. Using the
acceptance date as the entry trigger is what keeps the backtest free of
look-ahead bias.

The scan is resumable: per-ticker results are cached under
reports/cache/edgar_2023/ so a re-run skips finished tickers.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
import xml.etree.ElementTree as ET


class RateLimiter:
    """Global token bucket so all worker threads together stay under SEC's
    10 req/s ceiling (we target 8/s for headroom)."""

    def __init__(self, rate_per_sec: float = 8.0):
        self.min_interval = 1.0 / rate_per_sec
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            sleep = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.min_interval
        if sleep > 0:
            time.sleep(sleep)

UA = "financial-planner-backtest niloyericcosta@gmail.com"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
CACHE_DIR = Path("reports/cache/edgar_2023")
OUT_FILE = Path("reports/cache/insider_2023_buys.json")
YEAR = "2023"

# Open-market purchase only. Everything else is compensation/disposition.
PURCHASE_CODE = "P"


def get_universe() -> list[str]:
    """S&P SmallCap 600 constituents — a recognized, fixed, ex-ante universe.

    Small-caps are where insider open-market buying is both most common and
    most informative (the abnormal-return literature, e.g. Lakonishok & Lee
    2001, finds the signal concentrates in smaller names).
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
    r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=60, follow_redirects=True)
    for t in pd.read_html(r.text):
        cols = [str(c) for c in t.columns]
        for c in cols:
            if "symbol" in c.lower() or "ticker" in c.lower():
                syms = [str(s).strip().upper().replace(".", "-") for s in t[c].tolist()]
                syms = [s for s in syms if s and s != "NAN" and len(s) <= 6]
                if len(syms) > 100:
                    return sorted(set(syms))
    raise RuntimeError("could not parse S&P 600 constituents")


def load_cik_map(client: httpx.Client) -> dict[str, str]:
    r = client.get("https://www.sec.gov/files/company_tickers.json")
    r.raise_for_status()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in r.json().values()}


def raw_xml_url(cik: str, accno: str, primdoc: str) -> str:
    """The submissions API gives the xslF345*/ HTML *viewer* path as
    primaryDocument; stripping that prefix yields the raw Form 4 XML."""
    nodash = accno.replace("-", "")
    pd_ = primdoc
    if pd_.startswith("xsl"):
        pd_ = pd_.split("/", 1)[1]
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nodash}/{pd_}"


def _get(client: httpx.Client, url: str, limiter: "RateLimiter", retries: int = 4) -> httpx.Response | None:
    for attempt in range(retries):
        try:
            limiter.wait()
            resp = client.get(url)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503):
                time.sleep(1.5 * (attempt + 1))
                continue
            return None
        except Exception:
            time.sleep(0.8 * (attempt + 1))
    return None


def parse_form4_purchases(xml_text: str, disclosure_date: str) -> list[dict]:
    """Extract code-P open-market purchases from one Form 4 XML."""
    if "<ownershipDocument>" not in xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    issuer = root.find(".//issuer")
    ticker = ""
    if issuer is not None:
        te = issuer.find("issuerTradingSymbol")
        if te is not None and te.text:
            ticker = te.text.strip().upper()
    if not ticker or len(ticker) > 10:
        return []

    owner_name, owner_title = "Unknown", ""
    oe = root.find(".//reportingOwner")
    if oe is not None:
        ne = oe.find(".//rptOwnerName")
        if ne is not None and ne.text:
            owner_name = ne.text.strip()
        # officer title lives under reportingOwnerRelationship
        for tag in (".//officerTitle", ".//reportingOwnerRelationship/officerTitle"):
            el = oe.find(tag)
            if el is not None and el.text:
                owner_title = el.text.strip()
                break
        # flag directors/officers explicitly for the actor-reputation score
        rel = oe.find(".//reportingOwnerRelationship")
        if rel is not None and not owner_title:
            if (rel.findtext("isDirector") or "").strip() in ("1", "true"):
                owner_title = "Director"
            elif (rel.findtext("isOfficer") or "").strip() in ("1", "true"):
                owner_title = "Officer"

    actor = f"{owner_name} ({owner_title})" if owner_title else owner_name

    out = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code = (tx.findtext(".//transactionCode") or "").strip().upper()
        if code != PURCHASE_CODE:
            continue
        tdate = (tx.findtext(".//transactionDate/value") or "").strip()
        if not tdate:
            continue
        try:
            shares = float(tx.findtext(".//transactionShares/value") or 0)
            price = float(tx.findtext(".//transactionPricePerShare/value") or 0)
        except ValueError:
            shares, price = 0.0, 0.0
        dollar = shares * price if shares and price else None
        out.append({
            "ticker": ticker,
            "actor": actor,
            "direction": "buy",
            "shares": shares,
            "price_per_share": price,
            "dollar_value": dollar,
            "trade_date": tdate,
            "disclosure_date": disclosure_date,  # SEC acceptance/filing date
            "source_type": "insider",
            "transaction_code": code,
        })
    return out


def collect_ticker(client: httpx.Client, ticker: str, cik: str, limiter: "RateLimiter") -> list[dict]:
    cache = CACHE_DIR / f"{ticker}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    events: list[dict] = []
    sub = _get(client, f"https://data.sec.gov/submissions/CIK{cik}.json", limiter)
    if sub is not None:
        rec = sub.json().get("filings", {}).get("recent", {})
        forms = rec.get("form", [])
        dates = rec.get("filingDate", [])
        accs = rec.get("accessionNumber", [])
        docs = rec.get("primaryDocument", [])
        for i, form in enumerate(forms):
            if form != "4" or not str(dates[i]).startswith(YEAR):
                continue
            url = raw_xml_url(cik, accs[i], docs[i])
            resp = _get(client, url, limiter)
            if resp is None:
                continue
            events.extend(parse_form4_purchases(resp.text, str(dates[i])))

    cache.write_text(json.dumps(events))
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="scan only first N tickers (testing)")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    limiter = RateLimiter(rate_per_sec=8.0)
    with httpx.Client(headers=HEADERS, timeout=40, follow_redirects=True) as client:
        cik_map = load_cik_map(client)
        universe = get_universe()
        if args.limit:
            universe = universe[: args.limit]

        print(f"universe={len(universe)} tickers | year={YEAR} | code={PURCHASE_CODE} | 8 workers")
        targets = [(tk, cik_map[tk]) for tk in universe if tk in cik_map]
        missing_cik = len(universe) - len(targets)
        all_events: list[dict] = []
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(collect_ticker, client, tk, cik, limiter): tk
                       for tk, cik in targets}
            for fut in as_completed(futures):
                done += 1
                try:
                    all_events.extend(fut.result())
                except Exception as exc:
                    print(f"  ! {futures[fut]} failed: {exc}")
                if done % 50 == 0 or done == len(targets):
                    print(f"  [{done}/{len(targets)}] running buys={len(all_events)}")

    OUT_FILE.write_text(json.dumps({
        "generated_at": datetime.utcnow().isoformat(),
        "universe": "S&P SmallCap 600 (Wikipedia constituents)",
        "universe_size": len(universe),
        "missing_cik": missing_cik,
        "year": YEAR,
        "transaction_code": PURCHASE_CODE,
        "n_buy_events": len(all_events),
        "events": all_events,
    }, indent=2))
    print(f"\nDONE: {len(all_events)} open-market buy events -> {OUT_FILE}")


if __name__ == "__main__":
    main()
