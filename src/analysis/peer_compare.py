"""
Peer comparison — relative valuation vs same-sector peers.

Given a ticker, finds N sector peers (via a static mapping of S&P 500
large-cap companies by sector — no external API call that could rate-limit),
fetches fundamentals for each via yfinance, and computes relative-valuation
deltas (P/E, P/S, EV/EBITDA, margin gap).

The peer universe is intentionally narrow — top large-caps per sector — to
keep the report compact and avoid noisy comparisons against tiny-float names.
"""

from __future__ import annotations

import asyncio
from statistics import median

import structlog

logger = structlog.get_logger()


# Hand-curated large-cap peer universe per sector. Enough coverage to
# give every signal ticker a reasonable peer group without a bulk API pull.
SECTOR_PEER_UNIVERSE: dict[str, list[str]] = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "CRM",
        "ADBE", "CSCO", "AMD", "INTC", "QCOM", "TXN", "IBM", "INTU",
    ],
    "Communication Services": [
        "GOOGL", "META", "NFLX", "DIS", "T", "VZ", "CMCSA", "TMUS",
    ],
    "Consumer Cyclical": [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG",
    ],
    "Consumer Defensive": [
        "WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "CL",
    ],
    "Financial Services": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP",
        "V", "MA", "BRK-B", "SPGI",
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "MRK", "PFE", "ABBV", "TMO", "DHR", "ABT",
        "AMGN", "BMY", "GILD", "CVS",
    ],
    "Industrials": [
        "CAT", "GE", "HON", "UPS", "RTX", "BA", "LMT", "DE", "UNP", "MMM",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "SLB", "OXY", "PSX", "VLO", "MPC",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "SRE", "XEL",
    ],
    "Basic Materials": [
        "LIN", "SHW", "APD", "FCX", "NEM", "ECL",
    ],
    "Real Estate": [
        "PLD", "AMT", "EQIX", "CCI", "PSA", "O", "SPG",
    ],
}


async def compare_to_peers(
    ticker: str,
    sector: str | None = None,
    n_peers: int = 5,
) -> dict | None:
    """
    Fetch fundamentals for ticker + N sector peers, return a relative-valuation table.

    Args:
        ticker: Target ticker (uppercase).
        sector: Known sector. If None, yfinance .info['sector'] is used.
        n_peers: Max peers to include (after filtering).

    Returns:
        Dict with ticker_row, peer_rows, sector_medians, and deltas. None if
        sector unknown or no peers found.
    """
    return await asyncio.to_thread(_compare_sync, ticker.upper(), sector, n_peers)


def _compare_sync(ticker: str, sector: str | None, n_peers: int) -> dict | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("peer_compare.no_yfinance")
        return None

    target = _fetch_row(ticker)
    if not target:
        logger.warning("peer_compare.target_unavailable", ticker=ticker)
        return None

    resolved_sector = sector or target.get("sector")
    if not resolved_sector:
        logger.info("peer_compare.no_sector", ticker=ticker)
        return None

    universe = SECTOR_PEER_UNIVERSE.get(resolved_sector, [])
    peer_candidates = [p for p in universe if p.upper() != ticker.upper()]
    if not peer_candidates:
        logger.info("peer_compare.no_universe", sector=resolved_sector)
        return None

    peer_rows: list[dict] = []
    for peer in peer_candidates:
        if len(peer_rows) >= n_peers:
            break
        row = _fetch_row(peer)
        if row and row.get("market_cap"):
            peer_rows.append(row)

    if not peer_rows:
        return None

    # Sort peers by market cap descending for a stable display order
    peer_rows.sort(key=lambda r: r.get("market_cap") or 0, reverse=True)

    medians = _sector_medians(peer_rows)
    deltas = _compute_deltas(target, medians)

    return {
        "ticker": ticker,
        "sector": resolved_sector,
        "target": target,
        "peers": peer_rows,
        "sector_medians": medians,
        "deltas": deltas,
        "n_peers": len(peer_rows),
    }


def _fetch_row(ticker: str) -> dict | None:
    """Pull the essential valuation fields for one ticker."""
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        logger.debug("peer_compare.fetch_failed", ticker=ticker, error=str(e))
        return None

    pe = info.get("trailingPE") or info.get("forwardPE")
    ps = info.get("priceToSalesTrailing12Months")
    ev_ebitda = info.get("enterpriseToEbitda")
    gross_margin = info.get("grossMargins")
    operating_margin = info.get("operatingMargins")
    market_cap = info.get("marketCap")
    sector = info.get("sector")
    price = info.get("currentPrice") or info.get("regularMarketPrice")

    # Need at least one valuation metric to be useful
    if pe is None and ps is None and ev_ebitda is None:
        return None

    return {
        "ticker": ticker.upper(),
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": sector,
        "market_cap": float(market_cap) if market_cap else None,
        "price": float(price) if price else None,
        "pe": _safe_float(pe),
        "ps": _safe_float(ps),
        "ev_ebitda": _safe_float(ev_ebitda),
        "gross_margin": _safe_float(gross_margin),
        "operating_margin": _safe_float(operating_margin),
    }


def _safe_float(x) -> float | None:
    try:
        f = float(x)
        if f != f:  # NaN check
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def _sector_medians(peer_rows: list[dict]) -> dict:
    """Median fundamental metrics across peer rows."""
    medians = {}
    for field in ("pe", "ps", "ev_ebitda", "gross_margin", "operating_margin"):
        values = [r[field] for r in peer_rows if r.get(field) is not None]
        medians[field] = round(median(values), 4) if values else None
    return medians


def _compute_deltas(target: dict, medians: dict) -> dict:
    """
    Compute relative-valuation deltas: target vs sector median.

    For ratios (P/E, P/S, EV/EBITDA): delta_pct = (target - median) / median.
      Positive → target trades at a premium to sector.
    For margins: delta_bps = (target - median) * 10000.
      Positive → target is more profitable than sector.
    """
    deltas = {}
    for key in ("pe", "ps", "ev_ebitda"):
        t = target.get(key)
        m = medians.get(key)
        if t is not None and m is not None and m != 0:
            deltas[f"{key}_delta_pct"] = round((t - m) / m * 100, 1)
        else:
            deltas[f"{key}_delta_pct"] = None
    for key in ("gross_margin", "operating_margin"):
        t = target.get(key)
        m = medians.get(key)
        if t is not None and m is not None:
            deltas[f"{key}_delta_bps"] = round((t - m) * 10000, 0)
        else:
            deltas[f"{key}_delta_bps"] = None
    return deltas


def format_peer_table(comparison: dict) -> str:
    """Render the peer comparison as a markdown table."""
    if not comparison:
        return "_No peer comparison available._"

    target = comparison["target"]
    peers = comparison["peers"]
    medians = comparison["sector_medians"]
    deltas = comparison["deltas"]

    def _fmt_ratio(v):
        return f"{v:.1f}" if v is not None else "—"

    def _fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "—"

    lines = [
        f"**Sector:** {comparison['sector']}  |  **Peers compared:** {len(peers)}",
        "",
        "| Ticker | Mkt Cap | P/E | P/S | EV/EBITDA | Gross Mgn | Op Mgn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def _fmt_mcap(v):
        if v is None:
            return "—"
        if v >= 1e12:
            return f"${v/1e12:.2f}T"
        if v >= 1e9:
            return f"${v/1e9:.1f}B"
        return f"${v/1e6:.0f}M"

    lines.append(
        f"| **{target['ticker']}** | {_fmt_mcap(target.get('market_cap'))} "
        f"| {_fmt_ratio(target.get('pe'))} "
        f"| {_fmt_ratio(target.get('ps'))} "
        f"| {_fmt_ratio(target.get('ev_ebitda'))} "
        f"| {_fmt_pct(target.get('gross_margin'))} "
        f"| {_fmt_pct(target.get('operating_margin'))} |"
    )
    for p in peers:
        lines.append(
            f"| {p['ticker']} | {_fmt_mcap(p.get('market_cap'))} "
            f"| {_fmt_ratio(p.get('pe'))} "
            f"| {_fmt_ratio(p.get('ps'))} "
            f"| {_fmt_ratio(p.get('ev_ebitda'))} "
            f"| {_fmt_pct(p.get('gross_margin'))} "
            f"| {_fmt_pct(p.get('operating_margin'))} |"
        )
    lines.append(
        f"| _Sector median_ | — "
        f"| {_fmt_ratio(medians.get('pe'))} "
        f"| {_fmt_ratio(medians.get('ps'))} "
        f"| {_fmt_ratio(medians.get('ev_ebitda'))} "
        f"| {_fmt_pct(medians.get('gross_margin'))} "
        f"| {_fmt_pct(medians.get('operating_margin'))} |"
    )

    lines.append("")
    lines.append("**Relative valuation (vs. sector median):**")

    def _fmt_delta_pct(key, label):
        d = deltas.get(f"{key}_delta_pct")
        if d is None:
            return f"- {label}: not available"
        adj = "premium" if d > 0 else "discount"
        return f"- {label}: **{d:+.1f}%** {adj} to sector"

    def _fmt_delta_bps(key, label):
        d = deltas.get(f"{key}_delta_bps")
        if d is None:
            return f"- {label}: not available"
        adj = "richer" if d > 0 else "leaner"
        return f"- {label}: **{d:+.0f} bps** {adj} than sector"

    lines.append(_fmt_delta_pct("pe", "P/E"))
    lines.append(_fmt_delta_pct("ps", "P/S"))
    lines.append(_fmt_delta_pct("ev_ebitda", "EV/EBITDA"))
    lines.append(_fmt_delta_bps("gross_margin", "Gross margin"))
    lines.append(_fmt_delta_bps("operating_margin", "Operating margin"))

    return "\n".join(lines)
