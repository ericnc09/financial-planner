"""
Ticker symbol normalization.

Filings write the same security many ways ("BRK/B", "BRK.B", "NYSE: VTEX",
"ARES.PRB"), while the price source wants one spelling ("BRK-B", "VTEX",
"ARES-PB"). A symbol the price source cannot resolve is NOT evidence that
the company was delisted, so everything that looks up prices goes through
here first.
"""

import re

# "NYSE: VTEX", "NASDAQ:AAPL", "NYSE American: XYZ"
_EXCHANGE_PREFIX = re.compile(
    r"^\s*(NYSE(\s+(AMERICAN|ARCA|MKT))?|NASDAQ(\s*(GS|GM|CM))?|AMEX|ARCA|BATS|CBOE|OTC(QB|QX)?|TSX|LSE)\s*[:\-]\s*",
    re.IGNORECASE,
)
_VALID = re.compile(r"^[\^A-Z0-9][A-Z0-9.\-=]*$")


def normalize_symbol(raw: str | None) -> str | None:
    """Canonical stored form of a ticker, or None if it is not a ticker.

    Strips exchange prefixes and whitespace, upper-cases, and turns the
    share-class slash into a dot ("BRK/B" -> "BRK.B").
    """
    if not raw:
        return None
    symbol = _EXCHANGE_PREFIX.sub("", str(raw)).strip().upper()
    symbol = symbol.replace(" ", "").replace("/", ".")
    if not symbol or len(symbol) > 10 or not _VALID.match(symbol):
        return None
    return symbol


def yahoo_candidates(raw: str | None) -> list[str]:
    """Yahoo Finance spellings to try for a ticker, most likely first.

    Yahoo writes share classes with a hyphen (BRK-B) and preferred series as
    "-P<series>" (ARES-PB). The last-resort candidate splits a five-letter
    symbol ending in A or B ("UHALB" -> "UHAL-B"); it is only tried after the
    literal symbol has returned no data.
    """
    symbol = normalize_symbol(raw)
    if symbol is None:
        return []
    if symbol.startswith("^") or "=" in symbol:
        return [symbol]

    candidates: list[str] = []

    def add(s: str):
        if s and s not in candidates:
            candidates.append(s)

    preferred = re.match(r"^([A-Z]+)[.\-]PR?\.?([A-Z])$", symbol)
    if preferred:  # ARES.PRB / ARES-PB / ARES.PB
        add(f"{preferred.group(1)}-P{preferred.group(2)}")

    add(symbol.replace(".", "-"))

    share_class = re.match(r"^([A-Z]+)[.\-]([A-Z])$", symbol)
    if share_class and not preferred:
        # "AMH-H" may be a share class or preferred series H
        add(f"{share_class.group(1)}-P{share_class.group(2)}")

    if re.match(r"^[A-Z]{4}[AB]$", symbol):
        add(f"{symbol[:4]}-{symbol[4]}")

    return candidates


_FUND_QUOTE_TYPES = {"ETF", "MUTUALFUND", "INDEX", "CURRENCY", "FUTURE", "OPTION", "CRYPTOCURRENCY"}
_DEBT_OR_PREFERRED_NAME = re.compile(
    r"\d(\.\d+)?\s?%|notes?\s+due|debenture|preferred|depositary", re.IGNORECASE
)


def classify_security(yahoo_symbol: str | None, info: dict | None) -> str:
    """"common", "preferred_or_debt" or "fund_or_other".

    Yahoo reports quoteType "EQUITY" for exchange-traded notes and preferred
    shares too, so the name and symbol are checked as well. Equity factor
    models (Fama-French, market copula) are meaningless for anything but
    common stock.
    """
    info = info or {}
    quote_type = str(info.get("quoteType") or "").upper()
    if quote_type in _FUND_QUOTE_TYPES:
        return "fund_or_other"
    if yahoo_symbol and re.search(r"-P[A-Z]?$", yahoo_symbol.upper()):
        return "preferred_or_debt"
    name = f"{info.get('longName') or ''} {info.get('shortName') or ''}"
    if _DEBT_OR_PREFERRED_NAME.search(name):
        return "preferred_or_debt"
    return "common"
