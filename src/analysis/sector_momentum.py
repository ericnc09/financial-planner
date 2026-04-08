"""
C4 — Sector Momentum Factor.

Aggregates individual ticker analysis results by sector to produce a sector-level
momentum score. "3 insiders bought Energy this week with bullish regime" is more
actionable than any single signal.

Output per sector:
- aggregate buy / sell / hold counts
- avg ensemble score
- sector momentum score (0-100): directional conviction across the sector
- recommendation: overweight / neutral / underweight
"""

from collections import defaultdict

import structlog

logger = structlog.get_logger()

# Map yfinance sector strings to canonical names
SECTOR_ALIASES = {
    "Financial Services": "Financials",
    "Basic Materials": "Materials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
}


def compute_sector_momentum(results: list[dict]) -> dict[str, dict]:
    """
    Compute sector-level aggregates from individual ticker analysis results.

    Args:
        results: List of dicts from analyze_ticker(), each must have:
                 ticker, verdict, buy_score, sell_score, n_models,
                 models.hmm (optional), models.garch (optional).
                 'sector' can be at top level or omitted (→ 'Unknown').

    Returns:
        Dict keyed by sector name, each containing aggregate stats and
        a sector momentum score.
    """
    by_sector: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        raw_sector = r.get("sector") or "Unknown"
        sector = SECTOR_ALIASES.get(raw_sector, raw_sector)
        by_sector[sector].append(r)

    sector_scores: dict[str, dict] = {}

    for sector, tickers in by_sector.items():
        n = len(tickers)
        buys = sum(1 for r in tickers if r["verdict"] == "BUY")
        sells = sum(1 for r in tickers if r["verdict"] == "SELL")
        holds = sum(1 for r in tickers if r["verdict"] == "HOLD")

        avg_buy_score = sum(r["buy_score"] for r in tickers) / n
        avg_sell_score = sum(r["sell_score"] for r in tickers) / n

        # Bull regime fraction from HMM
        hmm_bulls = []
        for r in tickers:
            hmm = (r.get("models") or {}).get("hmm")
            if hmm:
                state = hmm.get("current_state", "")
                if "bull" in state:
                    hmm_bulls.append(1.0)
                elif "bear" in state:
                    hmm_bulls.append(0.0)
                else:
                    hmm_bulls.append(0.5)
        avg_hmm_bull = sum(hmm_bulls) / len(hmm_bulls) if hmm_bulls else 0.5

        # Average GARCH vol
        vols = []
        for r in tickers:
            garch = (r.get("models") or {}).get("garch")
            if garch:
                vols.append(garch.get("current_conditional_vol_annual", 0))
        avg_vol = sum(vols) / len(vols) if vols else None

        # Sector momentum score (0-100):
        # Directional breadth × avg score × regime alignment
        if n == 0:
            momentum_score = 50.0
        else:
            net_direction = (buys - sells) / n          # -1 to +1
            breadth = abs(net_direction)                 # 0 to 1
            direction_sign = 1 if net_direction >= 0 else -1
            base = avg_buy_score if direction_sign >= 0 else avg_sell_score
            regime_boost = 1.0 + (avg_hmm_bull - 0.5) * 0.3  # 0.85–1.15
            momentum_score = min(100, max(0, base * breadth * regime_boost + 50 * (1 - breadth)))

        if buys > sells and momentum_score >= 60:
            recommendation = "overweight"
        elif sells > buys and momentum_score <= 40:
            recommendation = "underweight"
        else:
            recommendation = "neutral"

        sector_scores[sector] = {
            "n_tickers": n,
            "tickers": [r["ticker"] for r in tickers],
            "buy_count": buys,
            "sell_count": sells,
            "hold_count": holds,
            "avg_buy_score": round(avg_buy_score, 1),
            "avg_sell_score": round(avg_sell_score, 1),
            "avg_hmm_bull_prob": round(avg_hmm_bull, 3),
            "avg_annual_vol": round(avg_vol, 3) if avg_vol else None,
            "momentum_score": round(momentum_score, 1),
            "recommendation": recommendation,
        }

        logger.info(
            "sector_momentum.computed",
            sector=sector, n=n, buys=buys, sells=sells,
            score=round(momentum_score, 1), rec=recommendation,
        )

    return dict(sorted(sector_scores.items(), key=lambda x: -x[1]["momentum_score"]))


def format_sector_report(sector_scores: dict[str, dict]) -> str:
    """Return a human-readable sector momentum summary."""
    lines = ["SECTOR MOMENTUM REPORT", "=" * 60]
    for sector, s in sector_scores.items():
        rec_symbol = {"overweight": "↑", "underweight": "↓", "neutral": "→"}.get(s["recommendation"], "")
        lines.append(
            f"{rec_symbol} {sector:<28} Score={s['momentum_score']:>5.1f}  "
            f"B/H/S={s['buy_count']}/{s['hold_count']}/{s['sell_count']}  "
            f"Tickers: {', '.join(s['tickers'])}"
        )
    return "\n".join(lines)
