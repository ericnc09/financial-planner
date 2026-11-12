"""
C5 — Volatility-Adjusted Position Sizing (Risk Parity / Inverse-Vol).

Instead of equal-weight allocation, sizes positions by inverse volatility so each
position contributes equal risk to the portfolio. Drawdown is lower because high-vol
names don't dominate portfolio P&L.

Method:
  weight_i = (1 / vol_i) / sum_j(1 / vol_j)

Where vol_i is the annualised GARCH conditional volatility (or historical 20d vol
if GARCH is unavailable). Only BUY / SELL verdicts are sized; HOLDs are excluded.

Returns:
  List of PositionRecommendation dicts with ticker, direction, weight_pct, and
  rationale.
"""

import numpy as np
import structlog

logger = structlog.get_logger()

# Cap any single position at 30% of portfolio (avoids concentration)
MAX_SINGLE_WEIGHT = 0.30
# Minimum meaningful weight — positions below this are dropped
MIN_WEIGHT = 0.02


def compute_position_sizes(
    results: list[dict],
    portfolio_value: float = 100_000,
    verdicts_to_size: tuple[str, ...] = ("BUY", "SELL"),
) -> list[dict]:
    """
    Compute volatility-adjusted position sizes for actionable verdicts.

    Args:
        results: List of dicts from analyze_ticker().
        portfolio_value: Total portfolio in currency units (used for $ sizing).
        verdicts_to_size: Only size positions with these verdicts.

    Returns:
        List of position dicts sorted by weight descending.
    """
    actionable = [r for r in results if r.get("verdict") in verdicts_to_size]

    if not actionable:
        return []

    # Extract volatility for each ticker
    vols: dict[str, float] = {}
    for r in actionable:
        ticker = r["ticker"]
        # Prefer GARCH conditional vol; fall back to historical from Monte Carlo
        garch = (r.get("models") or {}).get("garch")
        mc = (r.get("models") or {}).get("monte_carlo")

        if garch and garch.get("current_conditional_vol_annual"):
            vol = float(garch["current_conditional_vol_annual"])
        elif mc and mc.get("annual_volatility"):
            vol = float(mc["annual_volatility"])
        else:
            vol = 0.25  # default 25% if no model data
        vols[ticker] = max(vol, 0.05)  # floor at 5% to avoid division issues

    # Inverse-vol weights
    inv_vols = {t: 1.0 / v for t, v in vols.items()}
    total_inv_vol = sum(inv_vols.values())
    raw_weights = {t: iv / total_inv_vol for t, iv in inv_vols.items()}

    # Apply the max-weight cap. Whatever cannot be placed without breaking the
    # cap stays in cash (with fewer than four positions, most of it does).
    normalised = _cap_weights(raw_weights, MAX_SINGLE_WEIGHT)

    # Build output
    positions = []
    for r in actionable:
        ticker = r["ticker"]
        w = normalised.get(ticker, 0.0)
        if w < MIN_WEIGHT:
            continue

        vol_pct = vols[ticker] * 100
        dollar_size = portfolio_value * w
        ensemble_score = r["buy_score"] if r["verdict"] == "BUY" else r["sell_score"]

        positions.append({
            "ticker": ticker,
            "verdict": r["verdict"],
            "weight_pct": round(w * 100, 2),
            "dollar_size": round(dollar_size, 0),
            "vol_annual_pct": round(vol_pct, 1),
            "ensemble_score": round(ensemble_score, 1),
            "rationale": (
                f"Inverse-vol weight: 1/{vol_pct:.0f}% vol → {w*100:.1f}% of portfolio. "
                f"Ensemble score {ensemble_score:.0f}/100."
            ),
        })

    positions.sort(key=lambda x: -x["weight_pct"])

    logger.info(
        "position_sizing.complete",
        n_positions=len(positions),
        portfolio_value=portfolio_value,
        cash_pct=round(100.0 - sum(p["weight_pct"] for p in positions), 2),
        top=positions[0]["ticker"] if positions else None,
    )
    return positions


def _cap_weights(raw: dict[str, float], cap: float) -> dict[str, float]:
    """Cap every weight at `cap`, handing the excess to the uncapped names
    pro rata. Repeats until nothing exceeds the cap. If every name is capped
    the weights sum to less than 1 and the remainder is cash — renormalising
    at that point would simply undo the cap."""
    weights = dict(raw)
    capped: set[str] = set()
    for _ in range(len(weights)):
        over = [t for t, w in weights.items() if t not in capped and w > cap + 1e-12]
        if not over:
            break
        capped.update(over)
        for t in over:
            weights[t] = cap
        free = [t for t in weights if t not in capped]
        free_total = sum(raw[t] for t in free)
        if not free or free_total <= 0:
            break
        remaining = 1.0 - cap * len(capped)
        for t in free:
            weights[t] = raw[t] / free_total * remaining
    return weights


def format_sizing_report(positions: list[dict], portfolio_value: float = 100_000) -> str:
    """Return a human-readable position sizing table."""
    if not positions:
        return "No actionable positions to size."

    lines = [
        f"VOLATILITY-ADJUSTED POSITION SIZING  (Portfolio: ${portfolio_value:,.0f})",
        "=" * 75,
        f"{'Ticker':<8} {'Verdict':<6} {'Weight':>8} {'$ Size':>10} {'Vol':>8} {'Score':>7}",
        "-" * 75,
    ]
    for p in positions:
        lines.append(
            f"{p['ticker']:<8} {p['verdict']:<6} {p['weight_pct']:>7.1f}% "
            f"${p['dollar_size']:>9,.0f} {p['vol_annual_pct']:>7.1f}% "
            f"{p['ensemble_score']:>7.1f}"
        )
    lines.append("-" * 75)
    total_w = sum(p["weight_pct"] for p in positions)
    lines.append(f"{'TOTAL':<8} {'':<6} {total_w:>7.1f}%")
    if total_w < 99.95:
        lines.append(f"{'CASH':<8} {'':<6} {100 - total_w:>7.1f}%  (single-position cap {MAX_SINGLE_WEIGHT:.0%})")
    return "\n".join(lines)
