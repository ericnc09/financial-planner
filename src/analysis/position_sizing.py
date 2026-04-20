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

import json
from pathlib import Path

import numpy as np
import structlog

logger = structlog.get_logger()

# Fallback cap when no risk profile is provided (avoids concentration)
MAX_SINGLE_WEIGHT = 0.30
# Minimum meaningful weight — positions below this are dropped
MIN_WEIGHT = 0.02

_RISK_PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_profiles.json"


def load_risk_profile(profile_name: str = "moderate") -> dict:
    """Load risk profile config. Falls back to moderate defaults on failure."""
    try:
        with open(_RISK_PROFILES_PATH) as f:
            profiles = json.load(f)
        return profiles.get(profile_name) or profiles["moderate"]
    except Exception:
        # Hard-coded fallback mirrors config/risk_profiles.json
        return {"max_position_size": 0.04, "min_conviction": 0.70, "stop_loss_threshold": 0.08}


def compute_position_sizes(
    results: list[dict],
    portfolio_value: float = 100_000,
    verdicts_to_size: tuple[str, ...] = ("BUY", "SELL"),
    risk_profile: str = "moderate",
) -> list[dict]:
    """
    Compute volatility-adjusted position sizes for actionable verdicts.

    Args:
        results: List of dicts from analyze_ticker().
        portfolio_value: Total portfolio in currency units (used for $ sizing).
        verdicts_to_size: Only size positions with these verdicts.
        risk_profile: conservative / moderate / aggressive. Controls the per-position cap.

    Returns:
        List of position dicts sorted by weight descending.
    """
    profile = load_risk_profile(risk_profile)
    max_weight = float(profile.get("max_position_size", MAX_SINGLE_WEIGHT))
    stop_loss = float(profile.get("stop_loss_threshold", 0.08))

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

    # Apply max-weight cap per position. Excess above the cap flows to cash —
    # we do NOT renormalise to 100% after capping, because doing so defeats
    # the cap's purpose. Under a conservative profile (2% cap, 5 names),
    # the portfolio holds 10% equities and 90% cash, which is the intended
    # behavior of that risk profile.
    capped: dict[str, float] = {t: min(w, max_weight) for t, w in raw_weights.items()}

    # Redistribute residual (capped-off excess) among uncapped positions
    # proportionally to their inverse-vol weights — but only up to the cap.
    # This lifts the allocation towards full deployment without breaching caps.
    for _ in range(5):  # iterate a handful of times for convergence
        residual = 1.0 - sum(capped.values())
        if residual <= 1e-6:
            break
        uncapped = {t: w for t, w in capped.items() if w < max_weight - 1e-9}
        if not uncapped:
            break
        pool = sum(inv_vols[t] for t in uncapped)
        if pool <= 0:
            break
        progress = False
        for t in uncapped:
            room = max_weight - capped[t]
            add = min(room, residual * (inv_vols[t] / pool))
            if add > 1e-9:
                capped[t] += add
                progress = True
        if not progress:
            break

    normalised = capped

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
        entry_price = r.get("price")
        stop_price = None
        if entry_price is not None:
            if r["verdict"] == "BUY":
                stop_price = round(float(entry_price) * (1.0 - stop_loss), 2)
            else:  # SELL — stop above entry
                stop_price = round(float(entry_price) * (1.0 + stop_loss), 2)

        positions.append({
            "ticker": ticker,
            "verdict": r["verdict"],
            "weight_pct": round(w * 100, 2),
            "dollar_size": round(dollar_size, 0),
            "vol_annual_pct": round(vol_pct, 1),
            "ensemble_score": round(ensemble_score, 1),
            "stop_loss_pct": round(stop_loss * 100, 1),
            "stop_price": stop_price,
            "risk_profile": risk_profile,
            "rationale": (
                f"Inverse-vol weight: 1/{vol_pct:.0f}% vol → {w*100:.1f}% of portfolio. "
                f"Ensemble score {ensemble_score:.0f}/100. "
                f"Profile={risk_profile} (cap {max_weight*100:.0f}%, stop {stop_loss*100:.0f}%)."
            ),
        })

    positions.sort(key=lambda x: -x["weight_pct"])

    logger.info(
        "position_sizing.complete",
        n_positions=len(positions),
        portfolio_value=portfolio_value,
        risk_profile=risk_profile,
        top=positions[0]["ticker"] if positions else None,
    )
    return positions


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
    return "\n".join(lines)
