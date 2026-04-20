"""
Bull/Base/Bear scenario derivation from Monte Carlo percentiles and HMM state.

Wraps the existing Monte Carlo output in a probability-weighted scenario
framework used by the narrative equity report:

    Bear  = MC p25 terminal price  → "downside if bear regime persists"
    Base  = MC p50 terminal price  → "central expectation"
    Bull  = MC p75 terminal price  → "upside if bull regime resumes"

Probability weights come from the HMM state probabilities (prob_bear, prob_sideways,
prob_bull). When HMM output is missing, weights default to equal (1/3 each).
"""

from __future__ import annotations


def build_scenarios(
    mc_result: dict | None,
    hmm_result: dict | None = None,
    horizon: int = 63,
    current_price: float | None = None,
) -> dict | None:
    """
    Build Bull/Base/Bear scenarios from MC percentiles and HMM weights.

    Args:
        mc_result: Output of MonteCarloSimulator.simulate().
        hmm_result: Output of HMMRegimeDetector.fit_and_predict() (optional).
        horizon: MC horizon key (days). Defaults to 63 (~3 months).
        current_price: Current ticker price. If None, reads from mc_result.

    Returns:
        Dict with bear/base/bull entries, each containing target_price,
        upside_pct, and probability. Plus expected_price (prob-weighted mean)
        and horizon_days. None if mc_result missing or malformed.
    """
    if not mc_result:
        return None

    horizons = mc_result.get("horizons") or {}
    h = horizons.get(horizon) or horizons.get(str(horizon))
    if not h:
        # Fall back to the largest available horizon
        if horizons:
            keys = sorted(
                [int(k) if isinstance(k, str) else k for k in horizons.keys()]
            )
            fallback_key = keys[-1]
            h = horizons.get(fallback_key) or horizons.get(str(fallback_key))
            horizon = fallback_key
        if not h:
            return None

    percentiles = h.get("percentiles") or {}
    p25 = percentiles.get("p25")
    p50 = percentiles.get("p50")
    p75 = percentiles.get("p75")
    if p25 is None or p50 is None or p75 is None:
        return None

    price = current_price
    if price is None:
        price = mc_result.get("current_price")
    if price is None or price <= 0:
        return None

    # Probability weights from HMM (bear / sideways / bull) → (bear / base / bull)
    if hmm_result:
        prob_bear = float(hmm_result.get("prob_bear") or 0.0)
        prob_sideways = float(hmm_result.get("prob_sideways") or 0.0)
        prob_bull = float(hmm_result.get("prob_bull") or 0.0)
        total = prob_bear + prob_sideways + prob_bull
        if total > 0:
            prob_bear /= total
            prob_sideways /= total
            prob_bull /= total
        else:
            prob_bear = prob_sideways = prob_bull = 1.0 / 3.0
    else:
        prob_bear = prob_sideways = prob_bull = 1.0 / 3.0

    def upside(target: float) -> float:
        return (target - price) / price

    expected_price = (
        prob_bear * p25 + prob_sideways * p50 + prob_bull * p75
    )

    return {
        "horizon_days": horizon,
        "current_price": round(float(price), 2),
        "bear": {
            "label": "Bear",
            "target_price": round(float(p25), 2),
            "upside_pct": round(upside(p25) * 100, 1),
            "probability": round(prob_bear, 4),
            "description": "Downside case — MC 25th percentile; HMM bear probability.",
        },
        "base": {
            "label": "Base",
            "target_price": round(float(p50), 2),
            "upside_pct": round(upside(p50) * 100, 1),
            "probability": round(prob_sideways, 4),
            "description": "Central case — MC median; HMM sideways probability.",
        },
        "bull": {
            "label": "Bull",
            "target_price": round(float(p75), 2),
            "upside_pct": round(upside(p75) * 100, 1),
            "probability": round(prob_bull, 4),
            "description": "Upside case — MC 75th percentile; HMM bull probability.",
        },
        "expected_price": round(float(expected_price), 2),
        "expected_upside_pct": round(upside(expected_price) * 100, 1),
    }


def format_scenarios_table(scenarios: dict) -> str:
    """Render a markdown table for the scenario block."""
    if not scenarios:
        return "_No scenario data available._"

    header = (
        f"| Scenario | Target Price | Upside | Probability |\n"
        f"|---|---:|---:|---:|\n"
    )
    rows = []
    for key in ("bull", "base", "bear"):
        s = scenarios.get(key)
        if not s:
            continue
        rows.append(
            f"| **{s['label']}** | ${s['target_price']:.2f} "
            f"| {s['upside_pct']:+.1f}% "
            f"| {s['probability']*100:.0f}% |"
        )
    footer = (
        f"\n*Probability-weighted expected price "
        f"(~{scenarios['horizon_days']}-day horizon):* "
        f"**${scenarios['expected_price']:.2f} "
        f"({scenarios['expected_upside_pct']:+.1f}%)**"
    )
    return header + "\n".join(rows) + footer
