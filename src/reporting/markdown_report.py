"""
Institutional-grade equity research markdown report.

Takes the dict returned by `analyze_ticker()` and renders a narrative 8-section
markdown document: executive summary, fundamentals, catalysts, valuation,
risk, technicals, market positioning, and insider signals. Closes with the
mandatory compliance disclaimer.

Entry point:
    report_md = await generate_report(ticker, analysis, risk_profile="moderate")

The report mirrors the layout of the external claude-equity-research plugin's
/trading-ideas output, but sources every number from this repo's statistical
pipeline (Monte Carlo, HMM, GARCH, Fama-French, copula, Bayesian decay) —
not from web search.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import structlog

from src.analysis.catalyst_calendar import (
    format_catalyst_calendar,
    get_catalysts,
)
from src.analysis.peer_compare import compare_to_peers, format_peer_table
from src.analysis.position_sizing import compute_position_sizes, load_risk_profile
from src.analysis.scenarios import build_scenarios, format_scenarios_table
from src.reporting.compliance import compliance_check

logger = structlog.get_logger()


_RISK_PROFILES_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_profiles.json"


async def generate_report(
    ticker: str,
    analysis: dict,
    risk_profile: str = "moderate",
    portfolio_value: float = 100_000,
    include_peers: bool = True,
    include_catalysts: bool = True,
) -> str:
    """
    Render the 8-section equity research report as markdown.

    Args:
        ticker: Uppercase ticker.
        analysis: Output of `analyze_ticker()`.
        risk_profile: conservative / moderate / aggressive.
        portfolio_value: Used for dollar position sizing.
        include_peers: If False, skip the peer comparison (faster).
        include_catalysts: If False, skip the catalyst calendar (faster).

    Returns:
        Markdown string with disclaimer appended and prohibited terms replaced.
    """
    profile_config = load_risk_profile(risk_profile)

    # Parallelise the network-bound enrichments
    peer_task = compare_to_peers(ticker) if include_peers else _noop()
    catalyst_task = (
        get_catalysts(ticker, horizon_days=90) if include_catalysts else _noop()
    )
    peer_result, catalyst_result = await asyncio.gather(peer_task, catalyst_task)

    scenarios = build_scenarios(
        mc_result=(analysis.get("models") or {}).get("monte_carlo"),
        hmm_result=(analysis.get("models") or {}).get("hmm"),
        horizon=63,
        current_price=analysis.get("price"),
    )

    position = _compute_single_position(analysis, risk_profile, portfolio_value)

    sections = [
        _section_header(ticker, analysis),
        _section_executive_summary(ticker, analysis, scenarios, position, profile_config, risk_profile),
        _section_fundamentals(ticker, analysis, peer_result),
        _section_catalysts(catalyst_result),
        _section_valuation(analysis, scenarios),
        _section_risk(analysis, position, profile_config, risk_profile),
        _section_technicals(analysis),
        _section_market_positioning(analysis),
        _section_insider_signals(analysis),
        _section_metadata(analysis, risk_profile),
    ]

    report_md = "\n\n".join(s for s in sections if s)

    cleaned, violations = compliance_check(report_md)
    if violations:
        logger.info(
            "markdown_report.compliance_violations",
            ticker=ticker,
            count=len(violations),
        )

    return cleaned


async def _noop():
    return None


def _compute_single_position(
    analysis: dict, risk_profile: str, portfolio_value: float
) -> dict | None:
    """Size the target ticker as a one-name portfolio. Returns None if HOLD."""
    if analysis.get("verdict") not in ("BUY", "SELL"):
        return None
    positions = compute_position_sizes(
        [analysis],
        portfolio_value=portfolio_value,
        risk_profile=risk_profile,
    )
    return positions[0] if positions else None


# ── Section builders ────────────────────────────────────────────────────────

def _section_header(ticker: str, a: dict) -> str:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    price = a.get("price")
    price_str = f"${price:.2f}" if price is not None else "—"
    return (
        f"# {ticker} Equity Research\n\n"
        f"_Report generated {today} · Current price: {price_str}_"
    )


def _section_executive_summary(
    ticker: str,
    a: dict,
    scenarios: dict | None,
    position: dict | None,
    profile_config: dict,
    risk_profile: str,
) -> str:
    verdict = a.get("verdict", "HOLD")
    buy_score = a.get("buy_score", 0)
    sell_score = a.get("sell_score", 0)
    conviction = max(buy_score, sell_score) / 100.0
    confidence = a.get("confidence", 0)

    target_line = "_Target price not available (Monte Carlo output missing)._"
    if scenarios:
        target_line = (
            f"**~63-day base-case target:** ${scenarios['base']['target_price']:.2f} "
            f"({scenarios['base']['upside_pct']:+.1f}%)"
        )

    pos_line = "_Verdict is HOLD — no position recommended._"
    if position:
        pos_line = (
            f"**Suggested allocation:** {position['weight_pct']:.1f}% of portfolio "
            f"(~${position['dollar_size']:,.0f}), stop-loss {position['stop_loss_pct']:.0f}% "
            f"at ${position.get('stop_price', '—')}."
        )

    return (
        "## 1. Executive Summary\n\n"
        f"- **Verdict:** **{verdict}**  (buy score {buy_score:.0f}/100 · sell score {sell_score:.0f}/100)\n"
        f"- **Conviction:** {conviction*100:.0f}%  ·  **Model coverage:** {confidence*100:.0f}%\n"
        f"- **Risk profile:** {risk_profile} "
        f"(min conviction {profile_config.get('min_conviction', 0.7)*100:.0f}%, "
        f"max position {profile_config.get('max_position_size', 0.04)*100:.0f}%)\n"
        f"- {target_line}\n"
        f"- {pos_line}"
    )


def _section_fundamentals(ticker: str, a: dict, peer_result: dict | None) -> str:
    lines = ["## 2. Fundamental Analysis", ""]
    if peer_result:
        lines.append(format_peer_table(peer_result))
    else:
        lines.append("_Peer comparison not requested or unavailable for this sector._")
    return "\n".join(lines)


def _section_catalysts(catalyst_result: dict | None) -> str:
    lines = ["## 3. Catalyst Analysis", ""]
    if catalyst_result:
        lines.append(format_catalyst_calendar(catalyst_result))
    else:
        lines.append("_Catalyst calendar unavailable._")
    return "\n".join(lines)


def _section_valuation(a: dict, scenarios: dict | None) -> str:
    lines = ["## 4. Valuation & Price Targets", ""]
    if scenarios:
        lines.append(format_scenarios_table(scenarios))
    else:
        lines.append("_Monte Carlo did not produce valid percentiles; no scenarios._")

    mc = (a.get("models") or {}).get("monte_carlo")
    if mc:
        h63 = (mc.get("horizons") or {}).get(63) or (mc.get("horizons") or {}).get("63")
        if h63:
            prob = h63.get("probability_of_profit", 0.5) * 100
            exp_ret = h63.get("expected_return", 0) * 100
            lines.append("")
            lines.append(
                f"_Monte Carlo (10k paths, Student-t innovations): "
                f"{prob:.0f}% probability of profit over ~63 days, "
                f"expected return {exp_ret:+.1f}%._"
            )
    return "\n".join(lines)


def _section_risk(
    a: dict, position: dict | None, profile_config: dict, risk_profile: str
) -> str:
    models = a.get("models") or {}
    lines = ["## 5. Risk Assessment", ""]

    # Copula tail risk
    cop = models.get("copula")
    if cop:
        score = cop.get("tail_risk_score", 0)
        cvar = (cop.get("cvar_95") or 0) * 100
        label = "low" if score < 30 else "moderate" if score < 60 else "elevated"
        lines.append(
            f"- **Tail risk (Student-t copula):** {label} — score {score:.0f}/100, "
            f"CVaR₉₅ = {cvar:.1f}%."
        )
    else:
        lines.append("- **Tail risk:** copula model unavailable.")

    # GARCH vol regime
    garch = models.get("garch")
    if garch:
        cur_vol = (garch.get("current_conditional_vol_annual") or 0) * 100
        ratio_20 = (garch.get("forecasts") or {}).get(20, {}).get("volatility_ratio", 1.0)
        trend = (
            "contracting" if ratio_20 < 0.9
            else "expanding" if ratio_20 > 1.1
            else "stable"
        )
        lines.append(
            f"- **Volatility (GARCH):** current {cur_vol:.0f}% annualised, "
            f"20-day forecast {trend} ({ratio_20:.2f}× current)."
        )

    # Position sizing + stop loss
    if position:
        lines.append(
            f"- **Position sizing:** inverse-vol weight "
            f"{position['weight_pct']:.1f}% of portfolio "
            f"(~${position['dollar_size']:,.0f}), capped by `{risk_profile}` profile at "
            f"{profile_config.get('max_position_size', 0.04)*100:.0f}%."
        )
        lines.append(
            f"- **Stop loss:** {position['stop_loss_pct']:.0f}% "
            f"({'below' if position['verdict'] == 'BUY' else 'above'} entry) "
            f"→ trigger at ${position.get('stop_price', '—')}."
        )
    else:
        lines.append("- **Position sizing:** n/a (HOLD verdict).")

    return "\n".join(lines)


def _section_technicals(a: dict) -> str:
    models = a.get("models") or {}
    lines = ["## 6. Technical Context", ""]

    hmm = models.get("hmm")
    if hmm:
        state = (hmm.get("current_state") or "unknown").upper()
        bull_p = (hmm.get("prob_bull") or 0) * 100
        bear_p = (hmm.get("prob_bear") or 0) * 100
        side_p = (hmm.get("prob_sideways") or 0) * 100
        lines.append(
            f"- **Regime (HMM):** {state} "
            f"(P(bull)={bull_p:.0f}%, P(sideways)={side_p:.0f}%, P(bear)={bear_p:.0f}%)."
        )

    m30 = a.get("momentum_30d")
    m90 = a.get("momentum_90d")
    dd = a.get("drawdown_52w")
    if m30 is not None:
        lines.append(f"- **30-day momentum:** {m30:+.1f}%.")
    if m90 is not None:
        lines.append(f"- **90-day momentum:** {m90:+.1f}%.")
    if dd is not None:
        lines.append(f"- **Drawdown from 52-week high:** {dd:.1f}%.")

    bd = models.get("bayesian_decay")
    if bd:
        quality = (bd.get("decay_quality") or "no_signal").replace("_", " ")
        lines.append(f"- **Signal decay (Bayesian):** {quality}.")

    return "\n".join(lines)


def _section_market_positioning(a: dict) -> str:
    models = a.get("models") or {}
    lines = ["## 7. Market Positioning", ""]

    ff = models.get("fama_french")
    if ff:
        alpha = (ff.get("alpha_annual") or 0) * 100
        beta = ff.get("beta_market") or 0
        r2 = ff.get("r_squared") or 0
        smb = ff.get("beta_smb")
        hml = ff.get("beta_hml")
        lines.append(
            f"- **Fama-French 5-factor:** α = {alpha:+.1f}%/yr, "
            f"β_mkt = {beta:.2f}, R² = {r2:.2f}."
        )
        factor_bits = []
        if smb is not None:
            factor_bits.append(f"SMB {smb:+.2f}")
        if hml is not None:
            factor_bits.append(f"HML {hml:+.2f}")
        if factor_bits:
            lines.append(f"- **Factor exposures:** {', '.join(factor_bits)}.")
    else:
        lines.append("- Fama-French factor model unavailable for this name.")

    return "\n".join(lines)


def _section_insider_signals(a: dict) -> str:
    lines = ["## 8. Insider Signals & Ensemble Attribution", ""]

    comps = a.get("buy_components") or a.get("sell_components") or {}
    if comps:
        lines.append("**Ensemble component scores (0-100):**")
        lines.append("")
        lines.append("| Model | Score |")
        lines.append("|---|---:|")
        for name, score in sorted(comps.items(), key=lambda kv: -kv[1]):
            pretty = name.replace("_", " ").title()
            lines.append(f"| {pretty} | {score:.0f} |")
    else:
        lines.append("_Ensemble components unavailable._")

    earnings = (a.get("models") or {}).get("earnings")
    if earnings and earnings.get("earnings_nearby"):
        d = earnings.get("days_to_earnings")
        when = earnings.get("next_earnings_date", "—")
        sh = earnings.get("surprise_history") or {}
        beat_rate = sh.get("beat_rate")
        br = f"{beat_rate*100:.0f}% beat rate" if beat_rate is not None else "no history"
        lines.append("")
        lines.append(
            f"_Next earnings: **{when}** ({d:+d} days). Surprise history: {br}._"
        )

    return "\n".join(lines)


def _section_metadata(a: dict, risk_profile: str) -> str:
    verdict = a.get("verdict", "HOLD")
    n_models = a.get("n_models", 0)
    return (
        f"---\n\n"
        f"_Pipeline: {n_models} models contributed · risk profile `{risk_profile}` · "
        f"verdict **{verdict}**._"
    )
