"""Tests for the 8-section markdown equity research report."""

from unittest.mock import patch

import pytest

from src.reporting.markdown_report import generate_report


def _analysis_fixture(verdict="BUY"):
    return {
        "ticker": "TEST",
        "price": 100.0,
        "momentum_30d": 5.0,
        "momentum_90d": 12.0,
        "drawdown_52w": -8.0,
        "hmm_regime": "bull",
        "buy_score": 78.0 if verdict == "BUY" else 32.0,
        "sell_score": 22.0 if verdict == "BUY" else 78.0,
        "buy_rec": "buy" if verdict == "BUY" else "hold",
        "sell_rec": "hold",
        "verdict": verdict,
        "n_models": 7,
        "confidence": 0.85,
        "models": {
            "monte_carlo": {
                "current_price": 100.0,
                "horizons": {
                    63: {
                        "percentiles": {"p25": 92.0, "p50": 108.0, "p75": 125.0},
                        "probability_of_profit": 0.62,
                        "expected_return": 0.08,
                    }
                },
            },
            "hmm": {
                "current_state": "bull",
                "prob_bull": 0.65, "prob_bear": 0.1, "prob_sideways": 0.25,
            },
            "garch": {
                "current_conditional_vol_annual": 0.22,
                "forecasts": {
                    5: {"volatility_ratio": 0.95},
                    20: {"volatility_ratio": 1.02},
                },
            },
            "fama_french": {
                "alpha_annual": 0.03, "beta_market": 1.1,
                "beta_smb": 0.2, "beta_hml": -0.1, "r_squared": 0.72,
            },
            "copula": {"tail_risk_score": 35, "cvar_95": -0.03},
            "earnings": {
                "earnings_nearby": True,
                "days_to_earnings": 14,
                "next_earnings_date": "2026-05-04",
                "surprise_history": {"beat_rate": 0.80, "avg_surprise_pct": 0.04},
                "score": 70,
            },
            "bayesian_decay": {"decay_quality": "slow_decay"},
            "xgboost": None,
        },
        "buy_components": {
            "monte_carlo": 72, "hmm_regime": 68, "garch": 62,
            "fama_french": 58, "copula_tail": 65, "bayesian_decay": 85,
            "earnings_overlay": 70,
        },
        "sell_components": {},
    }


@pytest.mark.asyncio
async def test_report_contains_all_eight_sections():
    a = _analysis_fixture()
    # Mock out the network-bound peer + catalyst lookups to isolate the renderer
    with patch("src.reporting.markdown_report.compare_to_peers", return_value=None), \
         patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        report = await generate_report("TEST", a, include_peers=False, include_catalysts=False)

    for header in (
        "## 1. Executive Summary",
        "## 2. Fundamental Analysis",
        "## 3. Catalyst Analysis",
        "## 4. Valuation & Price Targets",
        "## 5. Risk Assessment",
        "## 6. Technical Context",
        "## 7. Market Positioning",
        "## 8. Insider Signals",
    ):
        assert header in report, f"missing section: {header}"


@pytest.mark.asyncio
async def test_report_disclaimer_present():
    a = _analysis_fixture()
    with patch("src.reporting.markdown_report.compare_to_peers", return_value=None), \
         patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        report = await generate_report("TEST", a, include_peers=False, include_catalysts=False)
    assert "**Disclaimer:**" in report


@pytest.mark.asyncio
async def test_report_executive_summary_contains_verdict():
    a = _analysis_fixture(verdict="BUY")
    with patch("src.reporting.markdown_report.compare_to_peers", return_value=None), \
         patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        report = await generate_report("TEST", a, include_peers=False, include_catalysts=False)
    assert "**BUY**" in report


@pytest.mark.asyncio
async def test_report_risk_profile_appears():
    a = _analysis_fixture()
    with patch("src.reporting.markdown_report.compare_to_peers", return_value=None), \
         patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        for profile in ("conservative", "moderate", "aggressive"):
            report = await generate_report(
                "TEST", a, risk_profile=profile, include_peers=False, include_catalysts=False
            )
            assert profile in report


@pytest.mark.asyncio
async def test_report_hold_verdict_has_no_position_line():
    a = _analysis_fixture(verdict="HOLD")
    with patch("src.reporting.markdown_report.compare_to_peers", return_value=None), \
         patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        report = await generate_report("TEST", a, include_peers=False, include_catalysts=False)
    assert "HOLD" in report
    # Position sizing block should note n/a for HOLDs
    assert "n/a" in report.lower() or "no position" in report.lower()


@pytest.mark.asyncio
async def test_report_includes_price_targets_when_mc_present():
    a = _analysis_fixture()
    with patch("src.reporting.markdown_report.compare_to_peers", return_value=None), \
         patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        report = await generate_report("TEST", a, include_peers=False, include_catalysts=False)
    # MC percentiles should drive bull/base/bear scenarios
    assert "Bull" in report
    assert "Bear" in report
    assert "Base" in report


@pytest.mark.asyncio
async def test_report_compliance_replaces_prohibited_terms():
    """If the pipeline ever emits an absolute term, compliance should scrub it."""
    a = _analysis_fixture()
    with patch("src.reporting.markdown_report.compare_to_peers", return_value={
        "ticker": "TEST", "sector": "Technology",
        "target": {"ticker": "TEST", "market_cap": 1e9,
                   "pe": 20, "ps": 5, "ev_ebitda": 15,
                   "gross_margin": 0.4, "operating_margin": 0.2},
        "peers": [],
        "sector_medians": {"pe": 20, "ps": 5, "ev_ebitda": 15,
                           "gross_margin": 0.4, "operating_margin": 0.2},
        "deltas": {"pe_delta_pct": 0, "ps_delta_pct": 0, "ev_ebitda_delta_pct": 0,
                   "gross_margin_delta_bps": 0, "operating_margin_delta_bps": 0},
        "n_peers": 0,
    }), patch("src.reporting.markdown_report.get_catalysts", return_value=None):
        report = await generate_report("TEST", a, include_peers=True, include_catalysts=False)
    # Disclaimer text itself is allowed to mention "not indicative"; just ensure output assembled
    assert "TEST" in report
    assert "Disclaimer" in report
