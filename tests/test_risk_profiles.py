"""Tests confirming risk profiles wire through conviction threshold and position cap."""

import json
from pathlib import Path

import pytest

from src.analysis.position_sizing import compute_position_sizes, load_risk_profile


RISK_PROFILES_PATH = Path(__file__).resolve().parents[1] / "config" / "risk_profiles.json"


def test_risk_profiles_file_exists_and_has_three_profiles():
    assert RISK_PROFILES_PATH.exists()
    profiles = json.loads(RISK_PROFILES_PATH.read_text())
    assert "conservative" in profiles
    assert "moderate" in profiles
    assert "aggressive" in profiles


def test_conservative_has_stricter_min_conviction_than_aggressive():
    cons = load_risk_profile("conservative")
    agg = load_risk_profile("aggressive")
    assert cons["min_conviction"] > agg["min_conviction"]


def test_conservative_has_smaller_max_position_than_aggressive():
    cons = load_risk_profile("conservative")
    agg = load_risk_profile("aggressive")
    assert cons["max_position_size"] < agg["max_position_size"]


def test_load_risk_profile_falls_back_to_moderate_on_unknown():
    fallback = load_risk_profile("ultra_yolo")
    moderate = load_risk_profile("moderate")
    assert fallback == moderate


def _fake_result(ticker: str, verdict: str, vol: float = 0.25):
    return {
        "ticker": ticker,
        "verdict": verdict,
        "price": 100.0,
        "buy_score": 80 if verdict == "BUY" else 30,
        "sell_score": 30 if verdict == "BUY" else 80,
        "models": {
            "monte_carlo": {"annual_volatility": vol},
            "garch": {"current_conditional_vol_annual": vol},
        },
    }


def test_position_sizing_conservative_weight_capped_tighter():
    """Single actionable verdict should hit the profile's max_position_size cap."""
    results = [_fake_result(f"T{i}", "BUY", vol=0.25) for i in range(5)]

    cons_positions = compute_position_sizes(results, risk_profile="conservative")
    agg_positions = compute_position_sizes(results, risk_profile="aggressive")

    cons_top_w = max(p["weight_pct"] for p in cons_positions)
    agg_top_w = max(p["weight_pct"] for p in agg_positions)

    assert cons_top_w <= 2.0 + 0.1  # conservative cap is 2%
    assert agg_top_w > cons_top_w


def test_position_sizing_reports_profile_metadata():
    results = [_fake_result("AAPL", "BUY")]
    positions = compute_position_sizes(results, risk_profile="moderate")
    assert positions
    p = positions[0]
    assert p["risk_profile"] == "moderate"
    assert p["stop_loss_pct"] > 0
    assert p["stop_price"] is not None
    assert p["stop_price"] < p.get("entry_price", 100.0) if p["verdict"] == "BUY" else True


def test_position_sizing_stop_direction_for_sell():
    results = [_fake_result("AAPL", "SELL")]
    positions = compute_position_sizes(results, risk_profile="moderate")
    assert positions
    p = positions[0]
    # For a SELL, stop is above entry
    assert p["stop_price"] > 100.0


def test_position_sizing_empty_when_only_holds():
    results = [{"ticker": "X", "verdict": "HOLD", "price": 50}]
    assert compute_position_sizes(results) == []
