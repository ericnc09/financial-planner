"""Tests for bull/base/bear scenario derivation from Monte Carlo + HMM."""

from src.analysis.scenarios import build_scenarios, format_scenarios_table


def _mc_fixture(current=100.0, p25=90.0, p50=105.0, p75=120.0):
    return {
        "current_price": current,
        "horizons": {
            63: {
                "percentiles": {"p25": p25, "p50": p50, "p75": p75},
            }
        },
    }


def test_build_scenarios_basic_shape():
    mc = _mc_fixture()
    s = build_scenarios(mc)
    assert s is not None
    assert s["horizon_days"] == 63
    assert s["current_price"] == 100.0
    assert s["bear"]["target_price"] == 90.0
    assert s["base"]["target_price"] == 105.0
    assert s["bull"]["target_price"] == 120.0


def test_build_scenarios_upside_sign():
    mc = _mc_fixture(current=100.0, p25=80.0, p50=100.0, p75=125.0)
    s = build_scenarios(mc)
    assert s["bear"]["upside_pct"] < 0
    assert abs(s["base"]["upside_pct"]) < 1e-6
    assert s["bull"]["upside_pct"] > 0


def test_build_scenarios_hmm_weights_respected():
    mc = _mc_fixture()
    hmm = {"prob_bull": 0.6, "prob_sideways": 0.3, "prob_bear": 0.1}
    s = build_scenarios(mc, hmm_result=hmm)
    assert abs(s["bull"]["probability"] - 0.6) < 1e-6
    assert abs(s["base"]["probability"] - 0.3) < 1e-6
    assert abs(s["bear"]["probability"] - 0.1) < 1e-6


def test_build_scenarios_no_hmm_equal_weights():
    mc = _mc_fixture()
    s = build_scenarios(mc, hmm_result=None)
    assert abs(s["bull"]["probability"] - 1 / 3) < 1e-4


def test_build_scenarios_expected_price_is_weighted_mean():
    mc = _mc_fixture(current=100.0, p25=80.0, p50=100.0, p75=120.0)
    hmm = {"prob_bull": 0.5, "prob_sideways": 0.3, "prob_bear": 0.2}
    s = build_scenarios(mc, hmm_result=hmm)
    expected = 0.2 * 80 + 0.3 * 100 + 0.5 * 120
    assert abs(s["expected_price"] - expected) < 0.01


def test_build_scenarios_returns_none_without_mc():
    assert build_scenarios(None) is None


def test_build_scenarios_returns_none_on_missing_percentiles():
    bad = {"current_price": 100.0, "horizons": {63: {"percentiles": {}}}}
    assert build_scenarios(bad) is None


def test_format_scenarios_table_contains_all_scenarios():
    s = build_scenarios(_mc_fixture())
    out = format_scenarios_table(s)
    assert "Bull" in out
    assert "Base" in out
    assert "Bear" in out
    assert "expected price" in out.lower()
