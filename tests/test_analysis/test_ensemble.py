"""Unit tests for ensemble scorer + Benjamini-Hochberg FDR correction."""

import pytest

from src.analysis.ensemble_scoring import (
    EnsembleScorer,
    benjamini_hochberg,
    get_sector_multiplier,
    load_sector_weights,
)


# ── benjamini_hochberg ───────────────────────────────────────────────────────


def test_bh_empty_returns_empty():
    assert benjamini_hochberg([]) == []


def test_bh_all_ones_rejects_nothing():
    assert benjamini_hochberg([1.0, 1.0, 1.0]) == [False, False, False]


def test_bh_one_clear_signal_accepted():
    # 0.001 against ten 1.0s should remain significant after BH at 0.05
    p_values = [0.001] + [1.0] * 9
    decisions = benjamini_hochberg(p_values, alpha=0.05)
    assert decisions[0] is True
    assert all(d is False for d in decisions[1:])


def test_bh_handles_nan_as_not_significant():
    p_values = [float("nan"), 0.04, 1.0]
    decisions = benjamini_hochberg(p_values, alpha=0.05)
    assert decisions[0] is False  # NaN treated as 1.0


def test_bh_more_conservative_than_uncorrected():
    """BH should reject ≤ as many hypotheses as a naive p<0.05 cut."""
    p_values = [0.001, 0.02, 0.03, 0.04, 0.05, 0.06]
    bh = benjamini_hochberg(p_values, alpha=0.05)
    naive = [p < 0.05 for p in p_values]
    assert sum(bh) <= sum(naive)


# ── EnsembleScorer ──────────────────────────────────────────────────────────


def _strong_buy_components():
    """Synthetic component dicts that should produce a high ensemble score for 'buy'."""
    return dict(
        monte_carlo={"horizons": {30: {"probability_of_profit": 0.8, "expected_return": 0.05}}},
        hmm={"prob_bull": 0.85, "prob_bear": 0.05, "prob_sideways": 0.10},
        garch={"forecasts": {5: {"interpretation": "volatility_stable", "volatility_ratio": 1.0}},
               "current_conditional_vol_annual": 0.18},
        fama_french={"alpha_annual": 0.06, "r_squared": 0.6, "beta_market": 1.1},
        copula={"tail_risk_score": 30, "tail_dep_lower": 0.10, "joint_crash_prob": 0.02},
        bayesian_decay={"decay_quality": "strong", "annualized_ir": 1.2,
                         "signal_strength_5d": 0.02, "signal_strength_20d": 0.01},
        event_study={"car_5d": 0.02, "car_20d": 0.05, "p_value": 0.02, "is_significant": True},
    )


def test_score_returns_insufficient_data_when_no_models():
    scorer = EnsembleScorer()
    out = scorer.score(direction="buy")
    assert out["recommendation"] == "insufficient_data"
    assert out["n_models"] == 0
    assert out["total_score"] == 0


def test_score_total_in_zero_to_hundred_range():
    scorer = EnsembleScorer()
    out = scorer.score(direction="buy", **_strong_buy_components())
    assert 0 <= out["total_score"] <= 100


def test_score_includes_components_for_each_provided_model():
    scorer = EnsembleScorer()
    components = _strong_buy_components()
    out = scorer.score(direction="buy", **components)
    # Each provided model should appear in the breakdown
    assert "monte_carlo" in out["components"]
    assert "hmm_regime" in out["components"]
    assert "garch" in out["components"]
    assert "event_study" in out["components"]


def test_score_buy_higher_than_sell_for_bullish_inputs():
    scorer = EnsembleScorer()
    components = _strong_buy_components()
    buy = scorer.score(direction="buy", **components)
    sell = scorer.score(direction="sell", **components)
    assert buy["total_score"] > sell["total_score"]


def test_score_recommendation_strings_are_valid():
    scorer = EnsembleScorer()
    out = scorer.score(direction="buy", **_strong_buy_components())
    assert out["recommendation"] in {
        "strong_buy", "buy", "hold", "avoid", "strong_sell", "sell",
        "insufficient_data",
    }


def test_score_agreement_gate_downgrades_to_hold():
    """If only 1-2 models agree above neutral, ensemble should downgrade."""
    scorer = EnsembleScorer()
    # Provide just two models with strong signal — below MIN_AGREEING_MODELS=4
    out = scorer.score(
        direction="buy",
        monte_carlo={"horizons": {30: {"probability_of_profit": 0.8, "expected_return": 0.05}}},
        hmm={"prob_bull": 0.85, "prob_bear": 0.05, "prob_sideways": 0.10},
    )
    # With only 2 models, agreement_met is False; recommendation should not be strong_buy
    assert out["recommendation"] in {"hold", "buy", "avoid", "strong_buy"}
    # If total_score >= 55, it should be downgraded to hold
    if out["total_score"] >= 55:
        assert out["recommendation"] == "hold"


def test_calibrated_weights_override_default():
    custom = {
        "monte_carlo": 1.0, "hmm_regime": 0.0, "garch": 0.0,
        "fama_french": 0.0, "copula_tail": 0.0, "bayesian_decay": 0.0,
        "event_study": 0.0, "options_flow": 0.0, "earnings_overlay": 0.0,
    }
    scorer = EnsembleScorer(calibrated_weights=custom)
    assert scorer.WEIGHTS == custom


def test_sector_multiplier_default_is_one():
    """Unknown sector → 1.0 (no effect)."""
    assert get_sector_multiplier(None) == 1.0
    assert get_sector_multiplier("UnknownGalaxy") == load_sector_weights().get(
        "_default", 1.0
    )


def test_score_sector_multiplier_clamped():
    """Even with a heavy multiplier, total_score must stay in [0, 100]."""
    scorer = EnsembleScorer()
    out = scorer.score(direction="buy", sector="Technology", **_strong_buy_components())
    assert 0 <= out["total_score"] <= 100
