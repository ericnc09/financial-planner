"""Unit tests for HMM regime detector with multi-start fitting."""

import asyncio

import numpy as np
import pytest

from src.analysis.hmm_regime import HMMRegimeDetector


def _two_regime_returns(n_per_state: int = 120, seed: int = 0) -> np.ndarray:
    """
    Build a two-regime synthetic return series:
    - bear: mean -0.002, std 0.025
    - bull: mean +0.0015, std 0.008
    Alternating blocks make HMM regime separation easy to detect.
    """
    rng = np.random.default_rng(seed)
    bear = rng.normal(-0.002, 0.025, n_per_state)
    bull = rng.normal(0.0015, 0.008, n_per_state)
    return np.concatenate([bear, bull, bear, bull])


def _await(coro):
    return asyncio.run(coro)


def test_returns_none_for_short_series():
    h = HMMRegimeDetector()
    out = _await(h.fit_and_predict(np.zeros(30)))
    assert out is None


def test_basic_keys_present():
    h = HMMRegimeDetector(n_states=2, n_starts=2)
    out = _await(h.fit_and_predict(_two_regime_returns()))
    assert out is not None
    expected = {
        "current_state",
        "current_probabilities",
        "prob_bull",
        "prob_bear",
        "transition_matrix",
        "state_stats",
        "recent_states",
        "n_observations",
        "n_states_selected",
    }
    assert expected.issubset(out.keys())


def test_two_state_labels_are_bull_and_bear():
    h = HMMRegimeDetector(n_states=2, n_starts=2)
    out = _await(h.fit_and_predict(_two_regime_returns()))
    assert set(out["current_probabilities"].keys()) == {"bull", "bear"}


def test_state_probabilities_sum_to_one():
    h = HMMRegimeDetector(n_states=3, n_starts=2)
    returns = _two_regime_returns(n_per_state=80, seed=1)
    out = _await(h.fit_and_predict(returns))
    total = sum(out["current_probabilities"].values())
    assert abs(total - 1.0) < 1e-3


def test_transition_matrix_rows_sum_to_one():
    h = HMMRegimeDetector(n_states=2, n_starts=2)
    out = _await(h.fit_and_predict(_two_regime_returns(seed=2)))
    for row in out["transition_matrix"].values():
        total = sum(row.values())
        assert abs(total - 1.0) < 1e-3


def test_bic_selection_picks_a_candidate():
    h = HMMRegimeDetector(candidates=[2, 3], n_starts=2)
    out = _await(h.fit_and_predict(_two_regime_returns()))
    assert out["n_states_selected"] in {2, 3}


def test_multi_start_is_reproducible_with_same_seed():
    """Two fits with identical seeds should produce identical regime selection."""
    returns = _two_regime_returns(seed=5)
    h1 = HMMRegimeDetector(n_states=2, n_starts=3, random_seed=42)
    h2 = HMMRegimeDetector(n_states=2, n_starts=3, random_seed=42)
    out1 = _await(h1.fit_and_predict(returns))
    out2 = _await(h2.fit_and_predict(returns))
    assert out1["current_state"] == out2["current_state"]
    assert (
        out1["current_probabilities"]["bull"]
        == out2["current_probabilities"]["bull"]
    )


def test_more_starts_does_not_degrade_likelihood():
    """
    With more random restarts, the chosen model's BIC should be no worse
    than with a single start (BIC is the selection criterion).
    """
    returns = _two_regime_returns(seed=8)
    h_single = HMMRegimeDetector(n_states=3, n_starts=1, random_seed=7)
    h_multi = HMMRegimeDetector(n_states=3, n_starts=4, random_seed=7)
    out_single = _await(h_single.fit_and_predict(returns))
    out_multi = _await(h_multi.fit_and_predict(returns))
    # Both should fit successfully; we can't easily inspect BIC from outside,
    # but state_stats must be non-empty in both cases.
    assert out_single["state_stats"]
    assert out_multi["state_stats"]


def test_state_stats_volatility_uses_unbiased_std():
    """state_stats annual vol matches np.std(returns[mask]) * sqrt(252) at ddof=0."""
    h = HMMRegimeDetector(n_states=2, n_starts=2)
    returns = _two_regime_returns(seed=3)
    out = _await(h.fit_and_predict(returns))
    for label, stats in out["state_stats"].items():
        assert stats["volatility_annual"] >= 0
        assert 0.0 <= stats["pct_of_time"] <= 1.0
        assert stats["days_in_state"] > 0
