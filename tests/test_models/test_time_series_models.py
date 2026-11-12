"""HMM regime, GARCH forecast, Bayesian decay."""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.analysis.bayesian_decay import BayesianDecayAnalyzer, decay_component
from src.analysis.ensemble_scoring import EnsembleScorer
from src.analysis.garch_forecast import GARCHForecaster
from src.analysis.hmm_regime import HMMRegimeDetector, fold_probabilities


def _regime_returns(seed=7):
    g = np.random.default_rng(seed)
    return np.concatenate([
        g.normal(0.002, 0.008, 160), g.normal(-0.003, 0.03, 120), g.normal(0.0005, 0.012, 200),
    ])


@pytest.mark.parametrize("n_states", [2, 3, 4])
def test_hmm_probabilities_always_fold_to_three_buckets(n_states):
    out = HMMRegimeDetector(n_states=n_states)._fit_predict_sync(_regime_returns(), None)
    assert out is not None
    assert out["current_state"] in ("bull", "bear", "sideways")
    assert out["prob_bull"] + out["prob_bear"] + out["prob_sideways"] == pytest.approx(1.0, abs=2e-3)
    assert sum(out["transition_from_current"].values()) == pytest.approx(1.0, abs=2e-3)


def test_mild_states_are_not_dropped():
    # 4/5-state models used to leave prob_bull+prob_bear+prob_sideways near 0
    folded = fold_probabilities({"mild_bull": 0.9, "bull": 0.05, "sideways": 0.03, "bear": 0.02})
    assert folded["bull"] == pytest.approx(0.95)
    scorer = EnsembleScorer()
    hmm = {"prob_bull": folded["bull"], "prob_bear": folded["bear"], "prob_sideways": folded["sideways"]}
    buy, sell = scorer._score_hmm(hmm, "buy"), scorer._score_hmm(hmm, "sell")
    assert buy > 90 and sell < 10
    assert buy + sell == pytest.approx(100.0)


def test_hmm_drops_non_finite_returns_instead_of_failing():
    r = _regime_returns()
    r[10] = np.nan
    r[50] = -np.inf
    assert HMMRegimeDetector(n_states=2)._fit_predict_sync(r, None) is not None


def _garch_series(seed, last_shock=None, n=600):
    g = np.random.default_rng(seed)
    r, v = np.zeros(n), 1.0
    for t in range(n):
        v = 0.05 + 0.10 * (r[t - 1] ** 2 if t else 1.0) + 0.85 * v
        r[t] = np.sqrt(v) * g.standard_normal()
    if last_shock is not None:
        r[-1] = last_shock
    return r / 100


@pytest.mark.parametrize("seed,shock", [(101, 6.0), (102, -6.0), (103, None)])
def test_garch_forecast_matches_arch_including_the_latest_return(seed, shock):
    from arch import arch_model

    ret = _garch_series(seed, shock)
    ours = GARCHForecaster()._forecast_sync(ret, [5, 20])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = arch_model(pd.Series(ret * 100), vol="Garch", p=1, q=1, dist="t", rescale=False).fit(
            disp="off", show_warning=False
        )
    path = fit.forecast(horizon=20, reindex=False).variance.iloc[-1].values
    for h in (5, 20):
        expected = np.sqrt(path[:h].mean()) / 100 * np.sqrt(252)
        assert ours["forecasts"][h]["predicted_volatility_annual"] == pytest.approx(expected, rel=0.05)


def test_big_last_day_move_raises_the_forecast():
    calm = GARCHForecaster()._forecast_sync(_garch_series(104), [5])
    shocked = GARCHForecaster()._forecast_sync(_garch_series(104, last_shock=8.0), [5])
    assert shocked["forecasts"][5]["predicted_volatility_annual"] > 1.3 * calm["forecasts"][5]["predicted_volatility_annual"]


def test_garch_ignores_non_finite_returns():
    ret = _garch_series(105)
    ret[5] = np.nan
    out = GARCHForecaster()._forecast_sync(ret, [5, 20])
    assert np.isfinite(out["current_conditional_vol_annual"])


def test_decay_recovers_half_life_and_sign():
    pytest.importorskip("emcee")
    g = np.random.default_rng(3)
    t = np.arange(21)
    positive = 0.004 * np.exp(-np.log(2) / 5 * t) + g.normal(0, 0.0006, 21)
    fit = BayesianDecayAnalyzer().analyze(positive)
    lo, hi = fit["ci_half_life_95"]
    assert lo <= 5 <= hi
    assert fit["prob_alpha_positive"] > 0.95
    assert fit["decay_quality"] in ("fast_decay", "moderate_decay")

    negative = BayesianDecayAnalyzer().analyze(-positive)
    assert negative["prob_alpha_positive"] < 0.05
    assert negative["decay_quality"] == "no_alpha"


def test_decay_fit_is_reproducible():
    pytest.importorskip("emcee")
    ar = np.random.default_rng(4).normal(0.0005, 0.001, 21)
    assert BayesianDecayAnalyzer().analyze(ar) == BayesianDecayAnalyzer().analyze(ar)


def test_decay_component_fades_with_age_and_scores_symmetrically():
    fit = {"posterior_lambda": np.log(2) / 10, "prob_alpha_positive": 0.9, "decay_quality": "moderate_decay"}
    fresh, aged = decay_component(fit, 0), decay_component(fit, 10)
    assert fresh["remaining_share"] == pytest.approx(1.0)
    assert aged["remaining_share"] == pytest.approx(0.5, abs=1e-3)
    scorer = EnsembleScorer()
    assert scorer._score_bayesian_decay(fresh) > scorer._score_bayesian_decay(aged) > 50
    unsure = decay_component({**fit, "prob_alpha_positive": 0.5}, 0)
    assert scorer._score_bayesian_decay(unsure) == pytest.approx(50.0)
    assert decay_component(None, 3) is None


# ── HMM labelling ─────────────────────────────────────────────────────────

def test_a_state_is_directional_only_when_its_mean_is_distinguishable_from_zero():
    from src.analysis.hmm_regime import directional_bucket

    g = np.random.default_rng(0)
    assert directional_bucket(g.normal(0.004, 0.01, 250))[0] == "bull"     # t ~ 6
    assert directional_bucket(g.normal(-0.004, 0.01, 250))[0] == "bear"
    assert directional_bucket(g.normal(0.0002, 0.02, 250))[0] == "sideways"  # +5%/yr in 32% vol: noise
    # a handful of days proves nothing, however extreme
    assert directional_bucket(np.array([0.25, 0.30, 0.28]))[0] == "sideways"


def test_one_day_outlier_does_not_make_the_rest_of_the_history_a_bear_market():
    # 400 quiet, mildly positive days plus a single +30% day. Ranking states
    # by mean used to call the outlier's state "bull" and everything else,
    # including today, "bear" with probability 1.
    g = np.random.default_rng(11)
    r = g.normal(0.0008, 0.010, 400)
    r[150] = 0.30
    out = HMMRegimeDetector(n_states=2)._fit_predict_sync(r, None)
    assert out["current_state"] != "bear"
    assert out["prob_bear"] < 0.5
    assert out["prob_bull"] + out["prob_bear"] + out["prob_sideways"] == pytest.approx(1.0, abs=2e-3)


def test_regime_does_not_flip_between_bull_and_bear_with_the_random_start():
    import src.analysis.hmm_regime as hmm_module

    calls = set()
    r = _regime_returns(seed=21)
    original = hmm_module.N_RESTARTS
    try:
        for restarts in (3, 10):
            hmm_module.N_RESTARTS = restarts
            calls.add(HMMRegimeDetector(candidates=[2, 3])._fit_predict_sync(r, None)["current_state"])
    finally:
        hmm_module.N_RESTARTS = original
    assert calls != {"bull", "bear"} and not {"bull", "bear"} <= calls


def test_ensemble_aggregate_matches_score():
    scorer = EnsembleScorer()
    hmm = {"prob_bull": 0.9, "prob_bear": 0.05, "prob_sideways": 0.05}
    full = scorer.score(direction="buy", hmm=hmm, fama_french={"alpha_tstat": 2.0})
    again = scorer.aggregate("buy", dict(full["components"]))
    assert again["total_score"] == full["total_score"] and again["recommendation"] == full["recommendation"]



def test_decay_is_absent_not_wrong_without_the_sampler(monkeypatch):
    # No emcee: the component must be missing, never a made-up number.
    import builtins

    real_import = builtins.__import__

    def no_emcee(name, *args, **kwargs):
        if name == "emcee":
            raise ImportError("emcee not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_emcee)
    ar = np.random.default_rng(5).normal(0.001, 0.001, 21)
    assert BayesianDecayAnalyzer().analyze(ar) is None
