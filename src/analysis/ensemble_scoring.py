"""
Ensemble Scoring — Combines all model outputs into a unified signal score.

Aggregates Monte Carlo, HMM, GARCH, Fama-French, Copula, Bayesian Decay,
and Event Study into a single composite score with component breakdown.

Supports walk-forward weight calibration: optimize weights on rolling
6-month windows, test on next month, to produce data-driven weights.
"""

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.stats import false_discovery_control

logger = structlog.get_logger()


# ── Multiple Testing Correction ──────────────────────────────────────────────

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """
    Benjamini-Hochberg FDR correction.

    Returns a boolean mask: True = reject null (signal is significant after correction).
    Uses scipy's false_discovery_control under the hood.
    """
    if not p_values:
        return []
    arr = np.array(p_values)
    # Replace NaN/None with 1.0 (not significant)
    arr = np.where(np.isfinite(arr), arr, 1.0)
    adjusted = false_discovery_control(arr, method="bh")
    return [bool(p <= alpha) for p in adjusted]


class EnsembleScorer:
    """
    Combines up to 11 model components (see DEFAULT_WEIGHTS for the
    authoritative weights): Monte Carlo, HMM regime, GARCH, Fama-French,
    copula tail risk, Bayesian decay, event study, options flow, earnings
    overlay, news sentiment, and weather overlay (weather only contributes
    for weather-sensitive sectors).

    Every component is scored 0-100 for the signal's own direction, with 50
    meaning "no information": a component that favours buying scores above 50
    for a buy and, by the same amount, below 50 for a sell.
    """

    # Components that describe how risky a position is rather than which way
    # the price should move. They move the total score but do not vote in the
    # agreement gate (low volatility is not evidence that a buy — or a sell —
    # is right).
    RISK_COMPONENTS = frozenset({"garch", "copula_tail"})

    # Agreement gate: an actionable recommendation needs at least this many
    # directional components taking a side (a score of exactly 50 abstains),
    # and at least this share of them on the signal's side.
    MIN_DIRECTIONAL_VOTES = 3
    MIN_AGREEMENT_RATIO = 0.60

    # Recommendation cut-offs on the 0-100 total.
    #
    # With every component centred on 50, totals cluster tightly: on the 102
    # most recent signals re-scored after the 2026-11 audit the median was
    # 48.6, p95 57.4, p99 66.0 and the maximum 67.0. The former "strong"
    # cut-off of 75 had never been reached by any of 4,621 stored scores.
    # 65 marks roughly the top 1-2% of signals in that sample. It is a thin
    # basis (one day, 61 tickers, pooled-alpha components not yet active):
    # re-derive it from the stored score distribution after a few weeks.
    ACTION_THRESHOLD = 55
    STRONG_THRESHOLD = 65
    HOLD_THRESHOLD = 40

    DEFAULT_WEIGHTS = {
        "monte_carlo": 0.140,
        "hmm_regime": 0.105,
        "garch": 0.075,
        "fama_french": 0.075,
        "copula_tail": 0.100,
        "bayesian_decay": 0.100,
        "event_study": 0.100,
        "options_flow": 0.090,
        "earnings_overlay": 0.085,
        "news_sentiment": 0.085,
        "weather_overlay": 0.045,
    }

    def __init__(self, calibrated_weights: dict[str, float] | None = None):
        """
        Args:
            calibrated_weights: Walk-forward optimized weights. Falls back to
                                DEFAULT_WEIGHTS if None or empty.
        """
        self.WEIGHTS = calibrated_weights if calibrated_weights else self.DEFAULT_WEIGHTS

    def score(
        self,
        direction: str,
        monte_carlo: dict | None = None,
        hmm: dict | None = None,
        garch: dict | None = None,
        fama_french: dict | None = None,
        copula: dict | None = None,
        bayesian_decay: dict | None = None,
        event_study: dict | None = None,
        options_flow: dict | None = None,
        earnings_overlay: dict | None = None,
        news_sentiment: dict | None = None,
        weather_overlay: dict | None = None,
    ) -> dict:
        """
        Compute ensemble score 0-100 from all available model outputs.

        Args:
            direction: 'buy' or 'sell'.
            Each model param: dict output from the respective analyzer, or None if unavailable.

        Returns:
            Dict with total score, component scores, confidence, and recommendation.
        """
        components = {}

        def add(name: str, value: float | None):
            # A scorer returns None when its input cannot support a score;
            # the component is then simply absent and weights renormalize.
            if value is not None and np.isfinite(value):
                components[name] = float(value)

        if monte_carlo:
            add("monte_carlo", self._score_monte_carlo(monte_carlo, direction))
        if hmm:
            add("hmm_regime", self._score_hmm(hmm, direction))
        if garch:
            add("garch", self._score_garch(garch))
        if fama_french:
            add("fama_french", self._score_fama_french(fama_french, direction))
        if copula:
            add("copula_tail", self._score_copula(copula))
        if bayesian_decay:
            add("bayesian_decay", self._score_bayesian_decay(bayesian_decay))
        if event_study:
            add("event_study", self._score_event_study(event_study, direction))
        if options_flow:
            add("options_flow", self._score_options_flow(options_flow, direction))
        if earnings_overlay:
            add("earnings_overlay", earnings_overlay.get("score", 50))
        if news_sentiment:
            from src.analysis.news_sentiment import NewsSentimentAnalyzer
            add("news_sentiment", NewsSentimentAnalyzer.score_for_ensemble(news_sentiment, direction))
        if weather_overlay:
            add("weather_overlay", weather_overlay.get("score"))

        return self.aggregate(direction, components)

    def aggregate(self, direction: str, components: dict[str, float]) -> dict:
        """Combine 0-100 component scores into the total, the agreement gate
        and the recommendation."""
        available_weight = sum(self.WEIGHTS[name] for name in components)

        # Weighted average (re-normalize if some models missing)
        if available_weight == 0:
            return {"total_score": 0, "components": {}, "confidence": 0, "recommendation": "insufficient_data", "n_models": 0}

        total = 0.0
        for name, score in components.items():
            total += score * (self.WEIGHTS[name] / available_weight)

        n_models = len(components)
        confidence = round(available_weight / sum(self.WEIGHTS.values()), 4)

        # Model agreement among directional components that take a side
        votes = {
            name: s for name, s in components.items()
            if name not in self.RISK_COMPONENTS and abs(s - 50.0) > 1e-6
        }
        n_votes = len(votes)
        agreeing_models = sum(1 for s in votes.values() if s > 50)
        agreement_ratio = agreeing_models / n_votes if n_votes else 0.0
        agreement_met = (
            n_votes >= self.MIN_DIRECTIONAL_VOTES
            and agreement_ratio >= self.MIN_AGREEMENT_RATIO
        )

        # Recommendation (gated by model agreement): with many components a
        # high total can come from one or two extreme scores, so an
        # actionable call also needs most directional models on its side.
        if not agreement_met and total >= self.ACTION_THRESHOLD:
            rec = "hold"
        elif total >= self.STRONG_THRESHOLD:
            rec = "strong_buy" if direction == "buy" else "strong_sell"
        elif total >= self.ACTION_THRESHOLD:
            rec = "buy" if direction == "buy" else "sell"
        elif total >= self.HOLD_THRESHOLD:
            rec = "hold"
        else:
            rec = "avoid"

        result = {
            "direction": direction,
            "total_score": round(total, 1),
            "components": {k: round(v, 1) for k, v in components.items()},
            "n_models": n_models,
            "agreeing_models": agreeing_models,
            "n_directional_votes": n_votes,
            "agreement_ratio": round(agreement_ratio, 4),
            "agreement_met": agreement_met,
            "confidence": confidence,
            "recommendation": rec,
        }

        logger.info(
            "ensemble.scored",
            direction=direction, total=round(total, 1),
            n_models=n_models, confidence=confidence, rec=rec,
        )
        return result

    # --- Component scorers (each returns 0-100, 50 = neutral, or None) ---

    def _score_monte_carlo(self, mc: dict, direction: str) -> float:
        h = mc.get("horizons", {})
        # Use 30-day horizon if available, else 90
        horizon = h.get(30) or h.get("30") or h.get(21) or h.get("21") or {}
        prob = horizon.get("probability_of_profit")
        exp_ret = horizon.get("expected_return")
        prob = 0.5 if prob is None else prob
        exp_ret = 0.0 if exp_ret is None else exp_ret

        if direction == "sell":
            prob = 1 - prob
            exp_ret = -exp_ret

        # prob_profit contributes 60pts, expected_return contributes 40pts
        prob_score = prob * 60
        ret_score = min(40, max(0, (exp_ret + 0.10) / 0.20 * 40))  # -10% to +10% range
        return min(100, prob_score + ret_score)

    def _score_hmm(self, hmm: dict, direction: str) -> float:
        # Folded probabilities (bull + bear + sideways = 1). Sideways is worth
        # exactly half, so a flat regime is neutral for buys and sells alike
        # and the two scores always sum to 100.
        bull = hmm.get("prob_bull", 0) or 0
        bear = hmm.get("prob_bear", 0) or 0
        sideways = hmm.get("prob_sideways", 0) or 0
        total = bull + bear + sideways
        if total < 0.5:
            # Legacy 4/5-state row whose mild_bull/mild_bear mass was dropped:
            # what is left is a rounding residue, not a regime.
            return None
        bull, bear, sideways = bull / total, bear / total, sideways / total
        favourable = bull if direction == "buy" else bear
        return favourable * 100 + sideways * 50

    def _score_garch(self, garch: dict) -> float:
        # Risk component: forecast volatility relative to recent realized
        # volatility. Ratio 1.0 (no change expected) is neutral at 50;
        # contracting vol scores higher, expanding vol lower.
        ratio_5d = garch.get("forecast_5d_ratio") or 1.0
        ratio_20d = garch.get("forecast_20d_ratio") or 1.0
        avg_ratio = (ratio_5d + ratio_20d) / 2
        return 50.0 - float(np.clip((avg_ratio - 1.0) / 0.5, -1.0, 1.0)) * 30.0

    def _score_fama_french(self, ff: dict, direction: str) -> float | None:
        # Factor-adjusted alpha, measured in standard errors. Positive alpha
        # supports a buy and counts against a sell. +/-3 standard errors maps
        # to the ends of the range.
        t_stat = ff.get("alpha_tstat")
        if t_stat is None:
            return None
        if direction == "sell":
            t_stat = -t_stat
        return 50.0 + 40.0 * float(np.clip(t_stat, -3.0, 3.0)) / 3.0

    def _score_copula(self, copula: dict) -> float:
        # Risk component, inverted: low tail risk = high score
        tail_score = copula.get("tail_risk_score", 50)
        return max(0, 100 - tail_score)

    def _score_bayesian_decay(self, bd: dict) -> float | None:
        # How much of this kind of signal's historical post-disclosure alpha
        # is still ahead of it. `remaining_share` falls from 1 toward 0 as the
        # signal ages; `prob_alpha_positive` is the posterior probability that
        # the pooled alpha (already direction-adjusted) is positive at all.
        remaining = bd.get("remaining_share")
        prob_positive = bd.get("prob_alpha_positive")
        if remaining is None or prob_positive is None:
            return None
        remaining = float(np.clip(remaining, 0.0, 1.0))
        edge = 2.0 * float(np.clip(prob_positive, 0.0, 1.0)) - 1.0  # -1 .. +1
        return 50.0 + 40.0 * remaining * edge

    def _score_event_study(self, es: dict, direction: str) -> float | None:
        # Pooled post-disclosure CAR of comparable past signals (or a single
        # event's CAR). Abnormal returns are direction-adjusted upstream:
        # positive = whoever followed the signal's direction gained.
        cars = [
            v for v in (
                es.get("car_5d") if es.get("car_5d") is not None else es.get("mean_car_5d"),
                es.get("car_20d") if es.get("car_20d") is not None else es.get("mean_car_20d"),
            ) if v is not None
        ]
        if not cars:
            return None
        car_avg = float(np.mean(cars))
        score = 50.0 + float(np.clip(car_avg / 0.05, -1.0, 1.0)) * 30.0
        if es.get("is_significant"):
            score += 10.0 if car_avg > 0 else -10.0
        return float(np.clip(score, 0.0, 100.0))

    def _score_options_flow(self, opts: dict, direction: str) -> float | None:
        """Score options positioning 0-100 from cross-sectional percentiles.

        Equity put skew and put/call ratios have no fixed neutral level, so
        each is read as a percentile against other tickers scored with the
        same method (the median ticker is neutral by construction).
        """
        pcr_pct = opts.get("pcr_percentile")
        skew_pct = opts.get("iv_skew_percentile")
        if pcr_pct is None and skew_pct is None:
            return None

        tilt = 0.0  # > 0 bullish
        if pcr_pct is not None:
            tilt += (0.5 - pcr_pct) * 40.0   # low put/call ratio = bullish, +/-20
        if skew_pct is not None:
            tilt += (0.5 - skew_pct) * 30.0  # cheap downside protection = bullish, +/-15
        if direction == "sell":
            tilt = -tilt

        # Unusual volume strengthens whatever the positioning says; it never
        # creates a signal on its own.
        uv = opts.get("unusual_volume_score") or 0
        if uv > 0.5:
            tilt *= 1.0 + 0.3 * min(float(uv), 1.0)

        return float(np.clip(50.0 + tilt, 0.0, 100.0))

    # ── Batch FDR Correction ─────────────────────────────────────────────

    @staticmethod
    def apply_fdr_filter(
        scored_signals: list[dict],
        alpha: float = 0.05,
    ) -> list[dict]:
        """
        Apply Benjamini-Hochberg FDR correction across a batch of scored signals.

        Collects event study p-values from each signal's components and corrects
        for multiple testing across the full ticker universe. Signals that fail
        FDR are downgraded to 'hold'.

        Args:
            scored_signals: List of dicts, each with at least:
                - "ensemble_result": dict from EnsembleScorer.score()
                - "event_study": dict with "p_value" field (optional)
            alpha: FDR significance level.

        Returns:
            Same list with "fdr_significant" and potentially adjusted recommendation.
        """
        # Collect p-values (use 1.0 for signals without event study)
        p_values = []
        for sig in scored_signals:
            es = sig.get("event_study") or {}
            p = es.get("p_value") or es.get("p_value_5d") or es.get("p_value_20d")
            p_values.append(float(p) if p is not None else 1.0)

        if not p_values:
            return scored_signals

        # Apply BH correction
        significant = benjamini_hochberg(p_values, alpha)

        for sig, is_sig in zip(scored_signals, significant):
            sig["fdr_significant"] = is_sig
            # Downgrade non-significant signals that got actionable recommendations
            if not is_sig:
                ens = sig.get("ensemble_result", {})
                rec = ens.get("recommendation", "")
                if rec in ("buy", "sell", "strong_buy", "strong_sell"):
                    ens["recommendation_pre_fdr"] = rec
                    ens["recommendation"] = "hold"
                    logger.info(
                        "fdr.downgraded",
                        ticker=sig.get("ticker", "?"),
                        original_rec=rec,
                        p_value=p_values[scored_signals.index(sig)],
                    )

        n_rejected = sum(significant)
        logger.info(
            "fdr.applied",
            n_signals=len(scored_signals),
            n_significant=n_rejected,
            n_downgraded=len(scored_signals) - n_rejected,
            alpha=alpha,
        )
        return scored_signals


class WalkForwardCalibrator:
    """
    Walk-forward optimization of ensemble weights.

    Given historical signal records (each with component scores and realized
    returns), optimizes weights on rolling train windows and evaluates on
    the next test window.

    Usage:
        calibrator = WalkForwardCalibrator()
        result = calibrator.calibrate(records)
        # result["optimized_weights"] can be passed to EnsembleScorer
    """

    MODEL_NAMES = list(EnsembleScorer.DEFAULT_WEIGHTS.keys())

    def __init__(
        self,
        train_months: int = 6,
        test_months: int = 1,
    ):
        self.train_months = train_months
        self.test_months = test_months

    def calibrate(self, records: list[dict]) -> dict:
        """
        Run walk-forward calibration on historical signal records.

        Args:
            records: List of dicts, each with:
                - "date": datetime or str (signal date)
                - "components": dict of model_name -> score (0-100)
                - "realized_return": float (actual return after hold period)
                - "direction": "buy" or "sell"

        Returns:
            Dict with optimized_weights, fold_results, and summary stats.
        """
        if len(records) < 30:
            logger.warning("walkforward.insufficient_records", n=len(records))
            return {
                "optimized_weights": EnsembleScorer.DEFAULT_WEIGHTS,
                "n_folds": 0,
                "status": "insufficient_data",
            }

        # Sort by date
        records = sorted(records, key=lambda r: r["date"])

        # Split into monthly folds
        folds = self._make_folds(records)
        if len(folds) < self.train_months + self.test_months:
            return {
                "optimized_weights": EnsembleScorer.DEFAULT_WEIGHTS,
                "n_folds": len(folds),
                "status": "insufficient_folds",
            }

        fold_results = []
        all_optimized_weights = []

        for i in range(self.train_months, len(folds) - self.test_months + 1):
            # Train on prior train_months folds
            train_data = []
            for f in range(i - self.train_months, i):
                train_data.extend(folds[f])

            # Test on next test_months folds
            test_data = []
            for f in range(i, min(i + self.test_months, len(folds))):
                test_data.extend(folds[f])

            if len(train_data) < 10 or len(test_data) < 3:
                continue

            # Optimize weights on train set
            opt_weights = self._optimize_weights(train_data)

            # Evaluate on test set
            train_corr = self._evaluate(train_data, opt_weights)
            test_corr = self._evaluate(test_data, opt_weights)

            fold_results.append({
                "fold": i,
                "train_n": len(train_data),
                "test_n": len(test_data),
                "train_corr": round(train_corr, 4),
                "test_corr": round(test_corr, 4),
                "weights": {k: round(v, 4) for k, v in opt_weights.items()},
            })
            all_optimized_weights.append(opt_weights)

        if not all_optimized_weights:
            return {
                "optimized_weights": EnsembleScorer.DEFAULT_WEIGHTS,
                "n_folds": 0,
                "status": "no_valid_folds",
            }

        # Average weights across all folds
        avg_weights = {}
        for name in self.MODEL_NAMES:
            avg_weights[name] = np.mean([w[name] for w in all_optimized_weights])
        # Re-normalize
        total = sum(avg_weights.values())
        avg_weights = {k: round(v / total, 4) for k, v in avg_weights.items()}

        avg_test_corr = np.mean([f["test_corr"] for f in fold_results])

        logger.info(
            "walkforward.calibrated",
            n_folds=len(fold_results),
            avg_test_corr=round(avg_test_corr, 4),
            weights=avg_weights,
        )

        return {
            "optimized_weights": avg_weights,
            "n_folds": len(fold_results),
            "avg_test_correlation": round(avg_test_corr, 4),
            "fold_results": fold_results,
            "status": "calibrated",
        }

    def _make_folds(self, records: list[dict]) -> list[list[dict]]:
        """Group records into monthly buckets."""
        folds = []
        current_month = None
        current_fold = []
        for r in records:
            d = r["date"]
            month_key = d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d)[:7]
            if month_key != current_month:
                if current_fold:
                    folds.append(current_fold)
                current_fold = [r]
                current_month = month_key
            else:
                current_fold.append(r)
        if current_fold:
            folds.append(current_fold)
        return folds

    def _optimize_weights(self, data: list[dict]) -> dict[str, float]:
        """Find weights that maximize rank correlation between ensemble score and realized return."""
        scores_matrix, returns_arr = self._build_matrices(data)
        n_models = len(self.MODEL_NAMES)

        if scores_matrix.shape[0] < n_models:
            return dict(EnsembleScorer.DEFAULT_WEIGHTS)

        # Spearman of the weighted score is piecewise-constant in w (ranks only
        # change at crossings), so gradient-based SLSQP sees a zero gradient and
        # returns the starting weights unchanged. Instead maximize the Pearson
        # correlation between the continuous weighted score and the FIXED ranks
        # of realized returns — smooth in w, same rank-based target.
        from scipy.stats import rankdata
        return_ranks = rankdata(returns_arr)

        def neg_corr(w):
            w = w / w.sum()  # normalize
            ensemble = scores_matrix @ w
            if np.std(ensemble) < 1e-12:
                return 0.0
            corr = np.corrcoef(ensemble, return_ranks)[0, 1]
            return -corr if not np.isnan(corr) else 0.0

        w0 = np.array([EnsembleScorer.DEFAULT_WEIGHTS[n] for n in self.MODEL_NAMES])
        bounds = [(0.01, 0.5)] * n_models
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        res = minimize(neg_corr, w0, method="SLSQP", bounds=bounds, constraints=cons)
        opt_w = res.x if res.success else w0
        opt_w = opt_w / opt_w.sum()

        return {name: float(opt_w[i]) for i, name in enumerate(self.MODEL_NAMES)}

    def _evaluate(self, data: list[dict], weights: dict[str, float]) -> float:
        """Compute Spearman correlation between weighted scores and realized returns."""
        scores_matrix, returns_arr = self._build_matrices(data)
        w = np.array([weights[n] for n in self.MODEL_NAMES])
        ensemble = scores_matrix @ w
        from scipy.stats import spearmanr
        corr, _ = spearmanr(ensemble, returns_arr)
        return corr if not np.isnan(corr) else 0.0

    def _build_matrices(self, data: list[dict]):
        """Extract score matrix (n_signals x n_models) and returns vector."""
        scores = []
        returns = []
        for r in data:
            comps = r.get("components", {})
            row = [comps.get(name, 50.0) for name in self.MODEL_NAMES]  # 50 = neutral default
            scores.append(row)
            returns.append(r["realized_return"])
        return np.array(scores), np.array(returns)


class RegimeConditionalCalibrator:
    """
    Extends walk-forward calibration with regime-conditional weights.

    Maintains separate weight vectors per market regime (bull/bear/sideways).
    Some models are more valuable in certain regimes — e.g. GARCH is more
    informative during vol spikes, HMM during regime transitions.

    Usage:
        calibrator = RegimeConditionalCalibrator()
        result = calibrator.calibrate(records)
        # result["regime_weights"]["bull"] etc.
    """

    REGIME_LABELS = ["bull", "bear", "sideways"]

    def __init__(self, train_months: int = 6, test_months: int = 1):
        self.train_months = train_months
        self.test_months = test_months
        self.base_calibrator = WalkForwardCalibrator(train_months, test_months)

    def calibrate(self, records: list[dict]) -> dict:
        """
        Calibrate separate weight vectors per regime.

        Args:
            records: Same as WalkForwardCalibrator, plus optional "regime" field
                     per record (e.g. "bull", "bear", "sideways").

        Returns:
            Dict with per-regime weights and a combined selector.
        """
        # Overall calibration (fallback)
        overall = self.base_calibrator.calibrate(records)

        # Per-regime calibration
        regime_weights = {}
        regime_stats = {}

        for regime in self.REGIME_LABELS:
            regime_records = [
                r for r in records
                if (r.get("regime", "").lower() == regime
                    or r.get("hmm_state", "").lower() == regime)
            ]

            if len(regime_records) >= 30:
                regime_result = self.base_calibrator.calibrate(regime_records)
                if regime_result.get("status") == "calibrated":
                    regime_weights[regime] = regime_result["optimized_weights"]
                    regime_stats[regime] = {
                        "n_records": len(regime_records),
                        "n_folds": regime_result["n_folds"],
                        "avg_test_corr": regime_result.get("avg_test_correlation"),
                    }
                else:
                    regime_weights[regime] = overall.get(
                        "optimized_weights", EnsembleScorer.DEFAULT_WEIGHTS
                    )
                    regime_stats[regime] = {
                        "n_records": len(regime_records),
                        "status": "fell_back_to_overall",
                    }
            else:
                regime_weights[regime] = overall.get(
                    "optimized_weights", EnsembleScorer.DEFAULT_WEIGHTS
                )
                regime_stats[regime] = {
                    "n_records": len(regime_records),
                    "status": "insufficient_data",
                }

        result = {
            "status": "calibrated",
            "overall_weights": overall.get("optimized_weights", EnsembleScorer.DEFAULT_WEIGHTS),
            "regime_weights": regime_weights,
            "regime_stats": regime_stats,
        }

        logger.info(
            "regime_calibrator.done",
            regimes_calibrated=[r for r, s in regime_stats.items() if s.get("n_folds")],
            regimes_fallback=[r for r, s in regime_stats.items() if not s.get("n_folds")],
        )
        return result

    def get_weights_for_regime(self, regime: str, calibration_result: dict) -> dict:
        """Look up the right weight vector for the current regime."""
        regime_key = regime.lower() if regime else "sideways"
        regime_weights = calibration_result.get("regime_weights", {})
        return regime_weights.get(
            regime_key,
            calibration_result.get("overall_weights", EnsembleScorer.DEFAULT_WEIGHTS),
        )
