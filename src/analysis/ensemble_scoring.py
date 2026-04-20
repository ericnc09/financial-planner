"""
Ensemble Scoring — Combines all model outputs into a unified signal score.

Aggregates Monte Carlo, HMM, GARCH, Fama-French, Copula, Bayesian Decay,
and Event Study into a single composite score with component breakdown.

Supports walk-forward weight calibration: optimize weights on rolling
6-month windows, test on next month, to produce data-driven weights.
"""

import json
from pathlib import Path

import numpy as np
import structlog
from scipy.optimize import minimize
from scipy.stats import false_discovery_control

logger = structlog.get_logger()


_SECTOR_WEIGHTS_PATH = Path(__file__).resolve().parents[2] / "config" / "sector_weights.json"
_SECTOR_WEIGHTS_CACHE: dict | None = None


def load_sector_weights() -> dict:
    """Load sector multipliers from config. Cached after first read."""
    global _SECTOR_WEIGHTS_CACHE
    if _SECTOR_WEIGHTS_CACHE is not None:
        return _SECTOR_WEIGHTS_CACHE
    try:
        with open(_SECTOR_WEIGHTS_PATH) as f:
            data = json.load(f)
        # Strip commentary keys
        _SECTOR_WEIGHTS_CACHE = {k: v for k, v in data.items() if not k.startswith("_") or k == "_default"}
    except Exception as e:
        logger.warning("sector_weights.load_failed", error=str(e))
        _SECTOR_WEIGHTS_CACHE = {"_default": 1.0}
    return _SECTOR_WEIGHTS_CACHE


def get_sector_multiplier(sector: str | None) -> float:
    """Look up a sector's ensemble multiplier. 1.0 (no effect) on miss."""
    if not sector:
        return 1.0
    weights = load_sector_weights()
    return float(weights.get(sector, weights.get("_default", 1.0)))


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
    Weights:
        monte_carlo  : 0.20  (probability + expected return)
        hmm_regime   : 0.15  (regime alignment with direction)
        garch        : 0.10  (volatility environment)
        fama_french  : 0.10  (alpha + factor alignment)
        copula_tail  : 0.15  (tail risk penalty)
        bayesian_decay: 0.15 (signal still active?)
        event_study  : 0.15  (historical CAR for this pattern)
    """

    # Minimum number of models that must independently score >= 50 (neutral)
    # to allow actionable (buy/sell) recommendations. Addresses multiple
    # testing: with 9 models, requiring 4+ agreement filters ~30% of
    # spurious signals where only 1-2 models happen to score high.
    MIN_AGREEING_MODELS = 4

    DEFAULT_WEIGHTS = {
        "monte_carlo": 0.160,
        "hmm_regime": 0.120,
        "garch": 0.085,
        "fama_french": 0.085,
        "copula_tail": 0.115,
        "bayesian_decay": 0.115,
        "event_study": 0.115,
        "options_flow": 0.105,
        "earnings_overlay": 0.100,
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
        sector: str | None = None,
    ) -> dict:
        """
        Compute ensemble score 0-100 from all available model outputs.

        Args:
            direction: 'buy' or 'sell'.
            Each model param: dict output from the respective analyzer, or None if unavailable.
            sector: Ticker's sector. If provided, a post-score multiplier is applied
                    from config/sector_weights.json (e.g. Technology 1.2x, Energy 0.8x).

        Returns:
            Dict with total score, component scores, confidence, and recommendation.
        """
        components = {}
        available_weight = 0.0

        # --- Monte Carlo (0-100) ---
        if monte_carlo:
            mc_score = self._score_monte_carlo(monte_carlo, direction)
            components["monte_carlo"] = mc_score
            available_weight += self.WEIGHTS["monte_carlo"]

        # --- HMM Regime (0-100) ---
        if hmm:
            hmm_score = self._score_hmm(hmm, direction)
            components["hmm_regime"] = hmm_score
            available_weight += self.WEIGHTS["hmm_regime"]

        # --- GARCH (0-100) ---
        if garch:
            garch_score = self._score_garch(garch)
            components["garch"] = garch_score
            available_weight += self.WEIGHTS["garch"]

        # --- Fama-French (0-100) ---
        if fama_french:
            ff_score = self._score_fama_french(fama_french)
            components["fama_french"] = ff_score
            available_weight += self.WEIGHTS["fama_french"]

        # --- Copula Tail Risk (0-100, inverted: low tail risk = high score) ---
        if copula:
            cop_score = self._score_copula(copula)
            components["copula_tail"] = cop_score
            available_weight += self.WEIGHTS["copula_tail"]

        # --- Bayesian Decay (0-100) ---
        if bayesian_decay:
            bd_score = self._score_bayesian_decay(bayesian_decay)
            components["bayesian_decay"] = bd_score
            available_weight += self.WEIGHTS["bayesian_decay"]

        # --- Event Study (0-100) ---
        if event_study:
            es_score = self._score_event_study(event_study, direction)
            components["event_study"] = es_score
            available_weight += self.WEIGHTS["event_study"]

        # --- Options Flow (0-100) ---
        if options_flow:
            of_score = self._score_options_flow(options_flow, direction)
            components["options_flow"] = of_score
            available_weight += self.WEIGHTS["options_flow"]

        # --- Earnings Overlay (0-100) ---
        if earnings_overlay:
            eo_score = earnings_overlay.get("score", 50)
            components["earnings_overlay"] = float(eo_score)
            available_weight += self.WEIGHTS["earnings_overlay"]

        # Weighted average (re-normalize if some models missing)
        if available_weight == 0:
            return {"total_score": 0, "components": {}, "confidence": 0, "recommendation": "insufficient_data", "n_models": 0}

        total = 0.0
        for name, score in components.items():
            total += score * (self.WEIGHTS[name] / available_weight)

        # Sector multiplier: applied *after* the weighted component average but
        # before recommendation thresholding. Statistical integrity of the
        # individual model scores (and their p-values) is preserved — this is a
        # prior on which sectors deserve higher conviction, not a re-weighting
        # of the underlying evidence. Clamp to [0, 100] after application.
        sector_mult = get_sector_multiplier(sector)
        raw_total = total
        if sector_mult != 1.0:
            total = min(100.0, max(0.0, total * sector_mult))

        n_models = len(components)
        confidence = round(available_weight / sum(self.WEIGHTS.values()), 4)

        # Model agreement: count how many models independently score above neutral (50)
        agreeing_models = sum(1 for s in components.values() if s >= 50)
        agreement_ratio = agreeing_models / n_models if n_models > 0 else 0

        # FDR-gated recommendation: require minimum model agreement for
        # actionable recommendations. This addresses the multiple testing
        # problem — with 9 models, some will score high by chance.
        min_agreement = self.MIN_AGREEING_MODELS
        agreement_met = agreeing_models >= min_agreement

        # Recommendation (gated by model agreement)
        if not agreement_met and total >= 55:
            # Downgrade: models disagree, ensemble score is misleading
            rec = "hold"
        elif total >= 75:
            rec = "strong_buy" if direction == "buy" else "strong_sell"
        elif total >= 55:
            rec = "buy" if direction == "buy" else "sell"
        elif total >= 40:
            rec = "hold"
        else:
            rec = "avoid"

        result = {
            "direction": direction,
            "total_score": round(total, 1),
            "raw_total_score": round(raw_total, 1),
            "sector": sector,
            "sector_multiplier": round(sector_mult, 3),
            "components": {k: round(v, 1) for k, v in components.items()},
            "n_models": n_models,
            "agreeing_models": agreeing_models,
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

    # --- Component scorers (each returns 0-100) ---

    def _score_monte_carlo(self, mc: dict, direction: str) -> float:
        h = mc.get("horizons", {})
        # Use 30-day horizon if available, else 90
        horizon = h.get(30) or h.get("30") or h.get(21) or h.get("21") or {}
        prob = horizon.get("probability_of_profit", 0.5)
        exp_ret = horizon.get("expected_return", 0)

        if direction == "sell":
            prob = 1 - prob
            exp_ret = -exp_ret

        # prob_profit contributes 60pts, expected_return contributes 40pts
        prob_score = prob * 60
        ret_score = min(40, max(0, (exp_ret + 0.10) / 0.20 * 40))  # -10% to +10% range
        return min(100, prob_score + ret_score)

    def _score_hmm(self, hmm: dict, direction: str) -> float:
        state = hmm.get("current_state", "sideways")
        probs = {
            "bull": hmm.get("prob_bull", 0) or 0,
            "bear": hmm.get("prob_bear", 0) or 0,
            "sideways": hmm.get("prob_sideways", 0) or 0,
        }
        if direction == "buy":
            return probs["bull"] * 100 + probs["sideways"] * 40
        else:  # sell
            return probs["bear"] * 100 + probs["sideways"] * 40

    def _score_garch(self, garch: dict) -> float:
        # Lower volatility = higher score (more predictable)
        # Contracting vol = better for entries
        ratio_5d = garch.get("forecast_5d_ratio") or 1.0
        ratio_20d = garch.get("forecast_20d_ratio") or 1.0
        avg_ratio = (ratio_5d + ratio_20d) / 2

        if avg_ratio < 0.8:
            return 80  # vol contracting significantly
        elif avg_ratio < 1.0:
            return 65  # vol mildly contracting
        elif avg_ratio < 1.2:
            return 45  # vol mildly expanding
        else:
            return 25  # vol expanding significantly

    def _score_fama_french(self, ff: dict) -> float:
        alpha = ff.get("alpha_annual") or 0
        # Positive alpha = outperformance
        alpha_score = min(60, max(0, (alpha + 0.10) / 0.20 * 60))

        r2 = ff.get("r_squared") or 0
        # Higher R² = model explains stock well = more reliable
        r2_score = r2 * 40

        return min(100, alpha_score + r2_score)

    def _score_copula(self, copula: dict) -> float:
        # Invert: low tail risk = high score
        tail_score = copula.get("tail_risk_score", 50)
        return max(0, 100 - tail_score)

    def _score_bayesian_decay(self, bd: dict) -> float:
        quality = bd.get("decay_quality", "no_signal")
        quality_map = {
            "slow_decay": 90,
            "moderate_decay": 70,
            "fast_decay": 45,
            "flash": 20,
            "no_alpha": 10,
            "no_signal": 5,
        }
        base = quality_map.get(quality, 30)

        # Boost if signal still strong at day 5
        strength_5d = bd.get("signal_strength", {}).get(5, {}).get("remaining_pct", 0)
        boost = min(10, strength_5d / 10)

        return min(100, base + boost)

    def _score_event_study(self, es: dict, direction: str) -> float:
        # Can be a single event dict or aggregate
        car_5d = es.get("car_5d") or es.get("mean_car_5d") or 0
        car_20d = es.get("car_20d") or es.get("mean_car_20d") or 0
        is_sig = es.get("is_significant", False)

        # Positive CAR = good signal
        car_avg = (car_5d + car_20d) / 2
        car_score = min(70, max(0, (car_avg + 0.05) / 0.10 * 70))

        # Significance bonus
        sig_bonus = 30 if is_sig else 0

        return min(100, car_score + sig_bonus)

    def _score_options_flow(self, opts: dict, direction: str) -> float:
        """Score options flow data 0-100 based on direction alignment."""
        score = 50.0  # neutral baseline

        # Put/Call Ratio alignment (0-35 points)
        pcr = opts.get("pcr")
        if pcr is not None:
            if direction == "buy":
                # Low PCR = bullish (more call buying)
                if pcr < 0.5:
                    score += 25
                elif pcr < 0.7:
                    score += 15
                elif pcr > 1.2:
                    score -= 20
                elif pcr > 1.0:
                    score -= 10
            else:  # sell
                # High PCR = bearish (more put buying)
                if pcr > 1.2:
                    score += 25
                elif pcr > 1.0:
                    score += 15
                elif pcr < 0.5:
                    score -= 20
                elif pcr < 0.7:
                    score -= 10

        # IV Skew alignment (0-20 points)
        iv_skew = opts.get("iv_skew")
        if iv_skew is not None:
            if direction == "buy":
                # Negative skew (calls more expensive) = bullish
                if iv_skew < -0.1:
                    score += 15
                elif iv_skew > 0.2:
                    score -= 10  # bearish skew on a buy signal
            else:
                # Positive skew (puts more expensive) = bearish
                if iv_skew > 0.2:
                    score += 15
                elif iv_skew < -0.1:
                    score -= 10

        # Unusual volume amplifier (0-15 points)
        uv = opts.get("unusual_volume_score", 0)
        if uv > 0.5:
            score += min(15, uv * 15)

        return max(0, min(100, score))

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

        def neg_corr(w):
            w = w / w.sum()  # normalize
            ensemble = scores_matrix @ w
            # Rank correlation (Spearman)
            from scipy.stats import spearmanr
            corr, _ = spearmanr(ensemble, returns_arr)
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
