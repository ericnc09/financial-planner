"""
Tier D — Validation Framework.

D1: Out-of-sample backtest (70/30 temporal split) — prevents overfitting.
D2: Calibration analysis — score bucket 0-20…80-100 vs actual win rate.
D3: Model contribution (leave-one-out) — identifies noise models.
D4: Bootstrap confidence intervals — quantifies metric uncertainty.
D5: Calibration feedback (isotonic regression) — maps raw scores to calibrated
    probabilities so the live system acts on accurate win-rate estimates.
D6: Deflated Sharpe Ratio — corrects for multiple testing at strategy level.

All functions accept a list of signal records with realized returns.
They work standalone (no database required) when you pass records directly,
or integrate with the Backtester for database-sourced records.
"""

import math
import random
from typing import Callable

import numpy as np
import structlog

logger = structlog.get_logger()


# ── D1: Out-of-Sample Split ───────────────────────────────────────────────────

def oos_split(
    records: list[dict],
    train_pct: float = 0.70,
    sort_key: str = "date",
) -> tuple[list[dict], list[dict]]:
    """
    Split records temporally into train and test sets.
    Records are sorted by `sort_key` before splitting so no future data leaks.

    Returns:
        (train_records, test_records)
    """
    sorted_recs = sorted(records, key=lambda r: r.get(sort_key, ""))
    split_idx = int(len(sorted_recs) * train_pct)
    train = sorted_recs[:split_idx]
    test = sorted_recs[split_idx:]
    logger.info("oos_split", n_train=len(train), n_test=len(test), train_pct=train_pct)
    return train, test


def evaluate_split(records: list[dict], return_field: str = "realized_return") -> dict:
    """Compute basic metrics on a set of records with realized returns."""
    returns = [r[return_field] for r in records if r.get(return_field) is not None]
    if not returns:
        return {"n": 0, "status": "no_data"}

    n = len(returns)
    wins = [r for r in returns if r > 0]
    avg_ret = sum(returns) / n
    win_rate = len(wins) / n

    if n > 1:
        std = float(np.std(returns, ddof=1))
        sharpe = (avg_ret / std * math.sqrt(252)) if std > 0 else 0.0
    else:
        std = 0.0
        sharpe = 0.0

    return {
        "n": n,
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_ret, 4),
        "sharpe_ratio": round(sharpe, 4),
        "std": round(std, 4),
    }


def run_oos_validation(
    records: list[dict],
    train_pct: float = 0.70,
    score_field: str = "ensemble_score",
    return_field: str = "realized_return",
) -> dict:
    """
    D1: Full OOS validation report.

    Args:
        records: Each must have `date`, `ensemble_score`, `realized_return`.

    Returns:
        Dict with train/test metrics and overfit diagnostics.
    """
    if len(records) < 20:
        return {"status": "insufficient_data", "n": len(records), "min_required": 20}

    train, test = oos_split(records, train_pct)

    train_metrics = evaluate_split(train, return_field)
    test_metrics = evaluate_split(test, return_field)

    sharpe_decay = None
    wr_decay = None
    if train_metrics["n"] > 0 and test_metrics["n"] > 0:
        sharpe_decay = train_metrics["sharpe_ratio"] - test_metrics["sharpe_ratio"]
        wr_decay = train_metrics["win_rate"] - test_metrics["win_rate"]

    return {
        "status": "complete",
        "train_pct": train_pct,
        "train": train_metrics,
        "test": test_metrics,
        "overfit": {
            "sharpe_decay": round(sharpe_decay, 4) if sharpe_decay is not None else None,
            "win_rate_decay": round(wr_decay, 4) if wr_decay is not None else None,
            "is_overfit": sharpe_decay is not None and sharpe_decay > 0.5,
            "verdict": (
                "OVERFIT" if (sharpe_decay or 0) > 0.5
                else "GENERALIZING" if (sharpe_decay or 0) < 0.1
                else "MILD_OVERFIT"
            ),
        },
    }


# ── D2: Calibration Analysis ──────────────────────────────────────────────────

CALIBRATION_BINS = [(0, 20), (20, 40), (40, 60), (60, 75), (75, 90), (90, 100)]


def calibration_analysis(
    records: list[dict],
    score_field: str = "ensemble_score",
    return_field: str = "realized_return",
) -> dict:
    """
    D2: For each score bucket, compute actual win rate.

    A well-calibrated model should have score=75 → ~75% win rate.

    Args:
        records: Each must have `ensemble_score` (0-100) and `realized_return`.

    Returns:
        Dict with per-bucket stats and calibration error.
    """
    buckets: dict[str, list[float]] = {}
    for lo, hi in CALIBRATION_BINS:
        label = f"{lo}-{hi}"
        buckets[label] = []

    for r in records:
        score = r.get(score_field)
        ret = r.get(return_field)
        if score is None or ret is None:
            continue
        for lo, hi in CALIBRATION_BINS:
            if lo <= score < hi or (hi == 100 and score == 100):
                buckets[f"{lo}-{hi}"].append(1.0 if ret > 0 else 0.0)
                break

    calibration = {}
    calibration_errors = []

    for (lo, hi), label in [(b, f"{b[0]}-{b[1]}") for b in CALIBRATION_BINS]:
        outcomes = buckets[label]
        if not outcomes:
            calibration[label] = {"n": 0, "actual_win_rate": None, "expected_win_rate": (lo + hi) / 200, "calibration_error": None}
            continue

        actual_wr = sum(outcomes) / len(outcomes)
        expected_wr = (lo + hi) / 200  # midpoint of bucket / 100
        error = abs(actual_wr - expected_wr)
        calibration_errors.append(error)

        calibration[label] = {
            "n": len(outcomes),
            "actual_win_rate": round(actual_wr, 4),
            "expected_win_rate": round(expected_wr, 4),
            "calibration_error": round(error, 4),
        }

    mean_cal_error = round(sum(calibration_errors) / len(calibration_errors), 4) if calibration_errors else None

    logger.info("calibration.computed", mean_error=mean_cal_error, n_buckets=len(CALIBRATION_BINS))

    return {
        "buckets": calibration,
        "mean_calibration_error": mean_cal_error,
        "well_calibrated": mean_cal_error is not None and mean_cal_error < 0.10,
    }


def format_calibration_report(cal: dict) -> str:
    """Human-readable calibration table."""
    lines = [
        "CALIBRATION REPORT  (expected vs actual win rate by score bucket)",
        "=" * 65,
        f"{'Score Bucket':<14} {'N':>6} {'Expected WR':>13} {'Actual WR':>11} {'Error':>8}",
        "-" * 65,
    ]
    for bucket, s in cal["buckets"].items():
        if s["n"] == 0:
            lines.append(f"{bucket:<14} {'—':>6}")
            continue
        lines.append(
            f"{bucket:<14} {s['n']:>6} {s['expected_win_rate']:>12.1%} "
            f"{s['actual_win_rate']:>10.1%} {s['calibration_error']:>7.1%}"
        )
    lines.append("-" * 65)
    mce = cal.get("mean_calibration_error")
    wc = "YES" if cal.get("well_calibrated") else "NO"
    lines.append(f"Mean calibration error: {mce:.3f}  |  Well-calibrated: {wc}")
    return "\n".join(lines)


# ── D3: Model Contribution (Leave-One-Out) ────────────────────────────────────

def model_contribution_analysis(
    records: list[dict],
    score_fn: Callable[[dict, list[str]], float],
    model_names: list[str],
    return_field: str = "realized_return",
) -> dict:
    """
    D3: Run backtest excluding each model one-at-a-time.

    If removing a model improves Sharpe, that model adds noise.

    Args:
        records: Signal records with component scores and realized returns.
        score_fn: Callable(record, excluded_models) → ensemble_score (0-100).
                  Receives the record dict and a list of model names to exclude.
        model_names: Names of all models in the ensemble.
        return_field: Key for realized return in each record.

    Returns:
        Dict with baseline metrics and per-model LOO delta Sharpe.
    """
    if len(records) < 20:
        return {"status": "insufficient_data", "n": len(records)}

    returns = [r[return_field] for r in records if r.get(return_field) is not None]
    baseline = _returns_to_metrics(returns)

    contributions = {}
    for model in model_names:
        loo_returns = []
        for r in records:
            ret = r.get(return_field)
            if ret is None:
                continue
            # Use provided scoring function to re-score without this model
            loo_score = score_fn(r, [model])
            # Only include signal if it would still pass a neutral threshold
            if loo_score >= 50:
                loo_returns.append(ret)

        loo_metrics = _returns_to_metrics(loo_returns)
        delta_sharpe = loo_metrics["sharpe_ratio"] - baseline["sharpe_ratio"]

        contributions[model] = {
            "n_signals": len(loo_returns),
            "sharpe_ratio": loo_metrics["sharpe_ratio"],
            "win_rate": loo_metrics["win_rate"],
            "delta_sharpe": round(delta_sharpe, 4),
            "verdict": (
                "REMOVES_NOISE" if delta_sharpe > 0.05
                else "NEUTRAL" if abs(delta_sharpe) <= 0.05
                else "ADDS_ALPHA"
            ),
        }

    # Sort by delta_sharpe descending (most noisy first)
    contributions = dict(sorted(
        contributions.items(), key=lambda x: -x[1]["delta_sharpe"]
    ))

    logger.info(
        "model_contribution.computed",
        baseline_sharpe=baseline["sharpe_ratio"],
        top_noise=next(iter(contributions)) if contributions else None,
    )

    return {
        "baseline": baseline,
        "model_contributions": contributions,
        "noisy_models": [m for m, v in contributions.items() if v["verdict"] == "REMOVES_NOISE"],
        "alpha_models": [m for m, v in contributions.items() if v["verdict"] == "ADDS_ALPHA"],
    }


def format_contribution_report(result: dict) -> str:
    """Human-readable model contribution table."""
    if "status" in result:
        return f"Model contribution: {result['status']}"

    lines = [
        "MODEL CONTRIBUTION ANALYSIS  (Leave-One-Out)",
        f"Baseline Sharpe: {result['baseline']['sharpe_ratio']:.3f}  "
        f"Win Rate: {result['baseline']['win_rate']:.1%}  "
        f"N={result['baseline']['n']}",
        "=" * 65,
        f"{'Model':<20} {'LOO Sharpe':>12} {'Delta':>10} {'Verdict':>16}",
        "-" * 65,
    ]
    for model, m in result["model_contributions"].items():
        symbol = "+" if m["delta_sharpe"] > 0 else ""
        lines.append(
            f"{model:<20} {m['sharpe_ratio']:>12.3f} "
            f"{symbol}{m['delta_sharpe']:>9.3f} {m['verdict']:>16}"
        )
    lines.append("-" * 65)
    noisy = result.get("noisy_models", [])
    alpha = result.get("alpha_models", [])
    if noisy:
        lines.append(f"Noisy (consider removing): {', '.join(noisy)}")
    if alpha:
        lines.append(f"Alpha-adding (keep): {', '.join(alpha)}")
    return "\n".join(lines)


# ── D4: Bootstrap Confidence Intervals ───────────────────────────────────────

def bootstrap_metrics(
    returns: list[float],
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    D4: Bootstrap confidence intervals on Sharpe ratio, win rate, and avg return.

    Args:
        returns: List of realized returns (decimal fractions).
        n_bootstrap: Number of bootstrap samples.
        ci: Confidence interval width (0.95 = 95%).

    Returns:
        Dict with point estimates and CI lower/upper for each metric.
    """
    if len(returns) < 10:
        return {"status": "insufficient_data", "n": len(returns)}

    rng = np.random.default_rng(seed)
    arr = np.array(returns)

    sharpes, win_rates, avg_rets = [], [], []
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = (1 + ci) / 2 * 100

    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        n = len(sample)
        wr = float(np.mean(sample > 0))
        avg_r = float(np.mean(sample))
        std = float(np.std(sample, ddof=1))
        sharpe = (avg_r / std * math.sqrt(252)) if std > 0 else 0.0

        sharpes.append(sharpe)
        win_rates.append(wr)
        avg_rets.append(avg_r)

    def _ci(vals):
        arr_v = np.array(vals)
        return {
            "point": round(float(np.mean(arr_v)), 4),
            "lower": round(float(np.percentile(arr_v, lo_pct)), 4),
            "upper": round(float(np.percentile(arr_v, hi_pct)), 4),
        }

    result = {
        "n": len(returns),
        "n_bootstrap": n_bootstrap,
        "confidence_level": ci,
        "sharpe_ratio": _ci(sharpes),
        "win_rate": _ci(win_rates),
        "avg_return": _ci(avg_rets),
    }

    logger.info(
        "bootstrap.computed",
        n=len(returns), sharpe=result["sharpe_ratio"]["point"],
        sharpe_lo=result["sharpe_ratio"]["lower"],
        sharpe_hi=result["sharpe_ratio"]["upper"],
    )
    return result


def format_bootstrap_report(result: dict) -> str:
    """Human-readable bootstrap CI table."""
    if "status" in result:
        return f"Bootstrap CI: {result['status']}"

    ci_pct = int(result["confidence_level"] * 100)
    lines = [
        f"BOOTSTRAP CONFIDENCE INTERVALS  ({ci_pct}% CI, {result['n_bootstrap']} samples, N={result['n']})",
        "=" * 65,
        f"{'Metric':<20} {'Point Est':>12} {'CI Lower':>12} {'CI Upper':>12}",
        "-" * 65,
    ]
    for metric, label in [
        ("sharpe_ratio", "Sharpe Ratio"),
        ("win_rate", "Win Rate"),
        ("avg_return", "Avg Return"),
    ]:
        m = result[metric]
        lines.append(f"{label:<20} {m['point']:>12.4f} {m['lower']:>12.4f} {m['upper']:>12.4f}")
    return "\n".join(lines)


# ── D5: Calibration Feedback (Isotonic Regression) ──────────────────────────

class ScoreCalibrator:
    """
    Maps raw ensemble scores to calibrated win-rate probabilities using
    isotonic regression. Unlike bucketed calibration (D2), this produces a
    smooth, monotonic mapping that can be applied live.

    Usage:
        calibrator = ScoreCalibrator()
        result = calibrator.fit(records)
        calibrated = calibrator.calibrate(75.0)  # → actual probability
    """

    def __init__(self):
        self.isotonic = None
        self.is_fitted = False
        self.fit_stats = None

    def fit(
        self,
        records: list[dict],
        score_field: str = "ensemble_score",
        return_field: str = "realized_return",
    ) -> dict:
        """
        Fit isotonic regression: raw_score → P(win).

        Args:
            records: Historical records with scores and realized returns.

        Returns:
            Dict with fit diagnostics.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            return {"status": "error", "message": "scikit-learn not installed"}

        scores = []
        outcomes = []
        for r in records:
            s = r.get(score_field)
            ret = r.get(return_field)
            if s is not None and ret is not None:
                scores.append(float(s))
                outcomes.append(1.0 if ret > 0 else 0.0)

        if len(scores) < 30:
            return {"status": "insufficient_data", "n": len(scores), "min_required": 30}

        X = np.array(scores)
        y = np.array(outcomes)

        self.isotonic = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip",
        )
        self.isotonic.fit(X, y)
        self.is_fitted = True

        # Diagnostics: check calibration at key score levels
        test_scores = [20, 40, 55, 65, 75, 85, 95]
        score_map = {}
        for ts in test_scores:
            calibrated = float(self.isotonic.predict([ts])[0])
            score_map[ts] = round(calibrated, 4)

        # Mean absolute calibration error (on training data)
        predicted_probs = self.isotonic.predict(X)
        cal_error = float(np.mean(np.abs(predicted_probs - y)))

        self.fit_stats = {
            "status": "fitted",
            "n_samples": len(scores),
            "calibration_map": score_map,
            "mean_absolute_error": round(cal_error, 4),
        }

        logger.info(
            "calibrator.fitted",
            n=len(scores),
            map=score_map,
            mae=round(cal_error, 4),
        )
        return self.fit_stats

    def calibrate(self, raw_score: float) -> float | None:
        """Map a raw ensemble score to a calibrated win probability."""
        if not self.is_fitted:
            return None
        return float(self.isotonic.predict([raw_score])[0])

    def calibrate_batch(
        self,
        signals: list[dict],
        score_field: str = "ensemble_score",
    ) -> list[dict]:
        """Add calibrated_probability to each signal dict."""
        if not self.is_fitted:
            return signals
        for sig in signals:
            score = sig.get(score_field)
            if score is not None:
                sig["calibrated_probability"] = round(self.calibrate(float(score)), 4)
        return signals


def format_calibrator_report(fit_result: dict) -> str:
    """Human-readable calibrator report."""
    if fit_result.get("status") != "fitted":
        return f"Score Calibrator: {fit_result.get('status', 'unknown')}"

    lines = [
        "SCORE CALIBRATION  (Isotonic Regression: raw score → P(win))",
        "=" * 55,
        f"Training samples: {fit_result['n_samples']}",
        f"Mean absolute error: {fit_result['mean_absolute_error']:.4f}",
        "",
        f"{'Raw Score':>12} {'→ P(win)':>12}",
        "-" * 26,
    ]
    for score, prob in fit_result["calibration_map"].items():
        lines.append(f"{score:>12} {prob:>12.1%}")
    return "\n".join(lines)


# ── D6: Deflated Sharpe Ratio ───────────────────────────────────────────────

def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_strategies_tried: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> dict:
    """
    D6: Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Adjusts the observed Sharpe ratio for:
    - Number of strategies tried (multiple testing)
    - Non-normality of returns (skewness, kurtosis)
    - Sample size

    A DSR p-value > 0.05 means the Sharpe is not significantly better
    than what you'd get by luck given the number of strategies tested.

    Args:
        observed_sharpe: Annualized Sharpe ratio of the chosen strategy.
        n_observations: Number of return observations.
        n_strategies_tried: Total strategies/parameter combos tested.
        skewness: Return distribution skewness (0 = normal).
        kurtosis: Return distribution kurtosis (3 = normal).

    Returns:
        Dict with DSR statistic, p-value, and verdict.
    """
    from scipy.stats import norm

    if n_observations < 10 or n_strategies_tried < 1:
        return {"status": "insufficient_data"}

    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    if n_strategies_tried > 1:
        expected_max_sharpe = (
            norm.ppf(1 - 1 / n_strategies_tried)
            * (1 - euler_mascheroni)
            + euler_mascheroni * norm.ppf(1 - 1 / (n_strategies_tried * math.e))
        )
    else:
        expected_max_sharpe = 0.0

    # Standard error of Sharpe ratio (Lo, 2002) with non-normality correction
    se_sharpe = math.sqrt(
        (1 + 0.5 * observed_sharpe**2 - skewness * observed_sharpe
         + ((kurtosis - 3) / 4) * observed_sharpe**2)
        / (n_observations - 1)
    )

    if se_sharpe == 0:
        return {"status": "zero_variance"}

    # DSR test statistic
    dsr_stat = (observed_sharpe - expected_max_sharpe) / se_sharpe
    p_value = 1 - norm.cdf(dsr_stat)

    is_significant = p_value < 0.05

    return {
        "status": "complete",
        "observed_sharpe": round(observed_sharpe, 4),
        "expected_max_sharpe": round(expected_max_sharpe, 4),
        "se_sharpe": round(se_sharpe, 4),
        "dsr_statistic": round(dsr_stat, 4),
        "p_value": round(p_value, 4),
        "n_observations": n_observations,
        "n_strategies_tried": n_strategies_tried,
        "is_significant": is_significant,
        "verdict": (
            "SIGNIFICANT — Sharpe survives multiple testing correction"
            if is_significant
            else "NOT_SIGNIFICANT — Sharpe could be due to luck given strategies tried"
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _returns_to_metrics(returns: list[float]) -> dict:
    if not returns:
        return {"n": 0, "sharpe_ratio": 0.0, "win_rate": 0.0, "avg_return": 0.0}
    n = len(returns)
    avg_ret = sum(returns) / n
    wins = sum(1 for r in returns if r > 0)
    std = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    sharpe = (avg_ret / std * math.sqrt(252)) if std > 0 else 0.0
    return {
        "n": n,
        "win_rate": round(wins / n, 4),
        "avg_return": round(avg_ret, 4),
        "sharpe_ratio": round(sharpe, 4),
    }


if __name__ == "__main__":
    # Demo mode — runs validation on synthetic data when no pipeline records available.
    import numpy as np
    rng = np.random.default_rng(42)

    print("Validation Framework Demo (synthetic data)\n")

    # Generate synthetic signal records
    records = []
    for i in range(200):
        score = float(rng.uniform(20, 95))
        # Simulate: higher score → slightly better expected return
        ret = float(rng.normal(loc=(score - 50) * 0.002, scale=0.05))
        records.append({
            "date": f"2024-{(i % 12)+1:02d}-{(i % 28)+1:02d}",
            "ensemble_score": score,
            "realized_return": ret,
        })

    # D1: OOS validation
    oos = run_oos_validation(records)
    print("D1 — Out-of-Sample Validation")
    print(f"  Train: Sharpe={oos['train']['sharpe_ratio']:.3f}  WinRate={oos['train']['win_rate']:.1%}  N={oos['train']['n']}")
    print(f"  Test:  Sharpe={oos['test']['sharpe_ratio']:.3f}  WinRate={oos['test']['win_rate']:.1%}  N={oos['test']['n']}")
    print(f"  Overfit verdict: {oos['overfit']['verdict']}\n")

    # D2: Calibration
    cal = calibration_analysis(records)
    print("D2 — Calibration Analysis")
    print(format_calibration_report(cal))
    print()

    # D4: Bootstrap CI
    all_returns = [r["realized_return"] for r in records]
    bs = bootstrap_metrics(all_returns)
    print("D4 — Bootstrap Confidence Intervals")
    print(format_bootstrap_report(bs))
    print()

    print("D3 — Model Contribution (LOO) requires pipeline data with component scores.")
    print("    Run after accumulating 50+ signals via: make run-daemon")
