# ML Model Review — Biases & Errors (2026-06-10)

Full review of every module in `src/analysis/`, `src/scoring/`, `src/clients/`
(model-relevant parts), and `src/pipeline/orchestrator.py`.

## Fixed in this review (11 issues)

| # | File | Issue | Impact |
|---|------|-------|--------|
| 1 | `scoring/macro_scorer.py` | Regime-based modifier block was dead code — unconditionally overwritten by regime_score scaling even when `regime_score` was None (defaulted to 0.5, turning an EXPANSION buy modifier of 1.2 into 0.95) | Macro layer systematically mis-scaled conviction whenever regime_score was missing |
| 2 | `analysis/monte_carlo.py` | Drift double-correction: `mu` is the mean of **log** returns (already μ−σ²/2), but the simulation subtracted σ²/2 again | Every expected return / P(profit) biased downward ~σ²/2 per day (≈5%/yr at 2% daily vol) |
| 3 | `analysis/event_study.py` | `car_5d/10d/20d` accumulated from day −5, mixing pre-event drift into "post-event alpha"; t-test and bootstrap also used the full window | CAR horizons systematically contaminated by pre-signal moves (insiders buy dips → negative bias; congressional leaks → positive bias) |
| 4 | `analysis/granger_causality.py` | Min p-value over 20 lags with no multiple-testing correction (≈64% false-positive rate under null) | "PREDICTIVE" verdicts wildly overstated; now Bonferroni-corrected |
| 5 | `analysis/copula_tail_risk.py` | t-copula likelihood dropped ν-dependent gamma constants | Biased degrees-of-freedom estimate → wrong tail-dependence λ. Verified post-fix: Gaussian data now correctly drives ν→bound |
| 6 | `analysis/ensemble_scoring.py` | Walk-forward weight calibration optimized Spearman with SLSQP — rank correlation is piecewise-constant in the weights, so the numerical gradient was 0 and the optimizer silently returned initial weights | Calibration was a no-op. Now maximizes Pearson corr vs fixed return ranks (smooth proxy); verified weights actually move |
| 7 | `analysis/xgboost_classifier.py` | `TimeSeriesSplit` on unsorted records | Temporal CV could leak future data into "past" folds; records now sorted by date |
| 8 | `clients/fama_french.py` + callers | FF factors lag publication by weeks, but stock returns were **tail-aligned** (`factors.iloc[-n:]` vs `returns[-n:]`) — every observation shifted by the lag | FF alpha/beta, copula dependence, and event-study market model all computed on mismatched dates. Now date-joined (`align_stock_to_factors`) |
| 9 | `analysis/ticker_analysis.py` | Earnings overlay and Bayesian decay computed once with `direction="buy"` then fed into **both** buy and sell ensembles | Sell scores polluted with buy-aligned components; now computed per direction |
| 10 | `pipeline/orchestrator.py` | OptionsFlow queried after `session.close()` | Works in SQLAlchemy but reopens a transaction implicitly; moved inside session scope |
| 11 | `api/api.py` | (introduced+fixed during this work) `sector` Query param broke internal plain-function callers | Dashboard endpoint 500 |

## Known biases flagged, NOT changed (strategy decisions)

These need a backtest before changing — they alter the strategy, not just code:

1. **Dip-buy tilt chases falling knives.** `conviction_engine._direction_boost`
   (+10% for ≥15% drawdown) and `fundamental_scorer._drawdown_opportunity_score`
   (max score at ≥20% drawdown) both reward buying drawdowns. The 2023 EDGAR
   backtest (see `reports/`) showed conviction↔return correlation ≈ 0 and the
   top-15 concentrated picks underperformed an equal-weight basket of all
   insider buys (+54% vs +99%). The breadth was the edge, not the selection.
   Recommendation: re-run the validation backtest with the drawdown components
   neutralized, and consider an equal-weight "all qualifying buys" portfolio
   mode alongside concentrated picks.
2. **Event studies anchor on `trade_date`, not `disclosure_date`.** A follower
   can only act at disclosure (Form 4: ≤2 business days; congressional: up to
   45 days). CARs from trade_date credit the signal with pre-disclosure moves
   that are not capturable. Recommendation: add a disclosure-anchored CAR
   variant and use it for any "followable alpha" claim.
3. **Stale sector medians.** `fundamental_scorer.py` P/E and short-interest
   medians are dated 2024-01-15 (the staleness warning now fires). Refresh
   from current S&P sector data or compute dynamically from the universe.
4. **IC monitor ICIR uses overlapping windows** (6-month window, 1-month step)
   — autocorrelated ICs understate the std, inflating ICIR. Treat the 0.5
   threshold as optimistic, or use non-overlapping windows.
5. **Survivorship in `earnings_overlay` backtests**: yfinance beat-rate history
   is as-of-today; in any backtest context it includes quarters after the
   signal. Fine live, lookahead in backtests.
6. **XGBoost missing-value imputation**: absent features default to 0.0, which
   is a meaningful value for momentum/alpha features. Consider explicit NaN
   handling (XGBoost supports native missing values).

## Verification performed

- Synthetic-data checks for fixes 1, 2, 3, 5, 6, 8 (see session log): macro
  modifier returns 1.2 in expansion; post-event CAR excludes injected
  pre-event drift; copula ν→bound on Gaussian data; calibrator moves the
  informative model's weight 0.16→0.50 cap; tz-aware date join intersects
  correctly.
- Live end-to-end `analyze_ticker("AAPL")`: 7 models, date-joined FF, per-
  direction earnings/decay, sensible HOLD verdict.
- New API endpoints exercised against the real `smart_money.db`.
