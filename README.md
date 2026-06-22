# Financial Market Analyzer

Research system for market, ticker, signal, and portfolio analysis.

This project is being converted from a disclosure-driven "Smart Money Follows"
app into a broader financial market analyzer. The current system still centers
on insider, congressional, and 13F disclosure events, but it now has a clearer
model-audit workflow, broader component persistence, and explicit validation
gates so claims can be checked against the actual database.

This is research software. It is not financial advice.

---

## Current Status

Status is generated from `smart_money.db` by:

```bash
make model-status
```

Latest report: [reports/model_status_report.json](reports/model_status_report.json)

As of the current report generated on 2026-06-22:

| Area | Current Result |
|---|---:|
| Events | 202 |
| Enrichments | 189 |
| Conviction scores | 189 |
| Ensemble score rows | 492 |
| Latest ensemble rows | 189 |
| Signal performance rows | 202 |
| Model validation gates | 3 / 7 passing |
| Overall validation status | fail |
| XGBoost model artifact | not trained |

Important truth-in-labeling notes:

- The system does not currently have a fully populated 11-component live
  ensemble in the existing database.
- Latest persisted ensemble rows currently contain mostly Monte Carlo, HMM,
  GARCH, Fama-French, and copula components.
- Event study, Bayesian decay, news sentiment, and several newly persisted
  ensemble component columns are still empty in latest rows until the pipeline
  is rerun and/or maturity-retry logic is added.
- Current 20-day evidence shows some value in ensemble BUY buckets, but
  conviction ranking and sell-side recommendations are not validated yet.

Current validation gates:

| Gate | Status |
|---|---|
| Minimum 20-day samples | pass |
| Ensemble rank IC positive | pass |
| BUY bucket beats HOLD | pass |
| Conviction rank IC positive | fail |
| SELL bucket positive | fail |
| High-conviction beats other | fail |
| All claimed components present | fail |

Current 20-day performance snapshot:

| Slice | N | Avg Return | Win Rate |
|---|---:|---:|---:|
| Ensemble BUY recommendations | 34 | 8.47% | 76.47% |
| Ensemble HOLD recommendations | 79 | -1.95% | 40.51% |
| Ensemble SELL recommendations | 17 | -6.02% | 52.94% |
| High-conviction signals | 88 | 0.92% | 54.55% |
| Other conviction signals | 46 | 1.26% | 50.00% |
| All BUY directions | 109 | 5.85% | 59.63% |
| All SELL directions | 56 | -2.18% | 50.00% |

Score correlations against realized 20-day returns:

| Score | N | Pearson | Spearman |
|---|---:|---:|---:|
| Conviction | 134 | -0.0061 | -0.0264 |
| Latest ensemble | 134 | -0.0598 | 0.0995 |

---

## What The System Does

The system currently supports five product surfaces:

1. Market context
   - FRED macro snapshots
   - Extended macro indicators
   - Macro regime scoring

2. Ticker analysis
   - Price history
   - Momentum, RSI, drawdown, liquidity
   - GARCH volatility forecasts
   - HMM regime state
   - Fama-French factor exposure
   - Copula tail-risk analysis
   - Options-flow data where available

3. Disclosure signal engine
   - SEC Form 4 insider trades
   - Congressional trades
   - SEC 13F institutional holdings diffs
   - Signal, fundamental, and macro conviction scoring

4. Portfolio and risk
   - Mean-variance optimization
   - Tail-risk metrics
   - Performance tracking
   - Backtesting

5. Validation and model governance
   - Model coverage report
   - Score calibration report
   - Bootstrap confidence intervals
   - Deflated Sharpe check
   - Acceptance gates for product claims
   - XGBoost training status and model-card metadata

---

## Quick Start

### Requirements

- Python 3.10+
- Node.js 18+
- SQLite by default
- A free FRED API key for macro data

Optional:

- Finnhub API key for company news
- Tiingo API key as price/enrichment fallback
- BLS API key for higher quota
- Slack webhook for alerts
- Sentry DSN for monitoring
- Admin token for mutating API endpoints

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd dashboard
npm install
cd ..
```

Create `config/.env`:

```env
FRED_API_KEY=your_fred_key_here
DATABASE_URL=sqlite:///./smart_money.db

# Optional
TIINGO_API_KEY=
FINNHUB_API_KEY=
BLS_API_KEY=
SLACK_WEBHOOK_URL=
SENTRY_DSN=

# Required for mutating API endpoints
ADMIN_API_TOKEN=generate_a_long_random_string

CONVICTION_THRESHOLD=0.6
PIPELINE_MODE=oneshot
SCHEDULE_INTERVAL_MINUTES=60
LOG_LEVEL=INFO
```

### Run

```bash
make init-db
make run-once
make run-api
```

In a second terminal:

```bash
cd dashboard
npm run dev
```

Default URLs:

- Dashboard: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`

---

## Core Commands

```bash
# Initialize or upgrade SQLite schema
make init-db

# Run one ingest/enrich/analyze/score cycle
make run-once

# Run scheduled pipeline loop
make run-daemon

# Start FastAPI
make run-api

# Start dashboard
make run-dashboard

# Update realized signal performance
.venv/bin/python -m src.tracking.performance

# Run backtest
make backtest

# Run validation/demo framework
make validate

# Generate model coverage and validation report
make model-status

# Python tests
.venv/bin/pytest -q

# Dashboard typecheck/build
npm run --prefix dashboard build
```

---

## Model Status Matrix

The table below separates implemented code from currently populated and
validated production data.

| Component | Implemented | Persisted | In Latest Ensemble Rows | Current Status |
|---|---:|---:|---:|---|
| Monte Carlo | yes | yes | yes, 89.42% coverage | active |
| HMM regime | yes | yes | yes, 88.89% coverage | active |
| GARCH | yes | yes | yes, 88.89% coverage | active |
| Fama-French | yes | yes | yes, 89.42% coverage | active |
| Copula tail risk | yes | yes | yes, 88.89% coverage | active |
| Mean-variance | yes | yes | not an ensemble component | active portfolio/risk output |
| Options flow | yes | yes | not in current latest rows | newly wired for future ensemble persistence |
| Earnings overlay | yes | component only | not in current latest rows | newly wired for future ensemble persistence |
| News sentiment | yes | table exists | no current rows | inactive until provider/run data exists |
| Weather overlay | yes | component only | not in current latest rows | newly wired for future ensemble persistence |
| Event study | yes | table exists | no current rows | needs maturity retry and disclosure-date variant |
| Bayesian decay | yes | table exists | no current rows | needs pipeline persistence/maturity workflow |
| XGBoost classifier | yes | artifact path exists | not trained | training endpoint now saves model when trained |

New schema fields are upgraded automatically for SQLite by `init_db()`:

- `ensemble_score_results.score_options_flow`
- `ensemble_score_results.score_earnings_overlay`
- `ensemble_score_results.score_news_sentiment`
- `ensemble_score_results.score_weather_overlay`
- `enrichments.short_ratio`

---

## Validation Workflow

Run:

```bash
make model-status
```

This writes [reports/model_status_report.json](reports/model_status_report.json)
and reports:

- table row counts
- latest ensemble model-count distribution
- component presence and coverage
- 20-day conviction and ensemble rank correlations
- component-level correlations
- recommendation bucket performance
- conviction bucket performance
- source and direction performance
- calibration buckets
- isotonic calibration diagnostics
- bootstrap confidence intervals
- deflated Sharpe checks
- acceptance gates
- XGBoost training/artifact status

Use this report as the source of truth before making model-performance claims.

---

## XGBoost Status

The XGBoost classifier is implemented but not currently trained in the checked
database state.

Improvements now in place:

- Training records include the declared model features where persisted:
  HMM probabilities, GARCH vol, Fama-French alpha, copula score, news fields,
  short ratio, enrichment fields, conviction fields, and direction encoding.
- Missing feature values are explicitly median-imputed instead of silently
  converted to `0.0`.
- Training metadata includes CV AUC, overfit flag, selected features, and
  imputation policy.
- Trained models are saved to `src/analysis/xgboost_model.pkl`.
- API startup loads the saved classifier if present.
- Predictions are refused when the trained model is not meaningfully better
  than the null threshold.

Admin endpoint:

```http
POST /api/ml/xgboost/train
X-Admin-Token: <ADMIN_API_TOKEN>
```

---

## API Overview

Important endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness and DB check |
| `/api/signals` | GET | Recent disclosure signals |
| `/api/signals/{ticker}` | GET | Signals for one ticker |
| `/api/top-pick` | GET | Current top pick from recent BUY signals |
| `/api/sectors` | GET | Sector/industry activity summary |
| `/api/dashboard` | GET | Dashboard aggregate payload |
| `/api/macro` | GET | Latest macro regime snapshot |
| `/api/macro/extended` | GET | Extended macro indicators |
| `/api/macro/history` | GET | Historical macro snapshots |
| `/api/macro/extra` | GET | BLS and Treasury macro data |
| `/api/prices/{ticker}` | GET | Historical price data |
| `/api/analysis/{ticker}` | GET | Model analysis for one ticker |
| `/api/analysis/{ticker}/options` | GET | Latest options-flow analysis |
| `/api/analysis/hmm/all` | GET | Latest HMM states |
| `/api/analysis/mean-variance` | GET | Portfolio optimizer output |
| `/api/analysis/ensemble/all` | GET | Latest ensemble scores |
| `/api/analysis/event-study/summary` | GET | Event-study aggregate, currently empty if no rows |
| `/api/news/{ticker}` | GET | Persisted news sentiment for ticker |
| `/api/performance/summary` | GET | Realized performance summary |
| `/api/backtest` | POST | Backtest by date range and threshold |
| `/api/export/signals` | GET | CSV/JSON signal export |
| `/api/export/analysis/{ticker}` | GET | CSV/JSON model export |
| `/api/pipeline/run` | POST | Admin-only pipeline trigger |
| `/api/performance/update` | POST | Admin-only performance refresh |
| `/api/ml/xgboost/train` | POST | Admin-only classifier training |
| `/api/ml/xgboost/predict` | POST | Classifier prediction if trained/reliable |

Admin endpoints require `ADMIN_API_TOKEN`. If the token is unset, mutating
endpoints return `503` by design.

---

## Dashboard

The React dashboard currently includes:

- Top Pick card
- Sector panel
- Signal table
- Macro gauge
- Extended macro panel
- Performance panel
- Backtest panel
- Stock price chart
- Mean-variance chart
- Ticker detail view
- Monte Carlo chart
- GARCH forecast
- Fama-French exposure
- Event study chart, empty until event-study rows exist
- Copula tail-risk gauge
- Bayesian decay chart, empty until Bayesian decay rows exist
- Ensemble score breakdown
- Options-flow panel
- Conviction bar

The dashboard build is verified with:

```bash
npm run --prefix dashboard build
```

---

## Data Sources

| Source | Current Use |
|---|---|
| SEC EDGAR Form 4 | Insider-trade ingestion |
| SEC EDGAR XBRL | Point-in-time fundamentals |
| SEC EDGAR 13F | Institutional holdings diffs |
| Capitol Trades | Congressional trades |
| Yahoo Finance | Prices, options, enrichment |
| FRED | Macro indicators |
| BLS | Payrolls and unemployment, optional key |
| Treasury FiscalData | Treasury rates and debt |
| Finnhub | Company news, optional key |
| GDELT | Keyless news fallback |
| Open-Meteo/NWS | Weather overlay |
| Fama-French data | Factor model and market returns |
| Tiingo | Optional fallback |

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for more detail.

---

## Repository Layout

```text
financial-planner/
|-- config/
|   `-- settings.py
|-- dashboard/
|   `-- src/
|-- docs/
|   |-- DATA_SOURCES.md
|   |-- ML_MODEL_REVIEW.md
|   |-- PUBLIC_LAUNCH_ROADMAP.md
|   `-- DEPLOYMENT_ROADMAP.md
|-- reports/
|   |-- model_status_report.json
|   `-- performance_vs_benchmarks.json
|-- scripts/
|   |-- model_status_report.py
|   |-- performance_vs_benchmarks.py
|   `-- run_analysis.py
|-- src/
|   |-- analysis/
|   |-- api/
|   |-- backtesting/
|   |-- clients/
|   |-- models/
|   |-- pipeline/
|   |-- scoring/
|   `-- tracking/
`-- tests/
```

---

## Known Gaps

Highest-priority remaining work:

1. Event study maturity retry
   - Current event-study table has zero rows.
   - Add a retry job for events that have matured past the post-event window.
   - Add disclosure-date-anchored event studies for followable-alpha claims.

2. Bayesian decay persistence
   - Current Bayesian decay table has zero rows.
   - Persist signal-specific decay results after enough realized abnormal
     return history exists.

3. Full component coverage
   - New columns now exist for options, earnings, news, and weather component
     scores, but latest rows predate the persistence change.
   - Rerun the pipeline and use `make model-status` to verify coverage.

4. Sell-side weakness
   - Current SELL direction and ensemble SELL bucket performance are negative.
   - Treat bearish output as diagnostic until sell-side evidence improves.

5. Conviction score weakness
   - Current conviction Spearman is negative.
   - Rework or demote conviction ranking until it beats baseline gates.

6. Public/product language
   - Do not claim fully validated alpha.
   - Use the model-status report as the public truth source.

7. Broader market-analyzer pivot
   - Add watchlist/universe analysis independent of new disclosure events.
   - Separate market analysis, ticker analysis, signal generation, and
     portfolio/risk workflows in the UI.

---

## Verification

Current verification commands:

```bash
.venv/bin/pytest -q
npm run --prefix dashboard build
.venv/bin/python scripts/model_status_report.py
```

Current passing state after the latest implementation:

- Python tests: 11 passed
- Dashboard build: passed
- Model status report generation: passed

---

## Disclaimer

This project is for research and education only. It does not provide financial,
investment, tax, or legal advice. Past performance, backtests, and model scores
do not guarantee future returns. Always perform independent due diligence.

---

## License

MIT License. See [LICENSE](LICENSE).
