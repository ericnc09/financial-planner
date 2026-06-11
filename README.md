# Smart Money Follows

**Track what insiders, Congress, and elite funds are buying before the market catches on.**

Smart Money Follows is an automated trading signal platform that ingests insider, congressional, and 13F institutional trade disclosures, enriches them with point-in-time SEC XBRL fundamentals, news sentiment, weather, and macro data, runs an 11-component quantitative ensemble, and produces a scored, ranked feed of trading signals — including a daily **Morning Top Pick** — through a real-time React dashboard with industry-level filtering.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the dashboard)
- Free API key: [FRED](https://fred.stlouisfed.org/docs/api/api_key.html) (required for macro data)
- Optional: [Tiingo](https://www.tiingo.com/) API key (fallback enrichment), Slack webhook URL (alerts)

### Installation

```bash
# Clone the repo
git clone https://github.com/ericnc09/financial-planner.git
cd financial-planner

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp config/.env.example config/.env
# Edit config/.env — add at minimum your FRED_API_KEY
```

### Configure Environment

Edit `config/.env`:

```env
# Required
FRED_API_KEY=your_fred_key_here

# Optional — Tiingo is fallback only (yfinance is primary)
TIINGO_API_KEY=your_tiingo_key_here

# Optional — Slack alerts for high-conviction signals
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Optional — Finnhub (free, 60 req/min) makes it the primary news source;
# without it news falls back to GDELT only (keyless, throttled to 1 req/6s)
FINNHUB_API_KEY=your_finnhub_key_here

# Optional — raises the BLS quota from 25 to 500 req/day
BLS_API_KEY=your_bls_key_here

# Production hardening (recommended before public deployment)
# Mutating endpoints (pipeline run, perf update, model train) require this
# token via the X-Admin-Token header. Unset = those endpoints return 503.
ADMIN_API_TOKEN=generate_a_long_random_string
# Error monitoring (free tier at sentry.io). Unset = Sentry disabled.
SENTRY_DSN=https://your_key@your_org.ingest.sentry.io/project_id

# Database (defaults to SQLite)
DATABASE_URL=sqlite:///./smart_money.db

# Scoring
CONVICTION_THRESHOLD=0.6

# Pipeline
PIPELINE_MODE=oneshot
SCHEDULE_INTERVAL_MINUTES=60
LOG_LEVEL=INFO
```

### Run the System

```bash
# 1. Initialize the database (creates all tables)
make init-db

# 2. Run the pipeline (ingest signals, enrich, run models, score)
make run-once

# 3. Start the API server (separate terminal)
make run-api

# 4. Start the dashboard (separate terminal)
cd dashboard && npm install && npm run dev
```

- Dashboard: `http://localhost:5173`
- API + docs: `http://localhost:8000/docs`

### Other Commands

```bash
# Run pipeline in daemon mode (repeats on interval)
make run-daemon

# Run a backtest from CLI
make backtest
# or with custom params:
.venv/bin/python -m src.backtesting.backtester --start 2025-01-01 --end 2026-04-01 --threshold 0.6

# Update signal performance tracking
.venv/bin/python -m src.tracking.performance

# Run tests
make test
```

---

## How It Works

### Pipeline Flow

```
Signal Ingestion        Enrichment           Analysis Models        Scoring & Alerts

SEC EDGAR (Form 4) ──┐                       Monte Carlo            Conviction Engine
Capitol Trades ──────┼─> Ingest ─> Enrich ─> Models ─> Ensemble ─> Score ─> Top Pick ─> Alert
SEC 13F (8 funds) ───┘    |          |                                |          |
                        Yahoo +   FRED + BLS                       Dashboard   Slack
                       SEC XBRL   + Treasury                       (React)    Webhook
```

**Step 1 — Ingest:** Fetches insider trades (SEC EDGAR Form 4 XML), congressional trades (Capitol Trades), and quarterly 13F holdings diffs from 8 curated institutional funds (Berkshire, Pershing Square, Scion, Bridgewater, RenTech, Tiger Global, Appaloosa, Third Point) — new positions and ≥25% adds are BUYs, full exits are SELLs. Deduplicates by (ticker, actor, trade_date, source_type).

**Step 2 — Enrich:** For each signal ticker, fetches price action via yfinance (momentum, RSI, drawdown, volume), then overlays **point-in-time SEC XBRL fundamentals** (EPS TTM with Q4 synthesis, EPS/revenue YoY growth, P/E recomputed from price ÷ XBRL TTM EPS) — filings-derived values that only use data filed on or before the evaluation date. Falls back to Tiingo if yfinance fails.

**Step 3 — Macro Context:** Fetches economic indicators from FRED (yield spread, unemployment claims, CPI, fed funds rate, VIX, consumer sentiment, M2, housing starts), plus BLS (payrolls YoY, unemployment rate) and Treasury FiscalData (avg marketable rate, total public debt). Classifies regime as expansion/transition/recession.

**Step 4 — Analysis Models (11 ensemble components per ticker):**

| Model | What It Does |
|---|---|
| **Monte Carlo** | 10,000-path simulation at 21d and 63d horizons. Probability of profit, expected return, VaR. |
| **HMM Regime** | Hidden Markov Model detecting bull/bear/sideways states with transition probabilities. |
| **GARCH** | Volatility forecasting at 5d and 20d. Detects vol expansion/contraction for timing. |
| **Fama-French** | 5-factor model (Mkt, SMB, HML, RMW, CMA). Measures alpha and systematic risk exposure. |
| **Event Study** | OLS market model with estimation window [-120,-10], event window [-5,+20]. Cumulative abnormal returns + t-test. |
| **Copula Tail Risk** | Student-t copula fitting. Tail dependence, VaR/CVaR at 95%/99%, conditional stress VaR. |
| **Bayesian Decay** | Exponential decay with Gamma prior. Posterior half-life, entry/exit windows, decay quality classification. |
| **Mean-Variance** | Markowitz optimization across all signal tickers. Max-Sharpe, min-variance, risk contribution. |
| **Options Flow** | Options chain analysis via yfinance. Put/call ratio, IV skew, unusual volume, max pain. |
| **News Sentiment** | Finnhub + GDELT headlines scored by FinBERT → VADER → lexicon fallback. Three features: `news_volume_z` (attention), `news_sentiment_mean`, `news_sentiment_trend_3d`. Bias-guarded: only articles published before evaluation time count, timestamps persisted. |
| **Weather Overlay** | Open-Meteo degree-day delta (6 US metros) + NWS severe-weather alerts. Only contributes for weather-sensitive sectors: Energy (fuel demand), Utilities (power demand), Insurance (claims risk), Agriculture (crop stress), Airlines (disruption). |

**Step 5 — Ensemble Scoring:** Weighted combination of up to 11 components into a single 0-100 score with recommendation (strong_buy / buy / hold / avoid). Requires 4+ models to independently agree (≥50) before issuing buy/sell — eliminates ~30% of spurious signals from the multiple-testing problem.

**Step 6 — Conviction Scoring:** Multi-factor model combining signal strength (40%), fundamental quality (35%), and macro regime modifier (25%) into a 0-1 conviction score. Threshold adapts to regime and volatility (0.45–0.80 range).

**Step 7 — Morning Top Pick:** Every pipeline run (including no-new-signal mornings) selects the single stock most probable for near-term outperformance — a blend of conviction (40%), ensemble score (40%), Monte Carlo 30-day P(profit) (20%), plus a multi-actor cluster bonus. Delivered via Slack, `/api/top-pick`, and the dashboard's Top Pick card.

**Step 8 — Alerts:** If Slack is configured, sends formatted alerts for signals passing the conviction threshold, ensemble scores > 70, and the daily top pick.

---

## Data Sources

All data sources are **free and require no paid subscriptions** (FRED key is free).

| Source | What | Rate Limits |
|---|---|---|
| **SEC EDGAR** | Insider trades (Form 4 filings) | Free, no key needed |
| **SEC EDGAR XBRL** | Point-in-time fundamentals from 10-K/10-Q company facts | Free, no key needed (10 req/s) |
| **SEC EDGAR 13F** | Institutional holdings of 8 curated funds | Free, no key needed |
| **Capitol Trades** | Congressional trade disclosures | Free, no key needed |
| **Yahoo Finance** (yfinance) | Price history, options chains, price-action enrichment | Free, unlimited |
| **FRED** | Macro indicators (yield curve, CPI, VIX, M2, etc.) | Free key, generous limits |
| **BLS** | Payrolls, unemployment rate | Free; 25/day (500 with free key) |
| **Treasury FiscalData** | Avg Treasury rates, public debt | Free, no key needed |
| **Finnhub** | Per-ticker company news (primary news source) | Free key, 60 req/min |
| **GDELT 2.0** | Global news breadth (keyless fallback) | Free; hard 1 req/5s limit (client throttles to 6s) |
| **Open-Meteo + NWS** | Weather forecast anomalies + severe-weather alerts | Free, no key needed |
| **Fama-French** (pandas-datareader) | 5-factor model data | Free, no key needed |
| **Tiingo** | Fallback enrichment + price history | Free key, 2000 req/month |

See [docs/DATA_SOURCES.md](docs/DATA_SOURCES.md) for the full catalog of additional free sources and integration notes.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Liveness + DB check for uptime monitors |
| `/api/top-pick` | GET | The single stock most probable for near-term outperformance (`lookback_days` param) |
| `/api/sectors` | GET | Industry/sector breakdown of recent activity — counts, avg conviction/ensemble, top ticker per sector |
| `/api/signals` | GET | All signals with filters: `days`, `source`, `min_conviction`, `sector` |
| `/api/signals/{ticker}` | GET | Signals for a specific ticker |
| `/api/news/{ticker}` | GET | Latest news-sentiment features + scored headlines for a ticker |
| `/api/macro/extra` | GET | BLS payrolls/unemployment + Treasury rates/debt snapshot |
| `/api/analysis/{ticker}` | GET | All model results for a ticker |
| `/api/analysis/{ticker}/options` | GET | Options flow analysis for a ticker |
| `/api/analysis/hmm/all` | GET | HMM regime states for all tickers |
| `/api/analysis/mean-variance` | GET | Portfolio optimization results |
| `/api/analysis/ensemble/all` | GET | Ensemble scores for all signals |
| `/api/analysis/event-study/summary` | GET | Aggregated event study statistics |
| `/api/macro` | GET | Latest macro regime snapshot |
| `/api/macro/extended` | GET | Extended macro (VIX, sentiment, M2, etc.) |
| `/api/macro/history` | GET | Historical macro data |
| `/api/prices/{ticker}` | GET | Historical price data for a ticker (`days` param: 7–730, default 365) |
| `/api/backtest` | POST | Run backtest with `start_date`, `end_date`, `conviction_threshold` |
| `/api/performance/summary` | GET | Signal performance tracking (win rates, returns) |
| `/api/performance/update` | POST | Trigger performance update for all signals 🔒 admin |
| `/api/export/signals` | GET | Export signals as CSV or JSON (`format` param) |
| `/api/export/analysis/{ticker}` | GET | Export ticker analysis as CSV or JSON |
| `/api/pipeline/run` | POST | Trigger full pipeline cycle 🔒 admin |
| `/api/dashboard` | GET | Combined signals + macro for dashboard |
| `/docs` | GET | Interactive Swagger API documentation |

---

## Dashboard

The React dashboard provides:

- **Morning Top Pick Card** — the day's highest-probability stock with composite score, conviction, ensemble, Monte Carlo P(profit), signal context, and runners-up.
- **Industry Panel** — pick one industry and see what happened there: signal counts, buy/sell split, avg conviction/ensemble, top ticker, member tickers. Selecting a sector filters the signal table and stat cards.
- **Signal Table** — sortable by conviction, date, ticker. Filterable by source and sector. CSV/JSON export.
- **Macro Gauge** — SVG regime indicator with yield spread, unemployment, CPI, fed funds.
- **Extended Macro** — VIX, consumer sentiment, M2, housing starts, industrial production.
- **Performance Panel** — Win rates, average returns by direction/source/conviction bucket, top winners/losers.
- **Backtest Engine** — Date range picker, conviction slider, equity curve, filtered vs unfiltered metrics comparison. Includes "Last 30 Days" quick-run button for instant recent-period analysis.
- **Stock Price History** — Interactive price chart for any analyzed ticker. Dropdown to select from all signal tickers, time range buttons (30D / 90D / 1Y / 2Y), SVG line chart color-coded green/red by period performance, with stats row (current price, change %, period high/low).
- **Mean-Variance Chart** — Efficient frontier, max-Sharpe/min-variance allocation bars, risk contribution.
- **Ticker Detail** (click any signal) — Full model breakdown:
  - Ensemble score banner with recommendation
  - Monte Carlo simulation grid
  - GARCH volatility forecast
  - HMM regime probabilities + transition matrix
  - Fama-French factor exposures
  - Copula tail risk gauge
  - Bayesian decay curve with half-life
  - Event study CAR timeline
  - Options flow (PCR, IV skew, unusual volume, max pain, volume/OI breakdown)

---

## Project Structure

```
financial-planner/
├── config/
│   ├── .env.example              # Environment template
│   └── settings.py               # Pydantic settings
├── src/
│   ├── clients/
│   │   ├── edgar.py              # SEC EDGAR Form 4 parser
│   │   ├── edgar_xbrl.py         # ★ SEC XBRL point-in-time fundamentals
│   │   ├── edgar_13f.py          # ★ 13F institutional holdings diffs (8 funds)
│   │   ├── congress.py           # Capitol Trades scraper
│   │   ├── news.py               # ★ Finnhub + GDELT news client (throttled)
│   │   ├── macro_extra.py        # ★ BLS + Treasury FiscalData
│   │   ├── yahoo.py              # yfinance: prices, enrichment, options
│   │   ├── tiingo.py             # Tiingo fallback client
│   │   ├── fred.py               # FRED macro indicators
│   │   ├── fama_french.py        # Fama-French factors (date-joined alignment)
│   │   └── options.py            # Options chain analysis
│   ├── analysis/
│   │   ├── monte_carlo.py           # Monte Carlo simulation
│   │   ├── hmm_regime.py            # Hidden Markov Model
│   │   ├── garch_forecast.py        # GARCH volatility forecasting
│   │   ├── event_study.py           # Event study (CAR + OLS)
│   │   ├── copula_tail_risk.py      # Copula tail dependence
│   │   ├── bayesian_decay.py        # Bayesian signal decay
│   │   ├── mean_variance.py         # Mean-Variance optimization
│   │   ├── ensemble_scoring.py      # Ensemble + BH-FDR + regime weights
│   │   ├── xgboost_classifier.py    # ML classifier (nested CV, feature selection)
│   │   ├── granger_causality.py     # ★ Granger causality test
│   │   ├── conformal_prediction.py  # ★ Distribution-free prediction intervals
│   │   ├── adversarial_validation.py# ★ Train/test distribution shift detection
│   │   ├── ic_monitor.py            # ★ Rolling IC/ICIR signal quality monitor
│   │   ├── correlation_filter.py    # ★ Cross-asset cluster dampening
│   │   ├── structural_breaks.py     # ★ CUSUM + variance ratio break detection
│   │   ├── news_sentiment.py        # ★ FinBERT/VADER sentiment features (bias-guarded)
│   │   ├── weather_overlay.py       # ★ Sector-aware weather impact scoring
│   │   └── top_pick.py              # ★ Morning top-pick selection
│   ├── scoring/
│   │   ├── conviction_engine.py  # Combined conviction + adaptive threshold
│   │   ├── signal_scorer.py      # 5-factor signal scoring
│   │   ├── fundamental_scorer.py # 5-factor fundamental scoring
│   │   └── macro_scorer.py       # Regime detection + modifier
│   ├── alerts/
│   │   └── notifier.py           # Slack webhook alerts
│   ├── tracking/
│   │   └── performance.py        # Signal performance tracker
│   ├── backtesting/
│   │   ├── backtester.py         # Historical backtester
│   │   └── validator.py          # D1-D6 validation framework
│   ├── pipeline/
│   │   └── orchestrator.py       # End-to-end pipeline
│   ├── models/
│   │   ├── database.py           # SQLAlchemy models (16 tables)
│   │   └── schemas.py            # Pydantic validation schemas
│   └── api/
│       └── api.py                # FastAPI endpoints (18 routes)
├── dashboard/
│   └── src/
│       ├── components/
│       │   ├── Dashboard.tsx          # Main layout
│       │   ├── TopPickCard.tsx         # Morning top-pick card
│       │   ├── SectorPanel.tsx         # Industry picker + sector stats
│       │   ├── StockPriceChart.tsx     # Historical price chart with ticker/range selector
│       │   ├── SignalTable.tsx         # Signal table with export
│       │   ├── MacroGauge.tsx          # SVG regime gauge
│       │   ├── ExtendedMacro.tsx       # Extended macro panel
│       │   ├── PerformancePanel.tsx    # Performance tracking
│       │   ├── BacktestPanel.tsx       # Backtest UI
│       │   ├── MeanVarianceChart.tsx   # Efficient frontier
│       │   ├── TickerDetail.tsx        # Per-ticker drill-down
│       │   ├── MonteCarloChart.tsx     # MC simulation grid
│       │   ├── VolatilityForecast.tsx  # GARCH forecast
│       │   ├── FactorExposure.tsx      # Fama-French factors
│       │   ├── EventStudyChart.tsx     # CAR timeline
│       │   ├── TailRiskGauge.tsx       # Copula tail risk
│       │   ├── BayesianDecayChart.tsx  # Decay curve
│       │   ├── EnsembleScore.tsx       # Model consensus
│       │   ├── OptionsFlow.tsx         # Options flow analysis
│       │   └── ConvictionBar.tsx       # Conviction visualization
│       ├── api/client.ts              # API client
│       └── types/index.ts             # TypeScript interfaces
├── Makefile
├── requirements.txt
└── README.md
```

---

## False Positive Reduction & Bias Controls

This system implements institutional-grade statistical safeguards to eliminate false positives:

### Multiple Testing Correction
- **Benjamini-Hochberg FDR** (`ensemble_scoring.py`) — applied across the full ticker universe to control the false discovery rate. Signals that fail FDR are downgraded to "hold".
- **Model agreement gate** — requires ≥4 of 9 models to independently score above neutral before issuing a buy/sell recommendation.

### Signal Validation
- **Granger Causality Test** (`granger_causality.py`) — validates that each signal source (insider/congressional) actually *predicts* returns rather than just riding momentum. Runs quarterly; non-predictive sources get weight multipliers applied.
- **Adversarial Validation** (`adversarial_validation.py`) — detects distribution shift between train/test sets. If a classifier can distinguish the two (AUC > 0.55), the OOS backtest is flagged as unreliable.
- **Conformal Prediction Intervals** (`conformal_prediction.py`) — per-signal prediction intervals with distribution-free coverage guarantees. Signals where the interval includes zero (no directional edge) are flagged.

### Adaptive Thresholding
- **Regime-adaptive conviction threshold** (`conviction_engine.py`) — raises the bar by +10pp in recession (fewer buys), relaxes by -5pp in expansion. Additional +5pp for extreme RSI readings.
- **Structural Break Detection** (`structural_breaks.py`) — CUSUM and variance ratio tests detect *when* regime changes occur. Recent breaks trigger threshold elevation and shorter signal half-lives.

### Calibration & Monitoring
- **Isotonic Regression Calibration** (`validator.py` D5) — maps raw ensemble scores to calibrated win-rate probabilities. Applied live so threshold decisions are based on accurate probability estimates.
- **Rolling IC/ICIR Monitor** (`ic_monitor.py`) — tracks Information Coefficient on rolling 6-month windows. ICIR < 0.5 triggers a degradation alert before alpha decay becomes a real loss.
- **Deflated Sharpe Ratio** (`validator.py` D6) — corrects Sharpe for number of strategies tried (Bailey & Lopez de Prado, 2014). DSR p-value > 0.05 means the Sharpe isn't distinguishable from lucky chance.

### Bias Reduction
- **Full model audit (2026-06)** — every analysis/scoring module reviewed for statistical bias; 10 genuine defects fixed, including Monte Carlo drift double-correction, Fama-French/copula/event-study date misalignment against lagged FF data, pre-event drift contaminating CARs, a no-op walk-forward optimizer, and Granger multiple-testing inflation. Findings and remaining strategy-level caveats: [docs/ML_MODEL_REVIEW.md](docs/ML_MODEL_REVIEW.md).
- **News lookahead guard** (`news_sentiment.py`) — every article carries its publication timestamp; scoring at time T silently drops anything published after T, and timestamps are persisted so backtests can prove no future news leaked in.
- **Point-in-time fundamentals** (`edgar_xbrl.py`) — only facts with `filed <= as_of` are used, so a backtest can never see a quarter before it was actually disclosed.
- **Cross-Asset Correlation Filter** (`correlation_filter.py`) — detects sector clustering (e.g. 5 congressional trades in biotech in one week). Applies `1/sqrt(n)` conviction dampening for correlated clusters.
- **Regime-Conditional Ensemble Weights** (`ensemble_scoring.py`) — maintains separate weight vectors per HMM state (bull/bear/sideways). Models that add noise in a given regime are automatically downweighted.
- **XGBoost Safeguards** (`xgboost_classifier.py`) — minimum 200 samples for XGBoost (L1 logistic regression fallback for <200), permutation importance feature selection with random noise benchmark, nested temporal CV. Refuses to issue predictions if CV AUC ≤ 0.55.

### Validation Framework (Tier D)
| Check | What It Does |
|---|---|
| **D1 — OOS Split** | 70/30 temporal split; flags `OVERFIT` if Sharpe decays > 0.5 |
| **D2 — Calibration** | Per-bucket expected vs actual win rate |
| **D3 — LOO Contribution** | Identifies noisy vs alpha-adding models |
| **D4 — Bootstrap CI** | 95% confidence intervals on Sharpe/win-rate |
| **D5 — Isotonic Calibration** | Live score → P(win) mapping |
| **D6 — Deflated Sharpe** | Multiple-testing correction at strategy level |

---

## Production Hardening

Built-in safeguards for public deployment:

- **Admin gate** — `POST /api/pipeline/run`, `/api/performance/update`, and `/api/ml/xgboost/train` require the `X-Admin-Token` header (constant-time compared against `ADMIN_API_TOKEN`). Locked by default: if no token is configured, they return 503. The dashboard's Run Pipeline button prompts for the token once and remembers it locally.
- **Rate limiting** (slowapi, per-IP) — 120/min global default, 20/min on `/api/prices/{ticker}` (each uncached hit is a live yfinance call), 5/min on backtests. Price history is served from a 15-minute in-memory TTL cache.
- **Compliance banner** — "research signals, not financial advice" rendered persistently on the dashboard (publisher-exemption posture), in addition to the per-pick disclaimer in Slack and the Top Pick card.
- **Monitoring** — set `SENTRY_DSN` for error tracking (free tier works); point UptimeRobot (or any monitor) at `GET /api/health`, which checks API liveness and database connectivity.
- **Backups** — `.github/workflows/backup.yml` runs a nightly `pg_dump` at 3 AM ET against the `DATABASE_URL` secret and stores 14 days of artifacts. No-ops gracefully while still in SQLite mode.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, SQLAlchemy, SQLite (PostgreSQL-ready) |
| **Analysis** | NumPy, SciPy, arch (GARCH), hmmlearn (HMM), XGBoost, scikit-learn, statsmodels, yfinance, vaderSentiment (FinBERT optional via transformers) |
| **Data** | httpx (async HTTP), fredapi, pandas, pandas-datareader |
| **Frontend** | React 18, TypeScript, Vite |
| **Alerts** | Slack webhooks via httpx |
| **Infrastructure** | structlog (logging), APScheduler (daemon mode), Pydantic (settings + validation) |

---

## Roadmap

- [x] BH-FDR multiple testing correction across ticker universe
- [x] Granger causality test for signal source validation
- [x] Conformal prediction intervals per signal
- [x] Adversarial validation for distribution shift detection
- [x] Rolling IC/ICIR monitoring with degradation alerts
- [x] Cross-asset correlation filter with `1/sqrt(n)` cluster dampening
- [x] Structural break detection (CUSUM + variance ratio)
- [x] Adaptive conviction threshold (regime + volatility aware)
- [x] Regime-conditional ensemble weights
- [x] Deflated Sharpe Ratio (D6) — multiple testing correction at strategy level
- [x] Isotonic regression calibration (D5) — live score → P(win) mapping
- [x] XGBoost overhaul: nested CV, permutation feature selection, sample-size guards
- [x] Stock price history visualization — per-ticker SVG chart with time range selector and performance stats
- [x] Full ML bias/error audit — 10 defects fixed ([docs/ML_MODEL_REVIEW.md](docs/ML_MODEL_REVIEW.md))
- [x] Morning Top Pick — daily single-stock selection via Slack/API/dashboard
- [x] Industry/sector categorization — `/api/sectors` + dashboard picker with click-to-filter
- [x] News sentiment (Finnhub + GDELT + FinBERT/VADER) as ensemble component + XGBoost features
- [x] SEC XBRL point-in-time fundamentals replacing yfinance `.info`
- [x] 13F institutional holdings as a third signal source
- [x] Weather overlay for weather-sensitive sectors (Open-Meteo + NWS)
- [x] BLS + Treasury macro enrichment
- [ ] Scheduled pipeline + historical backfill (cron + 6-12 month SEC EDGAR archives)
- [ ] Network/graph analysis of insider co-trading patterns
- [ ] NLP sentiment on 10-K/10-Q filings (Loughran-McDonald wordlists)
- [ ] Paper trading mode with P&L tracking
- [ ] Real-time streaming (WebSocket + SSE)

Launch and deployment plans: [docs/PUBLIC_LAUNCH_ROADMAP.md](docs/PUBLIC_LAUNCH_ROADMAP.md) · [docs/DEPLOYMENT_ROADMAP.md](docs/DEPLOYMENT_ROADMAP.md)

---

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past performance of insider or congressional trading signals does not guarantee future results. Always do your own due diligence before making investment decisions.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
