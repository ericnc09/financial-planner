# Smart Money Follows

**Track what insiders and Congress are buying before the market catches on.**

Smart Money Follows is an automated trading signal platform that ingests insider and congressional trade disclosures from public filings, enriches them with fundamentals and macroeconomic context, and scores each signal for conviction — giving you a ranked, actionable feed of the highest-quality trades to inform your own investment decisions.

---

## The Problem

Every day, corporate insiders and members of Congress disclose securities trades — often days or weeks before the market prices in the same thesis. This data is public, but it's scattered across SEC filings and government disclosures, buried in XML and HTML, and impossible to act on without hours of manual research.

By the time retail investors notice, the edge is gone.

## The Solution

Smart Money Follows closes that gap. The system:

1. **Ingests** insider trades (SEC EDGAR Form 4) and congressional trades (Capitol Trades) automatically
2. **Enriches** each signal with real-time fundamentals (P/E, revenue growth, momentum, RSI) and macro regime data (yield curve, unemployment, CPI, fed funds rate)
3. **Scores** every trade on a 0-100% conviction scale using a weighted model across signal strength, fundamental quality, and macroeconomic conditions
4. **Surfaces** the highest-conviction opportunities through a real-time dashboard with filtering, sorting, and visual breakdowns

You decide. You execute. The system gives you the edge.

---

## Key Features

| Feature | Description |
|---|---|
| **Dual Signal Sources** | SEC EDGAR Form 4 filings (insider trades) + Capitol Trades (congressional trades) — zero cost, no paid APIs |
| **Conviction Scoring** | Multi-factor model combining signal strength (actor reputation, trade size, clustering, timing) with fundamental quality and macro regime |
| **Macro Regime Detection** | Real-time classification of market conditions (Expansion / Transition / Recession) using FRED economic indicators |
| **Direction-Aware Analysis** | Scoring adapts based on buy vs. sell — a buy during a drawdown scores differently than a sell at all-time highs |
| **Live Dashboard** | React-based UI with signal table, macro gauge, conviction breakdowns, and source filtering |
| **Pipeline Automation** | One-click or scheduled pipeline runs: ingest, enrich, score — all in one pass |
| **Backtesting Engine** | Historical performance analysis with Sharpe/Sortino ratios, comparing filtered vs. unfiltered signal sets |

---

## Architecture

```
Signal Ingestion          Enrichment & Scoring              Presentation

SEC EDGAR ──┐             Tiingo Fundamentals               React Dashboard
             ├──> Ingest ──> Enrich ──> Score ──> Store ──> FastAPI Backend
Capitol     ─┘             FRED Macro Data                  REST API
Trades
```

**Four-layer pipeline:**

| Layer | What it does | Data source |
|---|---|---|
| **Ingestion** | Fetches and normalizes trade disclosures into a unified event schema | SEC EDGAR RSS + XML, Capitol Trades RSC |
| **Enrichment** | Attaches fundamentals (P/E, momentum, RSI, drawdown) and sector data to each signal | Tiingo API |
| **Macro Context** | Classifies the current economic regime and computes a macro modifier (0.5x - 1.5x) | FRED API |
| **Conviction Scoring** | Produces a final 0-1 conviction score per signal, gates on configurable threshold | Internal scoring engine |

---

## Scoring Model

Each signal is scored across three dimensions:

### Signal Score (40% weight)
- **Actor Reputation** (25%) — Congressional committee members and C-suite insiders score higher
- **Trade Size** (20%) — Larger trades relative to typical ranges indicate stronger conviction
- **Cluster Signal** (25%) — Multiple insiders trading the same ticker compounds the signal
- **Disclosure Timing** (15%) — Trades disclosed quickly after execution are more informative
- **Consensus** (15%) — Alignment between insider and congressional activity on the same name

### Fundamental Score (35% weight)
- **Valuation** (25%) — P/E relative to sector, direction-aware (low P/E favors buys)
- **Momentum** (20%) — 30-day and 90-day price momentum
- **Volatility Regime** (15%) — RSI-based overbought/oversold detection
- **Drawdown Opportunity** (20%) — Distance from 52-week high (buying the dip scores well)
- **Liquidity** (20%) — Average volume ensures tradability

### Macro Modifier (25% weight)
- **Expansion** (regime score < 0.4) — 1.0x to 1.5x multiplier, favoring risk-on
- **Transition** (0.4 - 0.7) — 0.8x to 1.0x, neutral to cautious
- **Recession** (> 0.7) — 0.5x to 0.8x, risk-off dampening

**Final formula:**
```
conviction = (signal_score x W_sig + fundamental_score x W_fund) / (W_sig + W_fund) x macro_modifier x direction_boost
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the dashboard)
- Free API keys: [Tiingo](https://www.tiingo.com/) and [FRED](https://fred.stlouisfed.org/docs/api/api_key.html)

### Installation

```bash
# Clone the repo
git clone https://github.com/ericnc09/financial-planner.git
cd financial-planner

# Install Python dependencies
pip install -r requirements.txt

# Set up environment
cp config/.env.example config/.env
# Edit config/.env and add your Tiingo and FRED API keys

# Initialize the database
make init-db

# Install dashboard dependencies
cd dashboard && npm install && cd ..
```

### Running the System

```bash
# 1. Run the pipeline (fetches signals, enriches, scores)
make run-once

# 2. Start the API server
make run-api

# 3. Start the dashboard (separate terminal)
make run-dashboard
```

The dashboard will be available at `http://localhost:5173` and the API at `http://localhost:8000`.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/signals` | GET | All signals with optional filters (`days`, `source`, `min_conviction`) |
| `/api/signals/{ticker}` | GET | Signals for a specific ticker |
| `/api/macro` | GET | Latest macro regime snapshot |
| `/api/macro/history` | GET | Historical macro data (`days` parameter) |
| `/api/dashboard` | GET | Combined signals + macro for the dashboard |
| `/api/pipeline/run` | POST | Trigger a pipeline cycle |
| `/docs` | GET | Interactive API documentation (Swagger) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, SQLAlchemy, SQLite (PostgreSQL-ready) |
| **Data Clients** | httpx (async), fredapi, tenacity (retry logic) |
| **Scoring** | NumPy, custom weighted multi-factor model |
| **Frontend** | React 18, TypeScript, Vite, Recharts |
| **Infrastructure** | structlog (structured logging), APScheduler (daemon mode) |

---

## Data Sources

All data sources are **free and require no paid subscriptions**.

| Source | What | How |
|---|---|---|
| **SEC EDGAR** | Insider trades (Form 4) | RSS feed + XML parsing of ownership documents |
| **Capitol Trades** | Congressional trades | HTML scraping of Next.js RSC payloads |
| **Tiingo** | Fundamentals, price history, RSI | REST API (free tier) |
| **FRED** | Macro indicators (yield curve, unemployment, CPI, fed funds) | REST API (free tier) |

---

## Project Structure

```
financial-planner/
├── config/
│   ├── .env.example        # Environment template (safe to commit)
│   └── settings.py         # Pydantic settings with validation
├── src/
│   ├── clients/
│   │   ├── edgar.py        # SEC EDGAR Form 4 parser
│   │   ├── congress.py     # Capitol Trades scraper
│   │   ├── tiingo.py       # Tiingo fundamentals + price client
│   │   └── fred.py         # FRED macro indicators client
│   ├── scoring/
│   │   ├── signal_scorer.py       # 5-factor signal scoring
│   │   ├── fundamental_scorer.py  # 5-factor fundamental scoring
│   │   ├── macro_scorer.py        # Regime detection + modifier
│   │   └── conviction_engine.py   # Combined conviction pipeline
│   ├── models/
│   │   ├── database.py     # SQLAlchemy models + engine setup
│   │   └── schemas.py      # Pydantic validation schemas
│   ├── pipeline/
│   │   └── orchestrator.py # End-to-end pipeline orchestration
│   ├── backtesting/
│   │   └── backtester.py   # Historical performance analysis
│   └── api/
│       └── api.py          # FastAPI endpoints
├── dashboard/
│   └── src/
│       ├── App.tsx                    # Main application
│       ├── components/
│       │   ├── Dashboard.tsx          # Layout + stat cards
│       │   ├── SignalTable.tsx         # Sortable, filterable signal table
│       │   ├── MacroGauge.tsx         # SVG regime gauge
│       │   ├── ConvictionBar.tsx      # Stacked score breakdown
│       │   └── RadarBreakdown.tsx     # Radar chart visualization
│       ├── api/client.ts             # API client wrapper
│       └── types/index.ts            # TypeScript interfaces
├── tests/
├── Makefile
├── requirements.txt
└── README.md
```

---

## Roadmap

- [ ] Alerting — push notifications (email/Slack) when high-conviction signals fire
- [ ] Historical conviction tracking — monitor how signals performed post-detection
- [ ] Sector heatmap — visual clustering of smart money flow by sector
- [ ] Additional signal sources — 13F filings, dark pool activity
- [ ] Portfolio simulation — paper trading mode with P&L tracking

---

## Disclaimer

This project is for **educational and research purposes only**. It does not constitute financial advice. Past performance of insider or congressional trading signals does not guarantee future results. Always do your own due diligence before making investment decisions.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
