# Data Sources to Add — Business, Weather, News

Current integrations: SEC EDGAR (Form 4), House/Senate Stock Watcher, yfinance,
Tiingo (fallback), FRED, Fama-French, options chains (yfinance).

Everything below is **free** (or has a usable free tier). Priority = impact on
the conviction/ensemble pipeline vs integration effort.

## 1. Business & fundamental data (free)

| Source | What | Link | Free tier | Priority |
|--------|------|------|-----------|----------|
| SEC EDGAR XBRL "company facts" API | Standardized fundamentals (revenue, EPS, margins) straight from 10-K/10-Q — replaces unreliable yfinance `.info` | https://data.sec.gov/api/xbrl/companyfacts/ | Unlimited, no key (10 req/s) | **High** — feeds `fundamental_scorer` with point-in-time data (kills lookahead) |
| SEC EDGAR 13F filings | Institutional holdings (hedge funds) — a second "smart money" source beyond insiders/Congress | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=13F | Free | **High** — new signal source for `SignalScorer` |
| SEC EDGAR 8-K filings | Material corporate events (M&A, CEO exits, guidance) in near-real-time | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=8-K / full-text: https://efts.sec.gov/LATEST/search-index?q= | Free | High — event triggers per ticker |
| Financial Modeling Prep | Fundamentals, ratios, earnings calendar, sector P/E (can replace the hard-coded sector medians) | https://site.financialmodelingprep.com/developer/docs | 250 req/day | Medium |
| Finnhub | Fundamentals, earnings calendar, insider sentiment aggregate | https://finnhub.io/docs/api | 60 req/min | Medium |
| Alpha Vantage | Fundamentals + 50+ technical indicators | https://www.alphavantage.co/documentation/ | 25 req/day | Low (tight quota) |
| US Treasury FiscalData | Auctions, debt, interest rates | https://fiscaldata.treasury.gov/api-documentation/ | Unlimited | Medium — macro layer |
| BLS API | Employment, wages, CPI detail | https://www.bls.gov/developers/ | 500 req/day | Medium — macro layer |
| BEA API | GDP components, industry-level output | https://apps.bea.gov/api/ | Free key | Low |
| US Census | Retail trade, business formation statistics | https://www.census.gov/data/developers/data-sets.html | Free key | Low |
| World Bank / IMF | Global macro for non-US exposure | https://data.worldbank.org/ · https://data.imf.org/ | Free | Low |

## 2. Weather data (free) — and how it maps to stocks

| Source | What | Link | Free tier |
|--------|------|------|-----------|
| NOAA / NWS API | US forecasts, alerts, severe weather | https://api.weather.gov (no key!) | Unlimited |
| Open-Meteo | Global forecast + 80-yr historical, dead-simple JSON | https://open-meteo.com/ | 10k req/day, no key |
| NOAA Climate Data Online | Historical climate (degree days) | https://www.ncdc.noaa.gov/cdo-web/webservices/v2 | Free token |
| EIA API | Energy demand/supply/storage (the weather→price transmission channel) | https://www.eia.gov/opendata/ | Free key |
| USDA NASS | Crop progress & condition reports | https://quickstats.nass.usda.gov/api/ | Free key |

**Trading channels (this is where weather actually moves stocks):**
- **Heating/Cooling Degree Days → natural gas & utilities** (UNG, utilities sector; EIA storage + NOAA HDD/CDD forecasts)
- **Hurricane season → insurers** (ALL, TRV, RE) and Gulf refiners (VLO, MPC)
- **Drought/frost → agriculture** (ADM, BG, DE, fertilizer names)
- **Unseasonable weather → retail** (apparel comps, HD/LOW storm demand)
- **Severe winter → airlines** (cancellation cost spikes)

Suggested integration: a `WeatherOverlay` analysis module that emits a 0-100
score only for tickers in weather-sensitive sectors (Energy, Utilities,
Insurance, Agriculture, Airlines) and is otherwise neutral — same pattern as
`EarningsOverlay`, plugged into the ensemble with a small weight.

## 3. News with stock impact (free tiers)

| Source | What | Link | Free tier | Priority |
|--------|------|------|-----------|----------|
| GDELT 2.0 | Global news firehose w/ tone scores, 15-min updates | https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/ (DOC API: https://api.gdeltproject.org/api/v2/doc/doc) | Free, unlimited | **High** — volume + tone per company name |
| Finnhub company news | Per-ticker news + sentiment | https://finnhub.io/docs/api/company-news | 60 req/min | **High** — easiest per-ticker mapping |
| Alpha Vantage News & Sentiment | Per-ticker news with relevance + sentiment scores | https://www.alphavantage.co/documentation/#news-sentiment | 25 req/day | Medium |
| Marketaux | Ticker-tagged business news | https://www.marketaux.com/ | 100 req/day | Medium |
| NewsAPI.org | General headlines (keyword search) | https://newsapi.org/ | 100 req/day (dev) | Low — no ticker tagging |
| SEC 8-K (again) | The *primary source* behind most market-moving "news" | see above | Free | High |
| Yahoo Finance RSS | Per-ticker headlines | https://feeds.finance.yahoo.com/rss/2.0/headline?s=TICKER | Free | Medium |
| Reddit API (r/wallstreetbets, r/stocks) | Retail sentiment/attention spikes | https://www.reddit.com/dev/api/ | 100 req/min OAuth | Low-Medium |

**Suggested integration:** `NewsSentimentAnalyzer` module:
1. Pull per-ticker headlines (Finnhub primary, GDELT for breadth).
2. Score sentiment with FinBERT (free, local — `ProsusAI/finbert`) or VADER as
   a cheap fallback.
3. Aggregate into 3 features per ticker/day: `news_volume_z` (attention),
   `news_sentiment_mean`, `news_sentiment_trend_3d`.
4. Add as a 10th ensemble component + 3 new XGBoost features.
5. Important bias guard: store article timestamps and only use news published
   **before** signal evaluation time in any backtest.

## Recommended build order — ✅ ALL BUILT (2026-06-10)

1. ✅ SEC XBRL fundamentals — `src/clients/edgar_xbrl.py`. Point-in-time
   (`filed <= as_of`) EPS TTM (with Q4 = FY − Q1..Q3 synthesis), EPS/revenue
   YoY growth; overlaid onto enrichment in the orchestrator, P/E recomputed
   from price ÷ XBRL TTM EPS.
2. ✅ Finnhub + GDELT news sentiment — `src/clients/news.py` +
   `src/analysis/news_sentiment.py`. FinBERT → VADER → lexicon fallback chain
   (VADER active in CI; install `transformers` for FinBERT). The 3 features
   feed the ensemble (10th component, weight 0.085) and XGBoost. Bias guard:
   `analyze(as_of=...)` drops future articles; timestamps persisted in
   `news_sentiment_results`. GDELT throttled to 1 req/6 s (hard API limit).
3. ✅ 13F institutional holdings — `src/clients/edgar_13f.py`. 8 curated
   funds (CIKs verified), two-filing diff → new/added positions = BUY, exits
   = SELL, `SourceType.INSTITUTIONAL`. trade_date = quarter end,
   disclosure_date = filing date (45-day lag is explicit — anchor any
   "followable alpha" claim on disclosure_date).
4. ✅ WeatherOverlay — `src/analysis/weather_overlay.py`. Open-Meteo 6-metro
   degree-day delta + NWS severe-alert count; scores Energy / Utilities /
   Insurance / Agriculture / Airlines channels only (11th ensemble component,
   weight 0.045, max ±15 pts).
5. ✅ BLS/Treasury macro — `src/clients/macro_extra.py`. Payrolls YoY,
   unemployment rate, avg marketable Treasury rate, total public debt;
   persisted to `macro_extra_data`, exposed at `/api/macro/extra`.

New API endpoints: `/api/news/{ticker}`, `/api/macro/extra`.
New env keys (optional): `FINNHUB_API_KEY` (news primary), `BLS_API_KEY`
(quota 25→500/day).
