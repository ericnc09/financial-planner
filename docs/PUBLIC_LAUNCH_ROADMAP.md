# Public Launch & User Engagement Roadmap

## Where you are today (honest baseline)

- **Built:** 4-layer pipeline (EDGAR/congressional ingest → enrichment → 9-model
  ensemble → conviction), FastAPI API, React dashboard, GitHub Actions twice-daily
  runs, Slack alerts, morning top pick, sector browsing, CSV/JSON export.
- **Validated:** the 2023 EDGAR backtest showed the *breadth* of insider buys beat
  the market (+99% equal-weight) while concentrated model picks did not (+54%).
  Conviction↔return correlation ≈ 0 at that time; the bias fixes in
  `docs/ML_MODEL_REVIEW.md` change the models, so the track record restarts now.
- **Missing for public use:** hosting, auth, a real track record under the fixed
  models, legal disclaimers, onboarding.

The single most valuable asset you can build before launch is a **timestamped,
tamper-evident track record**. Every day the pipeline already picks stocks —
start recording picks publicly *now* so that at launch you have months of
verifiable history (good or bad — the transparency IS the product, per the
rigour-first positioning already chosen for ericcosta.ca).

## Phase 0 — Pre-launch hardening (2–3 weeks)

1. Deploy privately (see `DEPLOYMENT_ROADMAP.md`) so the system runs unattended.
2. Start the public track record: every morning top pick written to an
   append-only `picks` table + auto-published JSON; show live win-rate on site.
3. Compliance basics (you are a *publisher*, not an adviser):
   - "Not investment advice; for research/education" disclaimer on every page,
     email, and Slack/Discord message (the top-pick card already carries one).
   - No personalized recommendations, no "buy this for *you*" language.
   - Performance shown must include losers; never cherry-pick.
   - Terms of service + privacy policy (Termly/GetTerms templates).
4. Name + domain (Smart Money Follows → smartmoneyfollows.com or a subdomain of
   ericcosta.ca to start).

## Phase 1 — Private beta (4–6 weeks)

- **Audience:** 20–50 users from r/algotrading, X/Twitter fintwit, quant Discord
  servers, friends. Gate with a waitlist (builds the launch email list).
- **Auth:** magic-link email (Supabase Auth / Clerk free tier).
- **The hook:** "Every morning at 7am: the one stock insiders and Congress are
  buying — with the full model breakdown and our live track record."
- **Feedback loop:** weekly email to beta users + a #feedback Discord channel.
- Instrument everything (PostHog free tier): activation %, D7 retention, which
  panels get used.

## Phase 2 — Public launch (week 8–10)

- **Launch surfaces:** Product Hunt, Hacker News Show HN, r/algotrading +
  r/stocks (mind self-promo rules — lead with the open methodology), fintwit
  thread showing a real win AND a real loss (credibility).
- **Free tier:** morning top pick (delayed to market open), sector dashboard,
  weekly recap email.
- **SEO engine (compounding):** every daily pick auto-generates a public page
  ("Why the model picked TPC on 2026-06-10") from data you already store —
  hundreds of indexed, data-rich pages within months. Same for sector pages
  ("Insider buying in Healthcare this month").

## Phase 3 — Engagement loops (ongoing)

| Loop | Mechanism | Why it retains |
|------|-----------|----------------|
| Daily ritual | 7am top-pick email/push — same time every day | Habit formation; the product IS a morning ritual |
| Accountability | Public scoreboard of every past pick (win rate, vs SPY) | Trust + recurring "how did we do" visits |
| Watchlists | Follow tickers → alert when smart money touches them | Personal stake; n alerts/week per user |
| Sector subscriptions | Subscribe to industries (the new sector panel) → weekly sector digest | Matches user identity ("I'm a biotech person") |
| Weekly recap | "This week: 3 picks, 2 winners, Congress sold airlines" | Shareable; drives social loop |
| Streaks/digest gamification | "Model is 4/5 this month" badges in email | Curiosity gap → opens |
| Community | Discord with a #daily-pick-discussion auto-post | Users retain each other |
| Referral | "Give a friend 1 month of premium" | Cheap acquisition |

**North-star metric:** weekly returning users who view the morning pick.
Guardrail: never optimize engagement by overstating performance.

## Phase 4 — Monetization (only after retention proves out)

- **Premium ($10–20/mo):** real-time alerts (vs delayed), full ensemble
  breakdown, watchlist alerts, API access, CSV exports, all-history backtests.
- **Free stays genuinely useful** — the daily pick stays free; it's the
  acquisition engine.
- Later: B2B API for the signal feed.

## Risks to manage

1. **Performance risk:** the model may underperform publicly. Mitigation: the
   transparency positioning — you sell *the data and rigour*, not a promise of
   returns. Show the insider-breadth basket alongside concentrated picks.
2. **Regulatory drift:** if you ever add per-user portfolios/auto-trading,
   the publisher exemption stops covering you — that's an RIA conversation.
3. **Data ToS:** yfinance scraping is fine for research, gray for a commercial
   product — budget for a licensed quote feed (Polygon $29/mo) at monetization.
