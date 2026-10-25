# AI Builders Waterloo — AI x Money (Thu Oct 29, 2026)

12 minutes of talk, about 5 minutes of Q&A. Live demo, no deck. You can show
one or two charts.

**Working title:** *I let an AI pipeline watch insiders' money. Here's why it
doesn't get to press "buy".*

**The one idea the room should leave with:** with money, an agent workflow is
only worth trusting as far as you can check it. Breadth beat cleverness, and
the bugs that mattered failed silently.

---

## Run of show

| Time | Segment | On screen |
|---|---|---|
| 0:00–1:00 | Hook | Live dashboard |
| 1:00–3:30 | The workflow, step by step | `src/pipeline/orchestrator.py` → `run_cycle` |
| 3:30–6:30 | Live demo | Dashboard → Slack alert → `make model-status` |
| 6:30–9:30 | What I tried first, and why it failed | Commit history, the 2023 chart |
| 9:30–11:30 | What I'd tell you before you point an agent at your money | Terminal |
| 11:30–12:00 | Close + one contact line | — |

---

## 0:00 — Hook (1 min)

- "Executives have to report their own stock buys within two business days.
  Members of Congress have to report theirs too. All of it is public. I built
  a pipeline that reads every one of those filings, scores it with 11 models,
  and pings me on Slack when one looks good."
- "Then I tested it honestly, and it lost to just buying the index."
- "This talk covers how the workflow is built, how I found out it was wrong,
  and the guardrails I'd copy into anything that touches your money."
- Bridge to the theme (the night is about managing money with agents): "This is
  the investing corner of personal finance. The lessons carry over to budgeting,
  reconciliation, anything where an agent makes a call about your money."

## 1:00 — The workflow (2.5 min)

Open `orchestrator.py` at `run_cycle` and scroll it top to bottom. The code
reads as a list of steps, so let it be the diagram.

1. **Ingest.** SEC EDGAR Form 4 insider trades, congressional disclosures,
   and 13F filings from 8 funds (diffed quarter over quarter). All free.
2. **Enrich.** Prices (yfinance), fundamentals from SEC XBRL using only
   *what had been filed by that date*, macro data (FRED, BLS, Treasury), news
   sentiment (Finnhub/GDELT + VADER), even a weather overlay for energy and
   utilities.
3. **Model.** Monte Carlo, hidden Markov regimes, GARCH volatility,
   Fama-French factors, copula tail risk, event study, Bayesian decay. These are
   blended into an 11-component ensemble, and a conviction score sits on top.
4. **Decide and notify.** It writes a top pick to Slack before the market opens.
   **A human executes the trade.** There is no broker integration. I planned
   Alpaca auto-execution at first and deliberately never built it.
5. **Schedule.** GitHub Actions runs it twice each weekday (7 AM and 5:30 PM ET).
   SQLite holds the data, FastAPI serves it, and a React dashboard shows it.

Key point to land: *the AI was in the building, not the runtime.* No LLM
makes a decision here. Claude Code co-wrote 20 of the 25 commits, and it also
did the audit that found the bugs. At runtime everything is a deterministic,
inspectable statistical model. That was a choice, made because this touches money.

## 3:30 — Live demo (3 min)

1. **Live dashboard** (`localhost:5174`, run against fresh data beforehand):
   recent signals, macro regime gauge, signal performance by source and
   conviction.
2. **Slack alert:** show the top-pick message that arrived that morning.
3. **`make model-status`** in the terminal takes about 2 seconds. It prints
   the **validation gates**, and **3 of 7 pass**. Say it out loud: "My own
   system tells me which of its claims I'm not allowed to make."
4. *(Optional, 20 sec)* Show one row's ensemble breakdown: which models voted
   and how much each contributed.

## 6:30 — What I tried first, and why it failed (3 min)

This is the part the organizers asked for, so spend the most time here.

**The result: the model lost to the index.**
- Backtest: every open-market insider buy disclosed in 2023 across the S&P
  SmallCap 600 (1,533 scanned, 1,409 scored), scored using only data
  available on each disclosure date.
- Top-15 picks: **+54%**. S&P 500: **+80%**. Nasdaq: **+112%**. Only 2 of 15
  beat the S&P.
- Correlation between conviction score and return: **about 0**.
- **An equal-weight basket of *all* insider buys: +99%.** It beat the S&P.
  → *The signal is in the breadth. My clever concentration step was the
  part that destroyed value.*
- The live record says the same (Mar 30 → Jun 11, 2026): all buys **+8.9%**,
  high-conviction picks **+1.5%**, S&P **+16.6%**, Nasdaq **+24.1%**.
- Show `equity_curve_2023.png` from the `validation-backtest` branch. This is
  the one chart worth showing.

**The bugs that mattered all failed silently.** Pick 2 or 3 of these:
- **Zero buys and no error.** EDGAR's `primaryDocument` link points to an
  XSL-rendered *viewer*, not the raw XML. The parser found nothing and was
  happy about it. Fix: strip the viewer prefix.
- **A backtest that cheated.** Backfilled events carried pre-launch disclosure
  dates, so the "track record" took credit for gains from before the system
  existed. Fix: anchor on the date the system actually *saw* the signal.
- **A track record that froze.** The performance tracker existed but the
  pipeline never called it. The numbers went stale for two months and nothing
  complained.
- **An audit found 10 statistical bugs**, for example factor data misaligned
  by date, and an event study that started measuring 5 days *before* the event.
  Each one made the model look better than it was.
- **This week:** a single bad price (NaN) in 1 of ~1,300 tickers crashed the
  *entire* run at the portfolio-optimizer step, so no scoring and no alerts.
  The scheduled run would simply have stopped producing anything. The fix:
  skip bad inputs and isolate each step so one model can't take down the
  whole cycle. *(Show the diff: it's about 10 lines.)*

## 9:30 — What to copy if you point an agent at your money (2 min)

1. **Keep the human on the irreversible step.** Agents can read, score, and
   draft. The money moves when you press the button.
2. **Write the acceptance gates before you believe the output.** Mine are
   checked in and run with one command. "3 of 7 passing" is a feature.
3. **Make silence an error.** Zero results, a stale timestamp, a skipped
   step: alert on all of them. Every bug that cost me was a quiet one.
4. **Use point-in-time data or don't backtest.** Use only what was knowable
   *on that date*, including when the filing became public.
5. **Compare against the dumbest baseline.** "Buy all of them" beat my
   11-model ensemble. Run the boring version first.
6. **Isolate failures per step.** One bad row shouldn't cost you a day's run.

## 11:30 — Close (30 sec)

- "The most useful thing the AI did for my money wasn't picking stocks. It
  was finding out, rigorously, that my stock picking didn't work."
- One line: name + website/LinkedIn. No pitch.

---

## Likely Q&A

- **"Why not let it trade automatically?"** Irreversible step, unvalidated
  model, and 3 of 7 gates still fail. Even if all of them passed, a human
  approval step costs one tap.
- **"Isn't following Congress illegal / insider trading?"** No. These are
  mandatory *public* disclosures. The catch is the lag: Congress can report
  weeks after the trade, which is why the system anchors on disclosure date.
- **"Which LLM runs it?"** None at runtime. I used Claude Code to build and
  audit it. I wanted every decision to be reproducible and explainable.
- **"What does it cost?"** Close to $0. Free data APIs, GitHub Actions, SQLite.
- **"Would you use it with real money?"** Only the breadth version, and only
  after it passes its own gates on forward data.
- **"How did you catch the bugs?"** A deliberate bias review, plus asking every
  number "what would make this look too good?" Agents are good at grinding
  through that kind of audit.
- **"Ensemble BUY calls average +8.5% over 20 days. Isn't that good?"** It's a
  76% win rate, but on only 34 signals over a few months. That's promising,
  not proven, and the gates exist for exactly this reason.

---

## Before the talk

- [ ] **Reply to Prabal to confirm, by Mon Oct 26.**
- [ ] Run the catch-up pipeline on the real DB **days ahead**: clearing the
      June→October backlog took ~8.5 hours (ensemble scoring of ~1,570 events
      took ~7.7 of them). After that, run it again the morning of the talk;
      a normal daily run is much shorter.
- [ ] Pull `reports/charts/equity_curve_2023.png` from `validation-backtest`.
- [ ] Rehearse with a timer. The failures section always runs long.
- [ ] Backup plan: the offline demo (`cd dashboard && npm run dev`, no API
      needed) plus screenshots of the live dashboard and the Slack alert.
- [ ] Test USB-C video out, with editor/terminal font size bumped for the room.
