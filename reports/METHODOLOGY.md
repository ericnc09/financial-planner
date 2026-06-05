# Smart Money Follows — Performance Validation Methodology

*How we test whether the system's decisions actually worked — and why the test is honest.*

This document explains the rigour behind the performance numbers. It is written
to be read by a sceptic. Where the honest answer is unflattering, we say so.

---

## What this is, and what it is not

The validation has **two independent pillars**:

| Pillar | Period | Nature | What it proves |
|--------|--------|--------|----------------|
| **1. Point-in-time backtest** | Signals disclosed in **2023** | *Reconstructed* — the system did not exist in 2023, so we re-ran its exact scoring rules on data that was public at the time | Whether the methodology has an edge over a multi-year, fully-resolved window |
| **2. Live track record** | **Mar–Jun 2026** | *Real* — the actual signals the running system scored and stored | What the system has actually done since launch (short, window still open) |

The backtest is explicitly a **simulation**. The system was built in March 2026,
so there are no genuine "2023 recommendations." Rather than invent a list of
winners after the fact (which would be survivorship- and hindsight-biased), we
**re-derive what the system's rules would have flagged in 2023 using only
information available on each decision date.** That distinction is the whole
point of the exercise.

> **This is not investment advice and not a solicitation. Backtested /
> hypothetical results have inherent limitations and do not represent actual
> trading. Past performance — simulated or real — does not guarantee future
> results.**

---

## Pillar 1 — the 2023 point-in-time backtest

### 1. The signal universe (no cherry-picking)

- **Source:** SEC EDGAR Form 4 filings — the authoritative public record of
  insider transactions. Free, complete, and impossible to quietly curate.
- **Universe:** the **S&P SmallCap 600** constituents — a recognised index,
  fixed *before* looking at any outcome. Small-caps are deliberately chosen:
  the academic literature (e.g. Lakonishok & Lee, 2001) finds insider open-market
  buying carries its strongest abnormal-return signal in smaller names, and
  large-cap insiders almost never buy on the open market (they receive grants
  and sell).
- **What counts as a "buy":** only **transaction code "P" — a cash, open-market
  purchase.** Option exercises (M), grants/awards (A) and tax-withholding
  dispositions (F) are *compensation, not conviction*, and are excluded. (The
  live system's `BUY_CODES` set is broader; we tightened it here precisely
  because a grant is not a smart-money bet.)
- **No selection on outcome:** we take **every** code-P purchase across the
  **whole** universe for the **whole** year, then let the scoring engine rank
  them. We never hand-pick tickers.

### 2. Point-in-time discipline (no look-ahead, no leakage)

Every input is frozen as of the **disclosure date** — the day the filing became
public and the strategy could first act:

- **Entry** is the **next trading day after disclosure**, never the trade date
  (insiders' trades are disclosed days later; entering on the trade date would
  be impossible in real life and would leak future information).
- **Price-derived features** (momentum, RSI, 52-week drawdown, liquidity) are
  computed from prices **up to and including the disclosure date only.**
- **Cluster / consensus signals** count only **other filings already public**
  by that date — a later cluster member cannot inform an earlier decision.
- **Macro regime** is reconstructed from **FRED as of the disclosure date**
  (e.g. the inverted 10y–2y curve that actually prevailed in 2023).
- **Valuation (P/E) and short-interest are deliberately neutralised.** There is
  no free point-in-time fundamentals source here, and using *today's* P/E to
  score a 2023 decision is textbook look-ahead leakage. Rather than fake it, we
  set those sub-scores neutral and disclose it. ~68% of the fundamental layer
  (the price-derived part) is genuine and point-in-time; the rest is held at
  0.5.

### 3. The scoring engine is the real one

We do **not** re-implement a flattering scorer. The backtest instantiates the
production `ConvictionEngine`, `SignalScorer`, `FundamentalScorer` and
`MacroScorer` and feeds them the point-in-time inputs above. The number you see
is the number the live system would have produced on that date.

The top **15 distinct tickers by conviction** become the portfolio.

### 4. Realistic execution

- **Transaction costs** are deducted per position using the system's own
  liquidity-tier model (10–60 bps round-trip depending on market cap / volume).
- **Delisted names are exited at their last traded price — never silently
  dropped.** This removes survivorship bias (the single most common way
  backtests lie) without the opposite error of assuming every delisting is a
  bankruptcy: in the S&P 600, most are *acquisitions* that pay out at a premium,
  so a blanket −100% would understate returns. A genuine zero print is still
  booked as a total loss.
- Returns are reported at **1, 3, 6 and 12 months and to-date**, plus an
  equal-weight **equity curve**.

### 5. Honest benchmarking

Each position is compared to **SPY (S&P 500)** and **QQQ (Nasdaq-100)** over the
**identical** holding window. The portfolio equity curve uses **matched cash
flows**: the same dollar enters the benchmark on the same day the pick is
entered — not a single lump sum that would flatter or penalise the timing.

---

## Pillar 2 — the real 2026 track record

These are the actual buy signals stored in `smart_money.db` that **cleared the
live conviction threshold (≥ 0.60)** between March and May 2026 — i.e. the ones
the system would have acted on. We measure their realized return against
SPY/QQQ over the identical window, with the same cost and benchmark conventions.

**Data hygiene:** records disclosed *before* the trade occurred (impossible —
data errors) and anything outside the system's real operating window are dropped
and counted, not silently included.

This record is **short and the window is still open**, so it is shown as raw
evidence, not a finished result.

---

## Known limitations (read these)

1. **One historical regime.** The 2023 entries resolve across **2023–2026, an
   exceptionally strong, mega-cap-led bull market.** A small-cap strategy faces a
   very high bar here (the S&P 500 and especially the Nasdaq-100 rose sharply).
   One regime is not proof of a durable edge.
2. **Universe restriction.** Results are conditional on the S&P 600. A different
   universe (micro-caps, the full market) could differ materially.
3. **Insider-only.** The congressional-trading leg — the project's marquee
   thesis — is **excluded from the backtest** because every free historical
   source for it was blocked or paywalled in this environment. The backtest
   therefore tests only the insider sub-strategy.
4. **Fundamentals not fully point-in-time** (see §2). Valuation and short-interest
   are neutralised rather than reconstructed.
5. **FRED data vintage.** Macro values are point-in-time by *observation date* but
   use latest revisions, not the original release vintage (ALFRED). The effect on
   ranking is negligible (macro is a near-uniform scaler within a year).
6. **Small sample.** 15 picks is a portfolio, not a statistically conclusive
   sample. Treat the result as indicative, not definitive.

---

## Reproduce it

```bash
# 1. collect every 2023 open-market insider buy in the S&P 600 (cached, resumable)
python -m scripts.validation.collect_insider_2023

# 2. point-in-time backtest -> reports/backtest_2023_results.json
python -m scripts.validation.run_backtest_2023

# 3. real since-launch record -> reports/track_record_2026_results.json
python -m scripts.validation.run_track_record_2026

# 4. charts -> reports/charts/*.png   ;   summary -> reports/VALIDATION_SUMMARY.md
python -m scripts.validation.make_charts
python -m scripts.validation.summarize
```

All raw inputs (EDGAR filings, prices) are cached under `reports/cache/` so the
results are deterministic and independently checkable.
