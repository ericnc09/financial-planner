"""
News Sentiment — turns per-ticker headlines into three model features and a
0-100 ensemble component score.

Features (per ticker, at evaluation time):
  - news_volume_z         : attention — last-3-day daily article count vs the
                            trailing window baseline, in standard deviations
  - news_sentiment_mean   : mean headline sentiment over the window [-1, 1]
  - news_sentiment_trend_3d: mean(last 3d) − mean(rest of window)

Sentiment backends, best available first:
  1. FinBERT (ProsusAI/finbert via transformers) — finance-tuned, local, free
  2. VADER (vaderSentiment) — cheap lexicon+rules fallback
  3. Built-in mini financial lexicon — zero-dependency last resort

BIAS GUARD: analyze() takes `as_of` and silently drops any article with
published_at > as_of. Article timestamps are preserved in the output so a
backtest can prove no future news leaked into a historical score.
"""

import re
from datetime import datetime, timedelta

import numpy as np
import structlog

logger = structlog.get_logger()

# Minimal Loughran-McDonald-flavoured fallback lexicon (last resort only)
_POS_WORDS = {
    "beat", "beats", "upgrade", "upgraded", "surge", "surges", "rally",
    "rallies", "record", "growth", "profit", "profitable", "strong",
    "outperform", "buy", "bullish", "gain", "gains", "jump", "jumps",
    "soar", "soars", "raise", "raises", "raised", "exceed", "exceeds",
    "win", "wins", "approval", "approved", "breakthrough", "expansion",
}
_NEG_WORDS = {
    "miss", "misses", "downgrade", "downgraded", "plunge", "plunges",
    "fall", "falls", "drop", "drops", "loss", "losses", "weak", "lawsuit",
    "probe", "investigation", "recall", "bankruptcy", "default", "fraud",
    "underperform", "sell", "bearish", "cut", "cuts", "layoff", "layoffs",
    "warning", "warns", "decline", "declines", "crash", "halt", "halted",
}


class SentimentScorer:
    """Scores text sentiment in [-1, 1] using the best available backend."""

    def __init__(self, prefer: str | None = None):
        self.backend = None
        self.model_name = "lexicon"
        if prefer != "lexicon":
            self._init_backend(prefer)

    def _init_backend(self, prefer: str | None):
        if prefer in (None, "finbert"):
            try:
                from transformers import pipeline
                self.backend = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    truncation=True,
                )
                self.model_name = "finbert"
                logger.info("sentiment.backend", model="finbert")
                return
            except Exception:
                pass
        if prefer in (None, "vader", "finbert"):
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self.backend = SentimentIntensityAnalyzer()
                self.model_name = "vader"
                logger.info("sentiment.backend", model="vader")
                return
            except Exception:
                pass
        logger.info("sentiment.backend", model="lexicon")

    def score_batch(self, texts: list[str]) -> list[float]:
        if not texts:
            return []
        if self.model_name == "finbert":
            outputs = self.backend(texts, batch_size=16)
            scores = []
            for out in outputs:
                label = out["label"].lower()
                s = float(out["score"])
                if label == "positive":
                    scores.append(s)
                elif label == "negative":
                    scores.append(-s)
                else:
                    scores.append(0.0)
            return scores
        if self.model_name == "vader":
            return [float(self.backend.polarity_scores(t)["compound"]) for t in texts]
        return [self._lexicon_score(t) for t in texts]

    @staticmethod
    def _lexicon_score(text: str) -> float:
        words = re.findall(r"[a-z']+", text.lower())
        if not words:
            return 0.0
        pos = sum(1 for w in words if w in _POS_WORDS)
        neg = sum(1 for w in words if w in _NEG_WORDS)
        if pos == neg:
            return 0.0
        return float(np.clip((pos - neg) / np.sqrt(len(words)), -1.0, 1.0))


class NewsSentimentAnalyzer:
    """Aggregates scored articles into the three ticker/day features."""

    MIN_ARTICLES = 3  # below this, the component stays neutral

    def __init__(self, scorer: SentimentScorer | None = None):
        self.scorer = scorer or SentimentScorer()

    def analyze(
        self,
        ticker: str,
        articles: list[dict],
        as_of: datetime | None = None,
        window_days: int = 14,
    ) -> dict | None:
        """
        Args:
            ticker: Stock ticker.
            articles: Normalized article dicts (see NewsClient).
            as_of: Evaluation cutoff. Articles published AFTER this moment
                   are dropped (bias guard). Defaults to now (live mode).
            window_days: Trailing window for the volume baseline.

        Returns:
            Feature dict, or None when no usable articles exist.
        """
        as_of = as_of or datetime.utcnow()
        window_start = as_of - timedelta(days=window_days)

        usable = [
            a for a in articles
            if a.get("published_at") is not None
            and window_start <= a["published_at"] <= as_of
        ]
        n_dropped_future = sum(
            1 for a in articles
            if a.get("published_at") is not None and a["published_at"] > as_of
        )
        if n_dropped_future:
            logger.debug(
                "news_sentiment.bias_guard_dropped",
                ticker=ticker, n_future=n_dropped_future, as_of=as_of.isoformat(),
            )
        if not usable:
            return None

        sentiments = self.scorer.score_batch([a["headline"] for a in usable])
        for a, s in zip(usable, sentiments):
            a["sentiment"] = round(s, 4)

        # Daily article counts across the full window (including zero days)
        counts: dict[str, int] = {}
        for d in range(window_days):
            day = (window_start + timedelta(days=d + 1)).date().isoformat()
            counts[day] = 0
        for a in usable:
            day = a["published_at"].date().isoformat()
            if day in counts:
                counts[day] += 1

        cutoff_3d = as_of - timedelta(days=3)
        recent = [a for a in usable if a["published_at"] > cutoff_3d]
        older = [a for a in usable if a["published_at"] <= cutoff_3d]

        daily = np.array(list(counts.values()), dtype=float)
        baseline = daily[:-3] if len(daily) > 3 else daily
        recent_daily_avg = len(recent) / 3.0
        base_mean = float(np.mean(baseline)) if len(baseline) else 0.0
        base_std = float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0
        volume_z = (
            (recent_daily_avg - base_mean) / base_std if base_std > 1e-9 else 0.0
        )
        volume_z = float(np.clip(volume_z, -5.0, 5.0))

        sentiment_mean = float(np.mean(sentiments))
        recent_mean = float(np.mean([a["sentiment"] for a in recent])) if recent else None
        older_mean = float(np.mean([a["sentiment"] for a in older])) if older else None
        if recent_mean is not None and older_mean is not None:
            trend_3d = recent_mean - older_mean
        else:
            trend_3d = 0.0

        result = {
            "ticker": ticker,
            "as_of": as_of.isoformat(),
            "n_articles": len(usable),
            "n_articles_3d": len(recent),
            "news_volume_z": round(volume_z, 4),
            "news_sentiment_mean": round(sentiment_mean, 4),
            "news_sentiment_trend_3d": round(trend_3d, 4),
            "sentiment_model": self.scorer.model_name,
            "earliest_article_at": min(a["published_at"] for a in usable).isoformat(),
            "latest_article_at": max(a["published_at"] for a in usable).isoformat(),
            "articles": [
                {
                    "headline": a["headline"][:200],
                    "source": a.get("source", ""),
                    "published_at": a["published_at"].isoformat(),
                    "sentiment": a["sentiment"],
                    "provider": a.get("provider", ""),
                }
                for a in usable
            ],
        }

        logger.info(
            "news_sentiment.analyzed",
            ticker=ticker, n=len(usable),
            sent_mean=result["news_sentiment_mean"],
            trend_3d=result["news_sentiment_trend_3d"],
            volume_z=result["news_volume_z"],
            model=self.scorer.model_name,
        )
        return result

    @staticmethod
    def score_for_ensemble(result: dict | None, direction: str) -> float:
        """Convert feature dict into a 0-100 direction-aware component score."""
        if not result or result["n_articles"] < NewsSentimentAnalyzer.MIN_ARTICLES:
            return 50.0

        sent = result["news_sentiment_mean"]
        trend = result["news_sentiment_trend_3d"]
        vol_z = result["news_volume_z"]

        if direction == "sell":
            sent, trend = -sent, -trend

        score = 50.0 + sent * 30.0 + trend * 15.0
        # Attention amplifier: unusual news volume strengthens whatever the
        # sentiment is saying (positive or negative), never creates a signal
        # on its own.
        if vol_z > 1.0:
            score = 50.0 + (score - 50.0) * (1.0 + min(vol_z, 3.0) * 0.15)

        return float(np.clip(score, 0.0, 100.0))
