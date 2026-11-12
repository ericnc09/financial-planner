"""
The one place that turns a signal's context into the classifier's feature
record. Training (API), prediction (API) and the ticker-analysis path all go
through here, so a feature can never mean one thing when the model is trained
and another when it is used.

Anything unknown is left as None. The classifier imputes it; substituting a
number here (0.0 for volume, say) would be read by the model as a real value.
"""

from src.analysis.xgboost_classifier import FEATURE_COLS

_ENRICHMENT_FIELDS = (
    "pe_ratio", "market_cap", "revenue_growth_yoy", "eps_latest",
    "eps_growth_yoy", "short_ratio", "momentum_30d", "momentum_90d",
    "rsi_14d", "drawdown_from_52w_high", "avg_volume_30d",
)
_CONVICTION_FIELDS = ("signal_score", "fundamental_score", "macro_modifier", "conviction")
_NEWS_FIELDS = ("news_volume_z", "news_sentiment_mean", "news_sentiment_trend_3d")


def _get(source, name):
    """Read `name` from a dict or an object; None when absent."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def build_feature_record(
    *,
    direction: str,
    enrichment=None,
    conviction=None,
    hmm=None,
    garch_vol: float | None = None,
    ff_alpha: float | None = None,
    copula_score: float | None = None,
    news=None,
    date: str | None = None,
) -> dict:
    """
    Args:
        direction: "buy" or "sell".
        enrichment: Dict/object with the enrichment fields (price momentum,
            fundamentals, volume).
        conviction: Dict/object with signal_score, fundamental_score,
            macro_modifier, conviction.
        hmm: Dict/object with prob_bull and prob_bear (folded regime
            probabilities).
        garch_vol: Annualised conditional volatility.
        ff_alpha: Annualised Fama-French alpha.
        copula_score: Tail-risk score 0-100.
        news: Dict/object with the three news features.
        date: ISO date of the signal (used only to order training rows).
    """
    record = {col: None for col in FEATURE_COLS}
    record["date"] = date

    for field in _ENRICHMENT_FIELDS:
        record[field] = _get(enrichment, field)
    for field in _CONVICTION_FIELDS:
        record[field] = _get(conviction, field)
    for field in _NEWS_FIELDS:
        record[field] = _get(news, field)

    record["hmm_bull_prob"] = _get(hmm, "prob_bull")
    record["hmm_bear_prob"] = _get(hmm, "prob_bear")
    record["garch_current_vol"] = garch_vol
    record["ff_alpha"] = ff_alpha
    record["copula_tail_score"] = copula_score
    record["direction_encoded"] = 1.0 if direction == "buy" else -1.0
    return record
