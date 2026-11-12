import httpx
import structlog

logger = structlog.get_logger()

WHATSAPP_API_BASE = "https://graph.facebook.com/v24.0"
WHATSAPP_TEMPLATE_LANGUAGE = "en"
WHATSAPP_MAX_VARIABLE_CHARS = 1024
WHATSAPP_MAX_SUMMARY_TICKERS = 20


class SlackNotifier:
    """Sends formatted alerts to Slack via webhook. Gracefully no-ops if no URL configured."""

    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url
        self._client = httpx.AsyncClient(timeout=10.0) if webhook_url else None

    async def send_conviction_alert(
        self,
        ticker: str,
        direction: str,
        actor: str,
        conviction: float,
        signal_score: float,
        fundamental_score: float,
        sector: str | None = None,
        price: float | None = None,
    ) -> bool:
        """Send alert when a signal passes the conviction threshold."""
        if not self._client:
            return False

        emoji = ":chart_with_upwards_trend:" if direction == "buy" else ":chart_with_downwards_trend:"
        color = "#22c55e" if direction == "buy" else "#ef4444"
        price_str = f"${price:.2f}" if price else "N/A"

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} {ticker} — {direction.upper()}",
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Actor:* {actor}"},
                                {"type": "mrkdwn", "text": f"*Conviction:* {conviction:.1%}"},
                                {"type": "mrkdwn", "text": f"*Signal Score:* {signal_score:.2f}"},
                                {"type": "mrkdwn", "text": f"*Fundamental Score:* {fundamental_score:.2f}"},
                                {"type": "mrkdwn", "text": f"*Sector:* {sector or 'N/A'}"},
                                {"type": "mrkdwn", "text": f"*Price:* {price_str}"},
                            ],
                        },
                    ],
                }
            ],
        }

        return await self._send(payload, "conviction", ticker)

    async def send_ensemble_alert(
        self,
        ticker: str,
        direction: str,
        ensemble_score: float,
        confidence: float,
        recommendation: str,
        components: dict,
    ) -> bool:
        """Send alert when ensemble score exceeds threshold (>70)."""
        if not self._client:
            return False

        component_lines = "\n".join(
            f"  {k}: {v:.1f}" for k, v in components.items() if v is not None
        )

        payload = {
            "attachments": [
                {
                    "color": "#6366f1",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f":brain: Ensemble Alert — {ticker} ({direction.upper()})",
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Score:* {ensemble_score:.1f}/100"},
                                {"type": "mrkdwn", "text": f"*Confidence:* {confidence:.1%}"},
                                {"type": "mrkdwn", "text": f"*Recommendation:* {recommendation}"},
                            ],
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Model Components:*\n```{component_lines}```",
                            },
                        },
                    ],
                }
            ],
        }

        return await self._send(payload, "ensemble", ticker)

    async def send_top_pick(self, pick: dict) -> bool:
        """Send the morning top-pick message."""
        if not self._client:
            return False

        ticker = pick["ticker"]
        score = pick["score"]
        runner_ups = pick.get("runner_ups") or []
        runner_text = (
            "  •  Runners-up: " + ", ".join(
                f"{r['ticker']} ({r['score']:.2f})" for r in runner_ups
            ) if runner_ups else ""
        )

        fields = [
            {"type": "mrkdwn", "text": f"*Composite Score:* {score:.2f}"},
            {"type": "mrkdwn", "text": f"*Sector:* {pick.get('sector') or 'N/A'}"},
        ]
        if pick.get("conviction") is not None:
            fields.append({"type": "mrkdwn", "text": f"*Conviction:* {pick['conviction']:.1%}"})
        if pick.get("ensemble_score") is not None:
            fields.append({"type": "mrkdwn", "text": f"*Ensemble:* {pick['ensemble_score']:.0f}/100"})
        if pick.get("mc_prob_profit_30d") is not None:
            fields.append({"type": "mrkdwn", "text": f"*P(profit, 30d):* {pick['mc_prob_profit_30d']:.0%}"})
        fields.append({"type": "mrkdwn", "text": f"*Smart-money buys ({pick.get('lookback_days', 7)}d):* {pick.get('n_signals', 0)} from {pick.get('n_distinct_actors', 0)} actor(s)"})

        payload = {
            "attachments": [
                {
                    "color": "#f59e0b",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f":sunrise: Morning Top Pick — {ticker}",
                            },
                        },
                        {"type": "section", "fields": fields},
                        {
                            "type": "context",
                            "elements": [{
                                "type": "mrkdwn",
                                "text": (
                                    f"Latest signal: {pick.get('latest_actor', '?')} "
                                    f"({pick.get('source_type', '?')}) on "
                                    f"{(pick.get('latest_trade_date') or '')[:10]}{runner_text}\n"
                                    "_Research signal, not financial advice._"
                                ),
                            }],
                        },
                    ],
                }
            ],
        }
        return await self._send(payload, "top_pick", ticker)

    async def send_digest(self, digest: dict, url: str) -> bool:
        """Send the one compiled daily message, linking to the full list."""
        if not self._client:
            return False

        lines = "\n".join(
            f"*{h['ticker']}*: {h['headline']}" for h in digest["highlights"]
        ) or "No insider buys stood out in this batch."
        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text", "text": f"Disclosure digest: {digest['date']}"},
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"{_digest_summary(digest)}\n\n{lines}"},
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"<{url}|See all {digest['counts']['trades']} trades>",
                    },
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": "_Research signal, not financial advice._"}
                    ],
                },
            ]
        }
        return await self._send(payload, "digest", digest["date"])

    async def _send(self, payload: dict, alert_type: str, ticker: str) -> bool:
        """Post payload to Slack webhook."""
        try:
            resp = await self._client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
            logger.info("slack.alert_sent", type=alert_type, ticker=ticker)
            return True
        except Exception as e:
            logger.warning("slack.send_failed", type=alert_type, ticker=ticker, error=str(e))
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()


class WhatsAppNotifier:
    """Sends alerts through approved WhatsApp Cloud API templates.

    WhatsApp only allows templates for business-initiated messages, so each alert
    maps to a template whose variables must match docs/WHATSAPP_SETUP.md in order.
    Gracefully no-ops if not fully configured.
    """

    def __init__(
        self,
        token: str | None = None,
        phone_number_id: str | None = None,
        to: str | None = None,
    ):
        self.to = to
        self.configured = bool(token and phone_number_id and to)
        self._url = f"{WHATSAPP_API_BASE}/{phone_number_id}/messages"
        self._client = (
            httpx.AsyncClient(timeout=10.0, headers={"Authorization": f"Bearer {token}"})
            if self.configured
            else None
        )

    async def send_conviction_alert(
        self,
        ticker: str,
        direction: str,
        actor: str,
        conviction: float,
        signal_score: float,
        fundamental_score: float,
        sector: str | None = None,
        price: float | None = None,
    ) -> bool:
        """Send alert when a signal passes the conviction threshold."""
        return await self._send_template(
            "conviction_alert",
            [
                ticker,
                direction.upper(),
                actor,
                f"{conviction:.1%}",
                f"{signal_score:.2f}",
                f"{fundamental_score:.2f}",
                sector,
                f"${price:.2f}" if price else None,
            ],
            ticker,
        )

    async def send_ensemble_alert(
        self,
        ticker: str,
        direction: str,
        ensemble_score: float,
        confidence: float,
        recommendation: str,
        components: dict,
    ) -> bool:
        """Send alert when ensemble score exceeds threshold (>70)."""
        component_text = ", ".join(
            f"{k}: {v:.1f}" for k, v in components.items() if v is not None
        )
        return await self._send_template(
            "ensemble_alert",
            [
                ticker,
                direction.upper(),
                f"{ensemble_score:.1f}/100",
                f"{confidence:.1%}",
                recommendation,
                component_text,
            ],
            ticker,
        )

    async def send_top_pick(self, pick: dict) -> bool:
        """Send the morning top-pick message."""
        ticker = pick["ticker"]
        conviction = pick.get("conviction")
        ensemble = pick.get("ensemble_score")
        prob_profit = pick.get("mc_prob_profit_30d")
        runner_ups = pick.get("runner_ups") or []
        return await self._send_template(
            "top_pick",
            [
                ticker,
                f"{pick['score']:.2f}",
                pick.get("sector"),
                f"{conviction:.1%}" if conviction is not None else None,
                f"{ensemble:.0f}/100" if ensemble is not None else None,
                f"{prob_profit:.0%}" if prob_profit is not None else None,
                (
                    f"{pick.get('n_signals', 0)} from {pick.get('n_distinct_actors', 0)} "
                    f"actor(s) in {pick.get('lookback_days', 7)}d"
                ),
                (
                    f"{pick.get('latest_actor', '?')} ({pick.get('source_type', '?')}) "
                    f"on {(pick.get('latest_trade_date') or '')[:10]}"
                ),
                ", ".join(f"{r['ticker']} ({r['score']:.2f})" for r in runner_ups),
            ],
            ticker,
        )

    async def send_analysis_summary(
        self, title: str, results: list[dict], failed: list[str] | None = None
    ) -> bool:
        """Send a one-message verdict summary for a multi-ticker analysis run."""
        return await self._send_template(
            "analysis_summary",
            [
                title,
                _verdict_list(results, "BUY", "buy_score"),
                _verdict_list(results, "HOLD"),
                _verdict_list(results, "SELL", "sell_score"),
                _limited_list(failed or []),
            ],
            "summary",
        )

    async def send_digest(self, digest: dict, url: str) -> bool:
        """Send the one compiled daily message, linking to the full list."""
        highlights = "; ".join(
            f"{h['ticker']}: {h['headline']}" for h in digest["highlights"]
        )
        return await self._send_template(
            "daily_digest",
            [digest["date"], _digest_summary(digest), highlights, url],
            "digest",
        )

    async def _send_template(
        self, template: str, variables: list, ticker: str
    ) -> bool:
        """Post a template message. A success only means Meta accepted it, not
        that it was delivered."""
        if not self._client:
            return False

        payload = {
            "messaging_product": "whatsapp",
            "to": self.to,
            "type": "template",
            "template": {
                "name": template,
                "language": {"code": WHATSAPP_TEMPLATE_LANGUAGE},
                "components": [
                    {
                        "type": "body",
                        "parameters": [
                            {"type": "text", "text": _template_text(v)} for v in variables
                        ],
                    }
                ],
            },
        }
        try:
            resp = await self._client.post(self._url, json=payload)
            resp.raise_for_status()
            logger.info("whatsapp.alert_accepted", template=template, ticker=ticker)
            return True
        except httpx.HTTPStatusError as e:
            logger.warning(
                "whatsapp.send_failed",
                template=template,
                ticker=ticker,
                status=e.response.status_code,
                error=e.response.text[:500],
            )
            return False
        except Exception as e:
            logger.warning("whatsapp.send_failed", template=template, ticker=ticker, error=str(e))
            return False

    async def close(self):
        if self._client:
            await self._client.aclose()


def create_notifier(settings) -> SlackNotifier | WhatsAppNotifier:
    """Build the notifier for the configured alert channel."""
    if settings.alert_channel != "whatsapp":
        return SlackNotifier(settings.slack_webhook_url)

    notifier = WhatsAppNotifier(
        settings.whatsapp_token,
        settings.whatsapp_phone_number_id,
        settings.whatsapp_to,
    )
    if not notifier.configured:
        logger.warning("whatsapp.not_configured", detail="alerts will not be sent")
    return notifier


def _digest_summary(digest: dict) -> str:
    counts = digest["counts"]
    return (
        f"{counts['trades']} new trades across {counts['tickers']} tickers: "
        f"{counts['buys']} buys, {counts['sells']} sells"
    )


def _template_text(value) -> str:
    """Make a value safe for a template variable, which can't be empty or contain
    newlines, tabs, or long runs of spaces."""
    text = " ".join(str(value).split()) if value is not None else ""
    return text[:WHATSAPP_MAX_VARIABLE_CHARS] or "N/A"


def _limited_list(items: list[str]) -> str:
    """Join items, capping the list so a large run still fits in one message."""
    text = ", ".join(items[:WHATSAPP_MAX_SUMMARY_TICKERS])
    hidden = len(items) - WHATSAPP_MAX_SUMMARY_TICKERS
    return f"{text} +{hidden} more" if hidden > 0 else text


def _verdict_list(results: list[dict], verdict: str, score_key: str | None = None) -> str:
    """List tickers with a verdict, strongest first when a score key is given."""
    matching = [r for r in results if r["verdict"] == verdict]
    if not score_key:
        return _limited_list([r["ticker"] for r in matching])
    matching.sort(key=lambda r: r[score_key], reverse=True)
    return _limited_list([f"{r['ticker']} ({r[score_key]:.0f})" for r in matching])
