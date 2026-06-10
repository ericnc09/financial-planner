import httpx
import structlog

logger = structlog.get_logger()


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
