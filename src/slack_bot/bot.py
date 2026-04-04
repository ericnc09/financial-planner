"""
Slack bot — listens for 'analyze' commands and runs the multi-model analysis.

Usage in Slack:
    analyze AAPL MSFT TSLA
    @bot analyze RY CM TD
"""

import asyncio
import re

import structlog
from slack_bolt.async_app import AsyncApp

from src.slack_bot.runner import run_analysis
from src.slack_bot.formatter import format_results

logger = structlog.get_logger()

# Words that look like tickers but aren't
TICKER_BLOCKLIST = {
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE",
    "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR",
    "SO", "TO", "UP", "US", "WE", "THE", "AND", "FOR", "ARE", "BUT",
    "NOT", "YOU", "ALL", "CAN", "HER", "WAS", "ONE", "OUR", "OUT",
    "HAS", "HIM", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
    "WAY", "WHO", "DID", "GET", "HIT", "LET", "SAY", "SHE", "TOO",
    "USE", "RUN", "BUY", "SELL", "HOLD", "ANALYZE", "CHECK", "PLEASE",
    "WHAT", "WHEN", "THIS", "THAT", "WITH", "FROM", "JUST", "LIKE",
    "ALSO", "BEEN", "EACH", "HAVE", "KEEP", "MANY", "MUCH", "NEXT",
    "ONLY", "OVER", "SOME", "TAKE", "THAN", "THEM", "THEN", "THEY",
    "VERY", "WELL", "WERE", "WILL", "YOUR", "THESE", "THOSE", "WHICH",
    "ABOUT", "AFTER", "COULD", "EVERY", "FIRST", "GREAT", "MIGHT",
    "OTHER", "THEIR", "THERE", "THINK", "WOULD", "COULD", "LOOK",
    "STOCK", "STOCKS", "PRICE", "TELL", "GIVE", "SHOW", "HELP",
    "GOOD", "BAD", "BEST", "WORST", "QUICK", "FAST",
}

MAX_TICKERS = 10

ANALYZE_PATTERN = re.compile(r"\banalyze\b", re.IGNORECASE)


def parse_tickers(text: str) -> list[str]:
    """Extract ticker symbols from a message, filtering out common English words."""
    # Remove the "analyze" keyword and any bot mention
    cleaned = re.sub(r"<@\w+>", "", text)
    cleaned = ANALYZE_PATTERN.sub("", cleaned)

    # Find all uppercase-looking words (1-5 chars)
    candidates = re.findall(r"\b([A-Za-z]{1,5})\b", cleaned)

    tickers = []
    seen = set()
    for c in candidates:
        upper = c.upper()
        if upper not in TICKER_BLOCKLIST and upper not in seen:
            tickers.append(upper)
            seen.add(upper)

    return tickers[:MAX_TICKERS]


def create_app(bot_token: str) -> AsyncApp:
    """Create and configure the Slack Bolt async app."""
    app = AsyncApp(token=bot_token)

    @app.message(ANALYZE_PATTERN)
    async def handle_analyze(message, say, client):
        tickers = parse_tickers(message["text"])
        if not tickers:
            await say(
                ":x: No valid tickers found.\n"
                "Usage: `analyze AAPL MSFT TSLA` (up to 10 tickers)"
            )
            return

        ticker_list = ", ".join(tickers)
        logger.info("slack_bot.analyze_requested", tickers=tickers, user=message.get("user"))

        await say(
            f":hourglass_flowing_sand: Analyzing *{ticker_list}*... "
            f"Running 5 models per ticker, this may take ~{len(tickers) * 15}s."
        )

        # Run analysis in the background so we don't block the event handler
        asyncio.create_task(
            _run_and_post(tickers, message["channel"], client)
        )

    @app.event("app_mention")
    async def handle_mention(event, say, client):
        text = event.get("text", "")
        if ANALYZE_PATTERN.search(text):
            # Delegate to the analyze handler
            tickers = parse_tickers(text)
            if not tickers:
                await say(
                    ":x: No valid tickers found.\n"
                    "Usage: `@bot analyze AAPL MSFT TSLA`"
                )
                return

            ticker_list = ", ".join(tickers)
            await say(
                f":hourglass_flowing_sand: Analyzing *{ticker_list}*... "
                f"Running 5 models per ticker, this may take ~{len(tickers) * 15}s."
            )
            asyncio.create_task(
                _run_and_post(tickers, event["channel"], client)
            )
        else:
            await say(
                "Hi! I can analyze stocks for you.\n"
                "Usage: `analyze AAPL MSFT TSLA` or `@bot analyze RY TD`"
            )

    @app.event("message")
    async def handle_other_messages():
        # Required to prevent warnings for unhandled message subtypes
        pass

    return app


async def _run_and_post(tickers: list[str], channel: str, client):
    """Run the analysis and post results back to Slack."""
    try:
        results, failed = await run_analysis(tickers)

        if not results and not failed:
            await client.chat_postMessage(
                channel=channel,
                text=":warning: Analysis returned no results.",
            )
            return

        payload = format_results(results, failed if failed else None)

        await client.chat_postMessage(
            channel=channel,
            text=f"Analysis complete for {len(results)} tickers",
            blocks=payload.get("blocks"),
            attachments=payload.get("attachments"),
        )

        logger.info(
            "slack_bot.results_posted",
            n_results=len(results),
            n_failed=len(failed),
            channel=channel,
        )

    except Exception as e:
        logger.error("slack_bot.analysis_failed", error=str(e))
        try:
            await client.chat_postMessage(
                channel=channel,
                text=f":x: Analysis failed: {e}",
            )
        except Exception:
            pass
