"""
Entry point for the Slack bot (Socket Mode).

Usage:
    python -m src.slack_bot.main
    make run-slack-bot
"""

import asyncio
import sys

import structlog

from config.settings import Settings
from src.slack_bot.bot import create_app

logger = structlog.get_logger()


async def main():
    settings = Settings()

    if not settings.slack_bot_token:
        print("ERROR: SLACK_BOT_TOKEN not set. Add it to config/.env")
        print("  Create a Slack app at https://api.slack.com/apps")
        print("  Required Bot Token Scopes: chat:write, app_mentions:read, channels:history")
        sys.exit(1)

    if not settings.slack_app_token:
        print("ERROR: SLACK_APP_TOKEN not set. Add it to config/.env")
        print("  Enable Socket Mode in your Slack app settings to get an app-level token (xapp-...)")
        sys.exit(1)

    app = create_app(settings.slack_bot_token)

    # Import here to avoid import error if slack_bolt not installed
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    handler = AsyncSocketModeHandler(app, settings.slack_app_token)

    logger.info("slack_bot.starting", mode="socket_mode")
    print("Slack bot is running (Socket Mode). Press Ctrl+C to stop.")
    await handler.start_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSlack bot stopped.")
