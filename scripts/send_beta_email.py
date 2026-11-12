"""
Send today's top-10 email to approved beta users. Run by cron at 7:30 AM on
weekdays, after the morning pipeline run has published the day's digest.

Usage:
    python scripts/send_beta_email.py
"""

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)  # cron's working directory isn't the repo; settings and data are relative
sys.path.insert(0, str(ROOT))

import httpx

from config.settings import Settings
from src.alerts.beta_email import send_top_10
from src.alerts.digest_publisher import DigestPublisher


async def main() -> None:
    digest = DigestPublisher().read_latest()
    async with httpx.AsyncClient(timeout=30.0) as client:
        await send_top_10(digest, Settings(), client)


if __name__ == "__main__":
    asyncio.run(main())
