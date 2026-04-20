"""Pytest configuration for the financial-planner test suite."""

import asyncio
import sys
from pathlib import Path

import pytest

# Ensure the repo root is on sys.path so `src.*` and `config.*` resolve when
# pytest is invoked from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    """Mark async tests that aren't explicitly marked, so pytest-asyncio picks them up."""
    for item in items:
        if asyncio.iscoroutinefunction(getattr(item, "function", None)):
            if not item.get_closest_marker("asyncio"):
                item.add_marker(pytest.mark.asyncio)
