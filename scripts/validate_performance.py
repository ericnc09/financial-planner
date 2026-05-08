"""
Performance gate for CI: fails the build if Sharpe < SHARPE_FLOOR or
IC < IC_FLOOR on the latest signal records.

Loads signal records from one of two sources, in order of preference:
  1. The SQLite DB at $DATABASE_URL (real records, written by the orchestrator).
  2. The fixture at tests/fixtures/synthetic_signals.json (seeded synthetic data,
     used in CI when no real DB has accumulated yet).

Exit codes:
  0 — gate passed (Sharpe >= floor AND IC >= floor)
  1 — gate failed
  2 — insufficient data to evaluate (does NOT fail CI by default; pass
      --strict to make insufficient-data a failure)

Environment:
  SHARPE_FLOOR (default 0.5)
  IC_FLOOR     (default 0.03)
  DATABASE_URL (optional; default config.settings)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "synthetic_signals.json"


def _records_from_db(database_url: str) -> list[dict] | None:
    """Pull (conviction, realized_return) pairs from the live DB. Returns None if the table is empty."""
    try:
        from src.models.database import (
            ConvictionScore,
            SignalPerformance,
            SmartMoneyEvent,
            get_engine,
            get_session_factory,
        )
    except Exception as e:
        print(f"[gate] could not import DB models: {e}", file=sys.stderr)
        return None

    engine = get_engine(database_url)
    session = get_session_factory(engine)()
    try:
        rows = (
            session.query(SmartMoneyEvent, ConvictionScore, SignalPerformance)
            .join(ConvictionScore, SmartMoneyEvent.id == ConvictionScore.event_id)
            .join(SignalPerformance, SmartMoneyEvent.id == SignalPerformance.event_id)
            .filter(SignalPerformance.return_20d.isnot(None))
            .all()
        )
    finally:
        session.close()

    if not rows:
        return None

    return [
        {
            "date": e.trade_date.isoformat() if e.trade_date else None,
            "conviction": float(cs.conviction or 0.0),
            "realized_return": float(perf.return_20d),
        }
        for (e, cs, perf) in rows
    ]


def _records_from_fixture(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    with path.open() as f:
        return json.load(f)


def compute_sharpe(returns: list[float]) -> float:
    """Annualized Sharpe ratio (zero risk-free), ddof=1."""
    if len(returns) < 2:
        return 0.0
    arr = np.array(returns, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std <= 0:
        return 0.0
    return mean / std * math.sqrt(252)


def compute_ic(scores: list[float], returns: list[float]) -> float:
    """Spearman rank correlation between score and realized return."""
    if len(scores) < 5 or len(set(scores)) < 2 or len(set(returns)) < 2:
        return 0.0
    rho, _ = spearmanr(scores, returns)
    if rho is None or math.isnan(rho):
        return 0.0
    return float(rho)


def main() -> int:
    ap = argparse.ArgumentParser(description="CI performance gate.")
    ap.add_argument("--sharpe-floor", type=float,
                    default=float(os.environ.get("SHARPE_FLOOR", 0.5)))
    ap.add_argument("--ic-floor", type=float,
                    default=float(os.environ.get("IC_FLOOR", 0.03)))
    ap.add_argument("--min-samples", type=int, default=20,
                    help="Minimum sample size before the gate is meaningful.")
    ap.add_argument("--strict", action="store_true",
                    help="Treat insufficient data as failure (default: skip gate).")
    ap.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE,
                    help="Synthetic-data fixture to fall back on when the DB is empty.")
    args = ap.parse_args()

    # Source data from DB if available; otherwise from fixture.
    records: list[dict] | None = None
    db_url = os.environ.get("DATABASE_URL")
    if db_url:
        records = _records_from_db(db_url)
        if records:
            print(f"[gate] loaded {len(records)} records from DB ({db_url})")
        else:
            print(f"[gate] DB at {db_url} has no records with return_20d; "
                  f"falling back to fixture.")

    if not records:
        try:
            records = _records_from_fixture(args.fixture)
            print(f"[gate] loaded {len(records)} records from fixture ({args.fixture})")
        except FileNotFoundError as e:
            print(f"[gate] no DB records and {e}", file=sys.stderr)
            return 2 if not args.strict else 1

    n = len(records)
    if n < args.min_samples:
        print(f"[gate] insufficient samples ({n} < {args.min_samples}); "
              f"{'failing' if args.strict else 'skipping gate'}.")
        return 1 if args.strict else 2

    scores = [r["conviction"] for r in records if r.get("realized_return") is not None]
    returns = [r["realized_return"] for r in records if r.get("realized_return") is not None]

    sharpe = compute_sharpe(returns)
    ic = compute_ic(scores, returns)

    print(f"[gate] n={n}  Sharpe={sharpe:.4f}  IC={ic:.4f}")
    print(f"[gate] floors: Sharpe>={args.sharpe_floor}  IC>={args.ic_floor}")

    failed = []
    if sharpe < args.sharpe_floor:
        failed.append(f"Sharpe {sharpe:.4f} < floor {args.sharpe_floor}")
    if ic < args.ic_floor:
        failed.append(f"IC {ic:.4f} < floor {args.ic_floor}")

    if failed:
        print("[gate] FAIL — " + "; ".join(failed), file=sys.stderr)
        return 1

    print("[gate] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
