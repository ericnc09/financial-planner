import argparse
import asyncio
from datetime import datetime

import structlog
from sqlalchemy.exc import IntegrityError

from config.settings import Settings
from src.clients.congress import CongressClient
from src.clients.edgar import EdgarClient
from src.clients.fred import FredClient
from src.clients.tiingo import TiingoClient
from src.models.database import (
    ConvictionScore,
    Direction,
    Enrichment,
    MacroSnapshot,
    SmartMoneyEvent,
    SourceType,
    get_session_factory,
    init_db,
)
from src.models.schemas import MacroRegime
from src.scoring.conviction_engine import ConvictionEngine
from src.scoring.fundamental_scorer import FundamentalScorer
from src.scoring.macro_scorer import MacroScorer
from src.scoring.signal_scorer import SignalScorer

logger = structlog.get_logger()


class Orchestrator:
    """Pipeline: ingest → enrich → macro → score. No execution — manual trades."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.edgar: EdgarClient | None = None
        self.congress: CongressClient | None = None
        self.tiingo: TiingoClient | None = None
        self.fred: FredClient | None = None
        self.engine = None
        self.session_factory = None
        self.conviction_engine: ConvictionEngine | None = None

    async def initialize(self):
        logger.info("orchestrator.initializing")

        # Database
        self.engine = init_db(self.settings.database_url)
        self.session_factory = get_session_factory(self.engine)

        # API clients (edgar + congress are free, no keys needed)
        self.edgar = EdgarClient()
        self.congress = CongressClient()
        self.tiingo = TiingoClient(self.settings.tiingo_api_key)
        self.fred = FredClient(self.settings.fred_api_key)

        # Scoring
        signal_scorer = SignalScorer()
        fundamental_scorer = FundamentalScorer()
        macro_scorer = MacroScorer()
        self.conviction_engine = ConvictionEngine(
            self.settings, signal_scorer, fundamental_scorer, macro_scorer
        )

        logger.info("orchestrator.initialized")

    async def run_cycle(self):
        logger.info("orchestrator.cycle_start")

        # --- Step 1: INGEST (SEC EDGAR + House/Senate Stock Watcher) ---
        insider_signals = await self.edgar.get_bulk_insider_trades(since_days=14)
        congressional_signals = await self.congress.get_all_congressional_trades(
            since_days=14
        )
        signals = insider_signals + congressional_signals
        signals.sort(key=lambda e: e.trade_date, reverse=True)
        new_events = self._persist_events(signals)
        logger.info(
            "orchestrator.ingested", new=len(new_events), total=len(signals)
        )

        if not new_events:
            logger.info("orchestrator.no_new_events")
            return

        # --- Step 2: ENRICH ---
        enrichments = await self._enrich_events(new_events)
        logger.info("orchestrator.enriched", count=len(enrichments))

        # --- Step 3: MACRO ---
        macro_snapshot = await asyncio.to_thread(self.fred.get_macro_snapshot)
        self._persist_macro_snapshot(macro_snapshot)

        # --- Step 4: SCORE ---
        scored = []
        for event_id, event_schema, enrichment_schema in enrichments:
            result = self.conviction_engine.compute(
                event_schema, enrichment_schema, macro_snapshot, signals
            )
            self._persist_conviction_score(event_id, result)
            scored.append((event_schema, enrichment_schema, result))

        passing = [s for s in scored if s[2].passes_threshold]
        logger.info(
            "orchestrator.scored",
            total=len(scored),
            passing=len(passing),
            threshold=self.settings.conviction_threshold,
        )

        # Log actionable signals for manual review
        for event, enrichment, result in passing:
            logger.info(
                "orchestrator.actionable_signal",
                ticker=event.ticker,
                actor=event.actor,
                direction=event.direction.value,
                conviction=result.conviction,
                price=enrichment.price_at_signal,
                sector=enrichment.sector,
            )

        logger.info("orchestrator.cycle_complete")

    def _persist_events(self, signals):
        session = self.session_factory()
        new_events = []
        try:
            for sig in signals:
                event = SmartMoneyEvent(
                    ticker=sig.ticker,
                    actor=sig.actor,
                    direction=Direction(sig.direction.value),
                    size_lower=sig.size_lower,
                    size_upper=sig.size_upper,
                    trade_date=sig.trade_date,
                    disclosure_date=sig.disclosure_date,
                    source_type=SourceType(sig.source_type.value),
                    raw_payload=sig.raw_payload,
                )
                session.add(event)
                try:
                    session.flush()
                    new_events.append((event.id, sig))
                except IntegrityError:
                    session.rollback()
            session.commit()
        finally:
            session.close()
        return new_events

    async def _enrich_events(self, new_events):
        sem = asyncio.Semaphore(5)
        results = []

        async def _enrich_one(event_id, event_schema):
            async with sem:
                try:
                    enrichment = await self.tiingo.enrich_ticker(
                        event_schema.ticker
                    )
                    self._persist_enrichment(event_id, enrichment)
                    return (event_id, event_schema, enrichment)
                except Exception as e:
                    logger.warning(
                        "orchestrator.enrich_failed",
                        ticker=event_schema.ticker,
                        error=str(e),
                    )
                    return None

        tasks = [_enrich_one(eid, schema) for eid, schema in new_events]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
        return results

    def _persist_enrichment(self, event_id, enrichment_schema):
        session = self.session_factory()
        try:
            enr = Enrichment(
                event_id=event_id,
                pe_ratio=enrichment_schema.pe_ratio,
                market_cap=enrichment_schema.market_cap,
                revenue_growth_yoy=enrichment_schema.revenue_growth_yoy,
                eps_latest=enrichment_schema.eps_latest,
                eps_growth_yoy=enrichment_schema.eps_growth_yoy,
                price_at_signal=enrichment_schema.price_at_signal,
                momentum_30d=enrichment_schema.momentum_30d,
                momentum_90d=enrichment_schema.momentum_90d,
                rsi_14d=enrichment_schema.rsi_14d,
                drawdown_from_52w_high=enrichment_schema.drawdown_from_52w_high,
                avg_volume_30d=enrichment_schema.avg_volume_30d,
                sector=enrichment_schema.sector,
            )
            session.add(enr)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_macro_snapshot(self, snapshot):
        session = self.session_factory()
        try:
            ms = MacroSnapshot(
                snapshot_date=snapshot.snapshot_date or datetime.utcnow(),
                yield_spread_10y2y=snapshot.yield_spread_10y2y,
                unemployment_claims=snapshot.unemployment_claims,
                cpi_yoy=snapshot.cpi_yoy,
                fed_funds_rate=snapshot.fed_funds_rate,
                regime=MacroRegime(snapshot.regime.value)
                if snapshot.regime
                else None,
                regime_score=snapshot.regime_score,
            )
            session.add(ms)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    def _persist_conviction_score(self, event_id, result):
        session = self.session_factory()
        try:
            cs = ConvictionScore(
                event_id=event_id,
                signal_score=result.signal_score,
                fundamental_score=result.fundamental_score,
                macro_modifier=result.macro_modifier,
                direction_boost=result.direction_boost,
                conviction=result.conviction,
                passes_threshold=result.passes_threshold,
            )
            session.add(cs)
            session.commit()
        except IntegrityError:
            session.rollback()
        finally:
            session.close()

    async def run_oneshot(self):
        await self.initialize()
        await self.run_cycle()
        await self.shutdown()

    async def run_daemon(self):
        from apscheduler.schedulers.asyncio import AsyncIOScheduler

        await self.initialize()
        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            self.run_cycle,
            "interval",
            minutes=self.settings.schedule_interval_minutes,
        )
        scheduler.start()
        logger.info(
            "orchestrator.daemon_started",
            interval_min=self.settings.schedule_interval_minutes,
        )
        await self.run_cycle()
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, SystemExit):
            scheduler.shutdown()
            await self.shutdown()

    async def shutdown(self):
        logger.info("orchestrator.shutting_down")
        if self.edgar:
            await self.edgar.close()
        if self.congress:
            await self.congress.close()
        if self.tiingo:
            await self.tiingo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Money Follows Pipeline")
    parser.add_argument(
        "--mode",
        choices=["oneshot", "daemon"],
        default=None,
        help="Run mode (overrides settings)",
    )
    args = parser.parse_args()

    settings = Settings()
    if args.mode:
        settings.pipeline_mode = args.mode

    orchestrator = Orchestrator(settings)

    if settings.pipeline_mode == "daemon":
        asyncio.run(orchestrator.run_daemon())
    else:
        asyncio.run(orchestrator.run_oneshot())
