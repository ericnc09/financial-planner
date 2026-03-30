import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import structlog
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import desc

from config.settings import Settings
from src.models.database import (
    ConvictionScore,
    Enrichment,
    MacroSnapshot,
    SmartMoneyEvent,
    SourceType,
    get_session_factory,
    init_db,
)
from src.pipeline.orchestrator import Orchestrator

logger = structlog.get_logger()
settings = Settings()


# --- Response Models ---


class SignalResponse(BaseModel):
    id: int
    ticker: str
    actor: str
    direction: str
    size_estimate: float | None = None
    trade_date: datetime
    source_type: str
    signal_score: float | None = None
    fundamental_score: float | None = None
    macro_modifier: float | None = None
    conviction: float | None = None
    passes_threshold: bool | None = None
    sector: str | None = None
    price_at_signal: float | None = None


class MacroResponse(BaseModel):
    snapshot_date: datetime
    yield_spread: float | None = None
    unemployment_claims: float | None = None
    cpi_yoy: float | None = None
    fed_funds_rate: float | None = None
    regime: str | None = None
    regime_score: float | None = None


class DashboardResponse(BaseModel):
    signals: list[SignalResponse]
    macro: MacroResponse | None = None


# --- App Setup ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = init_db(settings.database_url)
    app.state.session_factory = get_session_factory(engine)
    app.state.settings = settings
    yield


app = FastAPI(
    title="Smart Money Follows",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Endpoints ---


@app.get("/")
def root():
    return {"name": "Smart Money Follows", "version": "1.0.0", "docs": "/docs", "dashboard": "/api/dashboard"}


@app.get("/api/signals", response_model=list[SignalResponse])
def get_signals(
    days: int = Query(default=14, ge=1, le=365),
    source: str | None = Query(default=None),
    min_conviction: float | None = Query(default=None, ge=0, le=1),
):
    session = app.state.session_factory()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = (
            session.query(SmartMoneyEvent, ConvictionScore, Enrichment)
            .outerjoin(
                ConvictionScore,
                SmartMoneyEvent.id == ConvictionScore.event_id,
            )
            .outerjoin(Enrichment, SmartMoneyEvent.id == Enrichment.event_id)
            .filter(SmartMoneyEvent.trade_date >= cutoff)
        )
        if source:
            try:
                source_enum = SourceType(source)
                query = query.filter(SmartMoneyEvent.source_type == source_enum)
            except ValueError:
                pass  # invalid source value, skip filter
        query = query.order_by(desc(SmartMoneyEvent.trade_date))

        results = []
        for event, cs, enr in query.all():
            conviction = cs.conviction if cs else None
            if min_conviction is not None and (
                conviction is None or conviction < min_conviction
            ):
                continue
            size_est = None
            if event.size_lower and event.size_upper:
                size_est = (event.size_lower + event.size_upper) / 2
            elif event.size_lower:
                size_est = event.size_lower

            results.append(
                SignalResponse(
                    id=event.id,
                    ticker=event.ticker,
                    actor=event.actor,
                    direction=event.direction.value,
                    size_estimate=size_est,
                    trade_date=event.trade_date,
                    source_type=event.source_type.value,
                    signal_score=cs.signal_score if cs else None,
                    fundamental_score=cs.fundamental_score if cs else None,
                    macro_modifier=cs.macro_modifier if cs else None,
                    conviction=conviction,
                    passes_threshold=cs.passes_threshold if cs else None,
                    sector=enr.sector if enr else None,
                    price_at_signal=enr.price_at_signal if enr else None,
                )
            )
        return results
    finally:
        session.close()


@app.get("/api/signals/{ticker}", response_model=list[SignalResponse])
def get_signals_by_ticker(ticker: str):
    session = app.state.session_factory()
    try:
        query = (
            session.query(SmartMoneyEvent, ConvictionScore, Enrichment)
            .outerjoin(
                ConvictionScore,
                SmartMoneyEvent.id == ConvictionScore.event_id,
            )
            .outerjoin(Enrichment, SmartMoneyEvent.id == Enrichment.event_id)
            .filter(SmartMoneyEvent.ticker == ticker.upper())
            .order_by(desc(SmartMoneyEvent.trade_date))
        )
        results = []
        for event, cs, enr in query.all():
            size_est = None
            if event.size_lower and event.size_upper:
                size_est = (event.size_lower + event.size_upper) / 2
            results.append(
                SignalResponse(
                    id=event.id,
                    ticker=event.ticker,
                    actor=event.actor,
                    direction=event.direction.value,
                    size_estimate=size_est,
                    trade_date=event.trade_date,
                    source_type=event.source_type.value,
                    signal_score=cs.signal_score if cs else None,
                    fundamental_score=cs.fundamental_score if cs else None,
                    macro_modifier=cs.macro_modifier if cs else None,
                    conviction=cs.conviction if cs else None,
                    passes_threshold=cs.passes_threshold if cs else None,
                    sector=enr.sector if enr else None,
                    price_at_signal=enr.price_at_signal if enr else None,
                )
            )
        return results
    finally:
        session.close()


@app.get("/api/macro", response_model=MacroResponse | None)
def get_macro():
    session = app.state.session_factory()
    try:
        snapshot = (
            session.query(MacroSnapshot)
            .order_by(desc(MacroSnapshot.snapshot_date))
            .first()
        )
        if not snapshot:
            return None
        return MacroResponse(
            snapshot_date=snapshot.snapshot_date,
            yield_spread=snapshot.yield_spread_10y2y,
            unemployment_claims=snapshot.unemployment_claims,
            cpi_yoy=snapshot.cpi_yoy,
            fed_funds_rate=snapshot.fed_funds_rate,
            regime=snapshot.regime.value if snapshot.regime else None,
            regime_score=snapshot.regime_score,
        )
    finally:
        session.close()


@app.get("/api/macro/history", response_model=list[MacroResponse])
def get_macro_history(days: int = Query(default=90, ge=1, le=365)):
    session = app.state.session_factory()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        snapshots = (
            session.query(MacroSnapshot)
            .filter(MacroSnapshot.snapshot_date >= cutoff)
            .order_by(desc(MacroSnapshot.snapshot_date))
            .all()
        )
        return [
            MacroResponse(
                snapshot_date=s.snapshot_date,
                yield_spread=s.yield_spread_10y2y,
                unemployment_claims=s.unemployment_claims,
                cpi_yoy=s.cpi_yoy,
                fed_funds_rate=s.fed_funds_rate,
                regime=s.regime.value if s.regime else None,
                regime_score=s.regime_score,
            )
            for s in snapshots
        ]
    finally:
        session.close()


@app.post("/api/pipeline/run")
async def trigger_pipeline():
    orchestrator = Orchestrator(app.state.settings)
    asyncio.create_task(_run_pipeline(orchestrator))
    return {"status": "started", "message": "Pipeline cycle triggered in background"}


async def _run_pipeline(orchestrator: Orchestrator):
    try:
        await orchestrator.run_oneshot()
    except Exception as e:
        logger.error("api.pipeline_run_failed", error=str(e))


@app.get("/api/dashboard", response_model=DashboardResponse)
def get_dashboard():
    signals = get_signals(days=14, source=None, min_conviction=None)
    macro = get_macro()
    return DashboardResponse(
        signals=signals[:20],
        macro=macro,
    )
