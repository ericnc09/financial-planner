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
    ExtendedMacroData,
    FamaFrenchExposure,
    GARCHForecast,
    HMMRegimeState,
    MacroSnapshot,
    MonteCarloResult,
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


class MonteCarloResponse(BaseModel):
    ticker: str
    run_date: datetime
    current_price: float
    annual_drift: float | None = None
    annual_volatility: float | None = None
    n_simulations: int
    horizons: dict  # {30: {percentiles, prob_profit, ...}, 90: {...}}


class HMMResponse(BaseModel):
    ticker: str
    run_date: datetime
    current_state: str
    prob_bull: float | None = None
    prob_bear: float | None = None
    prob_sideways: float | None = None
    trans_to_bull: float | None = None
    trans_to_bear: float | None = None
    trans_to_sideways: float | None = None


class GARCHResponse(BaseModel):
    ticker: str
    run_date: datetime
    persistence: float | None = None
    current_vol_annual: float | None = None
    long_run_vol_annual: float | None = None
    historical_vol_20d: float | None = None
    historical_vol_60d: float | None = None
    forecast_5d_vol: float | None = None
    forecast_5d_ratio: float | None = None
    forecast_5d_interpretation: str | None = None
    forecast_20d_vol: float | None = None
    forecast_20d_ratio: float | None = None
    forecast_20d_interpretation: str | None = None


class FamaFrenchResponse(BaseModel):
    ticker: str
    run_date: datetime
    alpha_annual: float | None = None
    beta_market: float | None = None
    beta_smb: float | None = None
    beta_hml: float | None = None
    beta_rmw: float | None = None
    beta_cma: float | None = None
    r_squared: float | None = None


class ExtendedMacroResponse(BaseModel):
    snapshot_date: datetime
    vix: float | None = None
    consumer_sentiment: float | None = None
    money_supply_m2: float | None = None
    housing_starts: float | None = None
    industrial_production: float | None = None
    put_call_ratio: float | None = None


class TickerAnalysisResponse(BaseModel):
    ticker: str
    monte_carlo: MonteCarloResponse | None = None
    hmm: HMMResponse | None = None
    garch: GARCHResponse | None = None
    fama_french: FamaFrenchResponse | None = None


class DashboardResponse(BaseModel):
    signals: list[SignalResponse]
    macro: MacroResponse | None = None
    extended_macro: ExtendedMacroResponse | None = None


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


@app.get("/api/analysis/{ticker}", response_model=TickerAnalysisResponse)
def get_ticker_analysis(ticker: str):
    """Get all analysis model results for a ticker."""
    ticker = ticker.upper()
    session = app.state.session_factory()
    try:
        mc = session.query(MonteCarloResult).filter(
            MonteCarloResult.ticker == ticker
        ).order_by(desc(MonteCarloResult.run_date)).first()

        hmm = session.query(HMMRegimeState).filter(
            HMMRegimeState.ticker == ticker
        ).order_by(desc(HMMRegimeState.run_date)).first()

        garch = session.query(GARCHForecast).filter(
            GARCHForecast.ticker == ticker
        ).order_by(desc(GARCHForecast.run_date)).first()

        ff = session.query(FamaFrenchExposure).filter(
            FamaFrenchExposure.ticker == ticker
        ).order_by(desc(FamaFrenchExposure.run_date)).first()

        mc_resp = None
        if mc:
            mc_resp = MonteCarloResponse(
                ticker=mc.ticker,
                run_date=mc.run_date,
                current_price=mc.current_price,
                annual_drift=mc.annual_drift,
                annual_volatility=mc.annual_volatility,
                n_simulations=mc.n_simulations,
                horizons={
                    30: {
                        "percentiles": {"p10": mc.h30_p10, "p25": mc.h30_p25, "p50": mc.h30_p50, "p75": mc.h30_p75, "p90": mc.h30_p90},
                        "probability_of_profit": mc.h30_prob_profit,
                        "expected_return": mc.h30_expected_return,
                        "value_at_risk_95": mc.h30_var_95,
                    },
                    90: {
                        "percentiles": {"p10": mc.h90_p10, "p25": mc.h90_p25, "p50": mc.h90_p50, "p75": mc.h90_p75, "p90": mc.h90_p90},
                        "probability_of_profit": mc.h90_prob_profit,
                        "expected_return": mc.h90_expected_return,
                        "value_at_risk_95": mc.h90_var_95,
                    },
                },
            )

        hmm_resp = None
        if hmm:
            hmm_resp = HMMResponse(
                ticker=hmm.ticker, run_date=hmm.run_date,
                current_state=hmm.current_state,
                prob_bull=hmm.prob_bull, prob_bear=hmm.prob_bear, prob_sideways=hmm.prob_sideways,
                trans_to_bull=hmm.trans_to_bull, trans_to_bear=hmm.trans_to_bear, trans_to_sideways=hmm.trans_to_sideways,
            )

        garch_resp = None
        if garch:
            garch_resp = GARCHResponse(
                ticker=garch.ticker, run_date=garch.run_date,
                persistence=garch.persistence,
                current_vol_annual=garch.current_vol_annual, long_run_vol_annual=garch.long_run_vol_annual,
                historical_vol_20d=garch.historical_vol_20d, historical_vol_60d=garch.historical_vol_60d,
                forecast_5d_vol=garch.forecast_5d_vol, forecast_5d_ratio=garch.forecast_5d_ratio,
                forecast_5d_interpretation=garch.forecast_5d_interpretation,
                forecast_20d_vol=garch.forecast_20d_vol, forecast_20d_ratio=garch.forecast_20d_ratio,
                forecast_20d_interpretation=garch.forecast_20d_interpretation,
            )

        ff_resp = None
        if ff:
            ff_resp = FamaFrenchResponse(
                ticker=ff.ticker, run_date=ff.run_date,
                alpha_annual=ff.alpha_annual, beta_market=ff.beta_market,
                beta_smb=ff.beta_smb, beta_hml=ff.beta_hml,
                beta_rmw=ff.beta_rmw, beta_cma=ff.beta_cma,
                r_squared=ff.r_squared,
            )

        return TickerAnalysisResponse(
            ticker=ticker,
            monte_carlo=mc_resp,
            hmm=hmm_resp,
            garch=garch_resp,
            fama_french=ff_resp,
        )
    finally:
        session.close()


@app.get("/api/analysis/hmm/all", response_model=list[HMMResponse])
def get_all_hmm_states():
    """Get latest HMM regime state for all tickers."""
    session = app.state.session_factory()
    try:
        from sqlalchemy import func
        subq = session.query(
            HMMRegimeState.ticker,
            func.max(HMMRegimeState.run_date).label("max_date")
        ).group_by(HMMRegimeState.ticker).subquery()

        results = session.query(HMMRegimeState).join(
            subq,
            (HMMRegimeState.ticker == subq.c.ticker) &
            (HMMRegimeState.run_date == subq.c.max_date)
        ).all()

        return [
            HMMResponse(
                ticker=r.ticker, run_date=r.run_date,
                current_state=r.current_state,
                prob_bull=r.prob_bull, prob_bear=r.prob_bear, prob_sideways=r.prob_sideways,
                trans_to_bull=r.trans_to_bull, trans_to_bear=r.trans_to_bear, trans_to_sideways=r.trans_to_sideways,
            ) for r in results
        ]
    finally:
        session.close()


@app.get("/api/macro/extended", response_model=ExtendedMacroResponse | None)
def get_extended_macro():
    session = app.state.session_factory()
    try:
        em = session.query(ExtendedMacroData).order_by(
            desc(ExtendedMacroData.snapshot_date)
        ).first()
        if not em:
            return None
        return ExtendedMacroResponse(
            snapshot_date=em.snapshot_date,
            vix=em.vix,
            consumer_sentiment=em.consumer_sentiment,
            money_supply_m2=em.money_supply_m2,
            housing_starts=em.housing_starts,
            industrial_production=em.industrial_production,
            put_call_ratio=em.put_call_ratio,
        )
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
    ext_macro = get_extended_macro()
    return DashboardResponse(
        signals=signals[:20],
        macro=macro,
        extended_macro=ext_macro,
    )
