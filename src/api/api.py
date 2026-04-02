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
    BayesianDecayResult,
    ConvictionScore,
    CopulaTailRiskResult,
    EnsembleScoreResult,
    Enrichment,
    EventStudyResult,
    ExtendedMacroData,
    FamaFrenchExposure,
    GARCHForecast,
    HMMRegimeState,
    MacroSnapshot,
    MeanVarianceResult,
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


class EventStudyResponse(BaseModel):
    ticker: str
    event_id: int
    direction: str
    source_type: str | None = None
    car_1d: float | None = None
    car_5d: float | None = None
    car_10d: float | None = None
    car_20d: float | None = None
    t_statistic: float | None = None
    p_value: float | None = None
    is_significant: bool | None = None
    daily_cars: list[float] = []


class EventStudyAggregateResponse(BaseModel):
    n_events: int
    n_significant: int | None = None
    pct_significant: float | None = None
    mean_car_1d: float | None = None
    mean_car_5d: float | None = None
    mean_car_10d: float | None = None
    mean_car_20d: float | None = None
    t_stat_5d: float | None = None
    p_value_5d: float | None = None
    win_rate_5d: float | None = None
    t_stat_20d: float | None = None
    p_value_20d: float | None = None
    win_rate_20d: float | None = None
    by_direction: dict | None = None
    by_source: dict | None = None


class CopulaTailRiskResponse(BaseModel):
    ticker: str
    run_date: datetime
    gaussian_rho: float | None = None
    student_t_rho: float | None = None
    student_t_nu: float | None = None
    tail_dep_lower: float | None = None
    tail_dep_upper: float | None = None
    joint_crash_prob: float | None = None
    tail_dep_ratio: float | None = None
    var_95: float | None = None
    var_99: float | None = None
    cvar_95: float | None = None
    cvar_99: float | None = None
    conditional_var_95: float | None = None
    conditional_cvar_95: float | None = None
    tail_risk_score: float | None = None


class BayesianDecayResponse(BaseModel):
    ticker: str
    event_id: int
    direction: str
    total_car: float | None = None
    posterior_half_life: float | None = None
    entry_window_days: float | None = None
    exit_window_days: float | None = None
    annualized_ir: float | None = None
    decay_quality: str | None = None
    signal_strength_5d: float | None = None
    signal_strength_20d: float | None = None


class MeanVarianceResponse(BaseModel):
    run_date: datetime
    n_assets: int
    tickers: list[str]
    max_sharpe: dict
    min_variance: dict
    equal_weight: dict
    efficient_frontier: list[dict]
    risk_contribution: dict


class EnsembleScoreResponse(BaseModel):
    ticker: str
    event_id: int
    direction: str
    total_score: float
    confidence: float | None = None
    recommendation: str | None = None
    n_models: int | None = None
    components: dict


class TickerAnalysisResponse(BaseModel):
    ticker: str
    monte_carlo: MonteCarloResponse | None = None
    hmm: HMMResponse | None = None
    garch: GARCHResponse | None = None
    fama_french: FamaFrenchResponse | None = None
    copula_tail_risk: CopulaTailRiskResponse | None = None
    event_studies: list[EventStudyResponse] | None = None
    bayesian_decay: list[BayesianDecayResponse] | None = None
    ensemble_scores: list[EnsembleScoreResponse] | None = None


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

        # Copula Tail Risk
        copula_row = session.query(CopulaTailRiskResult).filter(
            CopulaTailRiskResult.ticker == ticker
        ).order_by(desc(CopulaTailRiskResult.run_date)).first()

        copula_resp = None
        if copula_row:
            copula_resp = CopulaTailRiskResponse(
                ticker=copula_row.ticker, run_date=copula_row.run_date,
                gaussian_rho=copula_row.gaussian_rho, student_t_rho=copula_row.student_t_rho,
                student_t_nu=copula_row.student_t_nu,
                tail_dep_lower=copula_row.tail_dep_lower, tail_dep_upper=copula_row.tail_dep_upper,
                joint_crash_prob=copula_row.joint_crash_prob, tail_dep_ratio=copula_row.tail_dep_ratio,
                var_95=copula_row.var_95, var_99=copula_row.var_99,
                cvar_95=copula_row.cvar_95, cvar_99=copula_row.cvar_99,
                conditional_var_95=copula_row.conditional_var_95, conditional_cvar_95=copula_row.conditional_cvar_95,
                tail_risk_score=copula_row.tail_risk_score,
            )

        # Event studies
        import json as _json
        es_rows = session.query(EventStudyResult).filter(
            EventStudyResult.ticker == ticker
        ).order_by(desc(EventStudyResult.run_date)).limit(20).all()

        es_resp = None
        if es_rows:
            es_resp = []
            for es in es_rows:
                daily = []
                if es.daily_cars:
                    try:
                        daily = _json.loads(es.daily_cars)
                    except Exception:
                        pass
                es_resp.append(EventStudyResponse(
                    ticker=es.ticker, event_id=es.event_id,
                    direction=es.direction, source_type=es.source_type,
                    car_1d=es.car_1d, car_5d=es.car_5d, car_10d=es.car_10d, car_20d=es.car_20d,
                    t_statistic=es.t_statistic, p_value=es.p_value, is_significant=es.is_significant,
                    daily_cars=daily,
                ))

        # Bayesian Decay
        bd_rows = session.query(BayesianDecayResult).filter(
            BayesianDecayResult.ticker == ticker
        ).order_by(desc(BayesianDecayResult.run_date)).limit(20).all()

        bd_resp = None
        if bd_rows:
            bd_resp = [BayesianDecayResponse(
                ticker=bd.ticker, event_id=bd.event_id, direction=bd.direction,
                total_car=bd.total_car, posterior_half_life=bd.posterior_half_life,
                entry_window_days=bd.entry_window_days, exit_window_days=bd.exit_window_days,
                annualized_ir=bd.annualized_ir, decay_quality=bd.decay_quality,
                signal_strength_5d=bd.signal_strength_5d, signal_strength_20d=bd.signal_strength_20d,
            ) for bd in bd_rows]

        # Ensemble Scores
        ens_rows = session.query(EnsembleScoreResult).filter(
            EnsembleScoreResult.ticker == ticker
        ).order_by(desc(EnsembleScoreResult.run_date)).limit(20).all()

        ens_resp = None
        if ens_rows:
            ens_resp = [EnsembleScoreResponse(
                ticker=e.ticker, event_id=e.event_id, direction=e.direction,
                total_score=e.total_score, confidence=e.confidence,
                recommendation=e.recommendation, n_models=e.n_models,
                components={
                    k: v for k, v in {
                        "monte_carlo": e.score_monte_carlo, "hmm_regime": e.score_hmm,
                        "garch": e.score_garch, "fama_french": e.score_fama_french,
                        "copula_tail": e.score_copula, "bayesian_decay": e.score_bayesian_decay,
                        "event_study": e.score_event_study,
                    }.items() if v is not None
                },
            ) for e in ens_rows]

        return TickerAnalysisResponse(
            ticker=ticker,
            monte_carlo=mc_resp,
            hmm=hmm_resp,
            garch=garch_resp,
            fama_french=ff_resp,
            copula_tail_risk=copula_resp,
            event_studies=es_resp,
            bayesian_decay=bd_resp,
            ensemble_scores=ens_resp,
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


@app.get("/api/analysis/event-study/summary", response_model=EventStudyAggregateResponse)
def get_event_study_summary():
    """Aggregate event study statistics across all events."""
    import json
    import numpy as np
    from scipy import stats as sp_stats

    session = app.state.session_factory()
    try:
        rows = session.query(EventStudyResult).all()
        if not rows:
            return EventStudyAggregateResponse(n_events=0)

        def _agg_horizon(values: list[float]) -> dict:
            if len(values) < 2:
                return {"mean": values[0] if values else None, "t_stat": None, "p_value": None, "win_rate": None}
            arr = np.array(values)
            mean = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            t = mean / se if se > 0 else 0
            p = float(2 * (1 - sp_stats.t.cdf(abs(t), df=len(arr) - 1)))
            return {"mean": round(mean, 6), "t_stat": round(t, 4), "p_value": round(p, 4), "win_rate": round(float(np.mean(arr > 0)), 4)}

        cars_1d = [r.car_1d for r in rows if r.car_1d is not None]
        cars_5d = [r.car_5d for r in rows if r.car_5d is not None]
        cars_10d = [r.car_10d for r in rows if r.car_10d is not None]
        cars_20d = [r.car_20d for r in rows if r.car_20d is not None]

        agg_5d = _agg_horizon(cars_5d)
        agg_20d = _agg_horizon(cars_20d)
        sig_count = sum(1 for r in rows if r.is_significant)

        # By direction
        by_dir = {}
        for d in ["buy", "sell"]:
            sub = [r for r in rows if r.direction == d]
            sub_5d = [r.car_5d for r in sub if r.car_5d is not None]
            sub_20d = [r.car_20d for r in sub if r.car_20d is not None]
            by_dir[d] = {
                "n": len(sub),
                "car_5d": _agg_horizon(sub_5d),
                "car_20d": _agg_horizon(sub_20d),
            }

        # By source
        by_src = {}
        for s in ["insider", "congressional"]:
            sub = [r for r in rows if r.source_type == s]
            sub_5d = [r.car_5d for r in sub if r.car_5d is not None]
            sub_20d = [r.car_20d for r in sub if r.car_20d is not None]
            by_src[s] = {
                "n": len(sub),
                "car_5d": _agg_horizon(sub_5d),
                "car_20d": _agg_horizon(sub_20d),
            }

        return EventStudyAggregateResponse(
            n_events=len(rows),
            n_significant=sig_count,
            pct_significant=round(sig_count / len(rows), 4) if rows else None,
            mean_car_1d=_agg_horizon(cars_1d).get("mean"),
            mean_car_5d=agg_5d.get("mean"),
            mean_car_10d=_agg_horizon(cars_10d).get("mean"),
            mean_car_20d=agg_20d.get("mean"),
            t_stat_5d=agg_5d.get("t_stat"),
            p_value_5d=agg_5d.get("p_value"),
            win_rate_5d=agg_5d.get("win_rate"),
            t_stat_20d=agg_20d.get("t_stat"),
            p_value_20d=agg_20d.get("p_value"),
            win_rate_20d=agg_20d.get("win_rate"),
            by_direction=by_dir,
            by_source=by_src,
        )
    finally:
        session.close()


@app.get("/api/analysis/mean-variance", response_model=MeanVarianceResponse | None)
def get_mean_variance():
    """Get latest mean-variance portfolio optimization."""
    import json as _json
    session = app.state.session_factory()
    try:
        row = session.query(MeanVarianceResult).order_by(
            desc(MeanVarianceResult.run_date)
        ).first()
        if not row:
            return None
        return MeanVarianceResponse(
            run_date=row.run_date,
            n_assets=row.n_assets,
            tickers=_json.loads(row.tickers),
            max_sharpe={"weights": _json.loads(row.ms_weights), "expected_return": row.ms_return,
                        "volatility": row.ms_volatility, "sharpe_ratio": row.ms_sharpe},
            min_variance={"weights": _json.loads(row.mv_weights), "expected_return": row.mv_return,
                          "volatility": row.mv_volatility},
            equal_weight={"expected_return": row.ew_return, "volatility": row.ew_volatility,
                          "sharpe_ratio": row.ew_sharpe},
            efficient_frontier=_json.loads(row.efficient_frontier) if row.efficient_frontier else [],
            risk_contribution=_json.loads(row.risk_contribution) if row.risk_contribution else {},
        )
    finally:
        session.close()


@app.get("/api/analysis/ensemble/all", response_model=list[EnsembleScoreResponse])
def get_all_ensemble_scores():
    """Get latest ensemble scores for all events."""
    session = app.state.session_factory()
    try:
        from sqlalchemy import func
        subq = session.query(
            EnsembleScoreResult.event_id,
            func.max(EnsembleScoreResult.run_date).label("max_date")
        ).group_by(EnsembleScoreResult.event_id).subquery()

        rows = session.query(EnsembleScoreResult).join(
            subq,
            (EnsembleScoreResult.event_id == subq.c.event_id) &
            (EnsembleScoreResult.run_date == subq.c.max_date)
        ).order_by(desc(EnsembleScoreResult.total_score)).all()

        return [EnsembleScoreResponse(
            ticker=e.ticker, event_id=e.event_id, direction=e.direction,
            total_score=e.total_score, confidence=e.confidence,
            recommendation=e.recommendation, n_models=e.n_models,
            components={
                k: v for k, v in {
                    "monte_carlo": e.score_monte_carlo, "hmm_regime": e.score_hmm,
                    "garch": e.score_garch, "fama_french": e.score_fama_french,
                    "copula_tail": e.score_copula, "bayesian_decay": e.score_bayesian_decay,
                    "event_study": e.score_event_study,
                }.items() if v is not None
            },
        ) for e in rows]
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
