import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class SourceType(enum.Enum):
    CONGRESSIONAL = "congressional"
    INSIDER = "insider"


class Direction(enum.Enum):
    BUY = "buy"
    SELL = "sell"


class MacroRegime(enum.Enum):
    EXPANSION = "expansion"
    TRANSITION = "transition"
    RECESSION = "recession"


class SmartMoneyEvent(Base):
    __tablename__ = "smart_money_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    actor = Column(String(200), nullable=False)
    direction = Column(Enum(Direction, values_callable=lambda x: [e.value for e in x]), nullable=False)
    size_lower = Column(Float, nullable=True)
    size_upper = Column(Float, nullable=True)
    trade_date = Column(DateTime, nullable=False)
    disclosure_date = Column(DateTime, nullable=True)
    source_type = Column(Enum(SourceType, values_callable=lambda x: [e.value for e in x]), nullable=False)
    raw_payload = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    enrichment = relationship("Enrichment", back_populates="event", uselist=False)
    conviction_score = relationship(
        "ConvictionScore", back_populates="event", uselist=False
    )

    __table_args__ = (
        UniqueConstraint(
            "ticker", "actor", "trade_date", "source_type", name="uq_event_dedup"
        ),
        Index("ix_events_date_ticker", "trade_date", "ticker"),
    )


class Enrichment(Base):
    __tablename__ = "enrichments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey("smart_money_events.id"), unique=True, nullable=False
    )

    # Fundamentals
    pe_ratio = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    revenue_growth_yoy = Column(Float, nullable=True)
    eps_latest = Column(Float, nullable=True)
    eps_growth_yoy = Column(Float, nullable=True)

    # Price action
    price_at_signal = Column(Float, nullable=True)
    momentum_30d = Column(Float, nullable=True)
    momentum_90d = Column(Float, nullable=True)
    rsi_14d = Column(Float, nullable=True)
    drawdown_from_52w_high = Column(Float, nullable=True)
    avg_volume_30d = Column(Float, nullable=True)

    sector = Column(String(100), nullable=True)
    enriched_at = Column(DateTime, default=datetime.utcnow)

    event = relationship("SmartMoneyEvent", back_populates="enrichment")


class MacroSnapshot(Base):
    __tablename__ = "macro_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(DateTime, nullable=False, unique=True, index=True)

    yield_spread_10y2y = Column(Float, nullable=True)
    unemployment_claims = Column(Float, nullable=True)
    cpi_yoy = Column(Float, nullable=True)
    fed_funds_rate = Column(Float, nullable=True)
    sp500_pe = Column(Float, nullable=True)

    regime = Column(Enum(MacroRegime, values_callable=lambda x: [e.value for e in x]), nullable=True)
    regime_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ConvictionScore(Base):
    __tablename__ = "conviction_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(
        Integer, ForeignKey("smart_money_events.id"), unique=True, nullable=False
    )

    signal_score = Column(Float, nullable=False)
    fundamental_score = Column(Float, nullable=False)
    macro_modifier = Column(Float, nullable=False)
    direction_boost = Column(Float, nullable=False)

    conviction = Column(Float, nullable=False, index=True)
    passes_threshold = Column(Boolean, nullable=False)
    scored_at = Column(DateTime, default=datetime.utcnow)

    event = relationship("SmartMoneyEvent", back_populates="conviction_score")


class MonteCarloResult(Base):
    __tablename__ = "monte_carlo_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False)

    current_price = Column(Float, nullable=False)
    annual_drift = Column(Float, nullable=True)
    annual_volatility = Column(Float, nullable=True)
    n_simulations = Column(Integer, nullable=False)

    # 30-day horizon
    h30_p10 = Column(Float, nullable=True)
    h30_p25 = Column(Float, nullable=True)
    h30_p50 = Column(Float, nullable=True)
    h30_p75 = Column(Float, nullable=True)
    h30_p90 = Column(Float, nullable=True)
    h30_prob_profit = Column(Float, nullable=True)
    h30_expected_return = Column(Float, nullable=True)
    h30_var_95 = Column(Float, nullable=True)

    # 90-day horizon
    h90_p10 = Column(Float, nullable=True)
    h90_p25 = Column(Float, nullable=True)
    h90_p50 = Column(Float, nullable=True)
    h90_p75 = Column(Float, nullable=True)
    h90_p90 = Column(Float, nullable=True)
    h90_prob_profit = Column(Float, nullable=True)
    h90_expected_return = Column(Float, nullable=True)
    h90_var_95 = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "run_date", name="uq_mc_ticker_date"),
    )


class HMMRegimeState(Base):
    __tablename__ = "hmm_regime_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False)

    current_state = Column(String(20), nullable=False)  # bull/bear/sideways
    prob_bull = Column(Float, nullable=True)
    prob_bear = Column(Float, nullable=True)
    prob_sideways = Column(Float, nullable=True)

    # Transition probabilities from current state
    trans_to_bull = Column(Float, nullable=True)
    trans_to_bear = Column(Float, nullable=True)
    trans_to_sideways = Column(Float, nullable=True)

    n_observations = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "run_date", name="uq_hmm_ticker_date"),
    )


class GARCHForecast(Base):
    __tablename__ = "garch_forecasts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False)

    # GARCH parameters
    omega = Column(Float, nullable=True)
    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    persistence = Column(Float, nullable=True)

    # Volatility values
    current_vol_annual = Column(Float, nullable=True)
    long_run_vol_annual = Column(Float, nullable=True)
    historical_vol_20d = Column(Float, nullable=True)
    historical_vol_60d = Column(Float, nullable=True)

    # Forecasts
    forecast_5d_vol = Column(Float, nullable=True)
    forecast_5d_ratio = Column(Float, nullable=True)
    forecast_5d_interpretation = Column(String(50), nullable=True)
    forecast_20d_vol = Column(Float, nullable=True)
    forecast_20d_ratio = Column(Float, nullable=True)
    forecast_20d_interpretation = Column(String(50), nullable=True)

    n_observations = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "run_date", name="uq_garch_ticker_date"),
    )


class FamaFrenchExposure(Base):
    __tablename__ = "fama_french_exposures"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False)

    alpha_daily = Column(Float, nullable=True)
    alpha_annual = Column(Float, nullable=True)
    beta_market = Column(Float, nullable=True)
    beta_smb = Column(Float, nullable=True)
    beta_hml = Column(Float, nullable=True)
    beta_rmw = Column(Float, nullable=True)
    beta_cma = Column(Float, nullable=True)
    r_squared = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "run_date", name="uq_ff_ticker_date"),
    )


class EventStudyResult(Base):
    __tablename__ = "event_study_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    event_id = Column(Integer, ForeignKey("smart_money_events.id"), nullable=False)
    run_date = Column(DateTime, nullable=False)

    direction = Column(String(10), nullable=False)
    source_type = Column(String(20), nullable=True)

    car_1d = Column(Float, nullable=True)
    car_5d = Column(Float, nullable=True)
    car_10d = Column(Float, nullable=True)
    car_20d = Column(Float, nullable=True)

    t_statistic = Column(Float, nullable=True)
    p_value = Column(Float, nullable=True)
    is_significant = Column(Boolean, nullable=True)

    alpha = Column(Float, nullable=True)
    beta = Column(Float, nullable=True)
    sigma_est = Column(Float, nullable=True)
    n_estimation = Column(Integer, nullable=True)

    daily_cars = Column(Text, nullable=True)  # JSON array of day-by-day CAR values

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("event_id", "run_date", name="uq_es_event_date"),
    )


class CopulaTailRiskResult(Base):
    __tablename__ = "copula_tail_risk_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    run_date = Column(DateTime, nullable=False)

    n_observations = Column(Integer, nullable=True)
    gaussian_rho = Column(Float, nullable=True)
    student_t_rho = Column(Float, nullable=True)
    student_t_nu = Column(Float, nullable=True)
    tail_dep_lower = Column(Float, nullable=True)
    tail_dep_upper = Column(Float, nullable=True)
    joint_crash_prob = Column(Float, nullable=True)
    tail_dep_ratio = Column(Float, nullable=True)
    var_95 = Column(Float, nullable=True)
    var_99 = Column(Float, nullable=True)
    cvar_95 = Column(Float, nullable=True)
    cvar_99 = Column(Float, nullable=True)
    conditional_var_95 = Column(Float, nullable=True)
    conditional_cvar_95 = Column(Float, nullable=True)
    tail_risk_score = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "run_date", name="uq_copula_ticker_date"),
    )


class BayesianDecayResult(Base):
    __tablename__ = "bayesian_decay_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    event_id = Column(Integer, ForeignKey("smart_money_events.id"), nullable=False)
    run_date = Column(DateTime, nullable=False)

    direction = Column(String(10), nullable=False)
    n_days = Column(Integer, nullable=True)
    total_car = Column(Float, nullable=True)
    posterior_half_life = Column(Float, nullable=True)
    posterior_lambda = Column(Float, nullable=True)
    ci_half_life_lower = Column(Float, nullable=True)
    ci_half_life_upper = Column(Float, nullable=True)
    entry_window_days = Column(Float, nullable=True)
    exit_window_days = Column(Float, nullable=True)
    annualized_ir = Column(Float, nullable=True)
    decay_quality = Column(String(20), nullable=True)
    signal_strength_5d = Column(Float, nullable=True)
    signal_strength_20d = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("event_id", "run_date", name="uq_bd_event_date"),
    )


class MeanVarianceResult(Base):
    __tablename__ = "mean_variance_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_date = Column(DateTime, nullable=False, unique=True, index=True)

    n_assets = Column(Integer, nullable=False)
    tickers = Column(Text, nullable=False)  # JSON array

    # Max Sharpe portfolio
    ms_weights = Column(Text, nullable=True)  # JSON {ticker: weight}
    ms_return = Column(Float, nullable=True)
    ms_volatility = Column(Float, nullable=True)
    ms_sharpe = Column(Float, nullable=True)

    # Min Variance portfolio
    mv_weights = Column(Text, nullable=True)
    mv_return = Column(Float, nullable=True)
    mv_volatility = Column(Float, nullable=True)

    # Equal Weight baseline
    ew_return = Column(Float, nullable=True)
    ew_volatility = Column(Float, nullable=True)
    ew_sharpe = Column(Float, nullable=True)

    efficient_frontier = Column(Text, nullable=True)  # JSON array
    risk_contribution = Column(Text, nullable=True)  # JSON {ticker: pct}

    created_at = Column(DateTime, default=datetime.utcnow)


class EnsembleScoreResult(Base):
    __tablename__ = "ensemble_score_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    event_id = Column(Integer, ForeignKey("smart_money_events.id"), nullable=False)
    run_date = Column(DateTime, nullable=False)

    direction = Column(String(10), nullable=False)
    total_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    recommendation = Column(String(20), nullable=True)
    n_models = Column(Integer, nullable=True)

    # Component scores
    score_monte_carlo = Column(Float, nullable=True)
    score_hmm = Column(Float, nullable=True)
    score_garch = Column(Float, nullable=True)
    score_fama_french = Column(Float, nullable=True)
    score_copula = Column(Float, nullable=True)
    score_bayesian_decay = Column(Float, nullable=True)
    score_event_study = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("event_id", "run_date", name="uq_ens_event_date"),
    )


class SignalPerformance(Base):
    __tablename__ = "signal_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(Integer, ForeignKey("smart_money_events.id"), unique=True, nullable=False)
    ticker = Column(String(10), nullable=False, index=True)
    direction = Column(String(10), nullable=False)
    source_type = Column(String(20), nullable=True)
    conviction = Column(Float, nullable=True)

    entry_price = Column(Float, nullable=True)
    entry_date = Column(DateTime, nullable=True)

    price_5d = Column(Float, nullable=True)
    price_10d = Column(Float, nullable=True)
    price_20d = Column(Float, nullable=True)
    price_60d = Column(Float, nullable=True)

    return_5d = Column(Float, nullable=True)
    return_10d = Column(Float, nullable=True)
    return_20d = Column(Float, nullable=True)
    return_60d = Column(Float, nullable=True)

    is_winner_5d = Column(Boolean, nullable=True)
    is_winner_20d = Column(Boolean, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExtendedMacroData(Base):
    __tablename__ = "extended_macro_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(DateTime, nullable=False, unique=True, index=True)

    vix = Column(Float, nullable=True)
    consumer_sentiment = Column(Float, nullable=True)
    money_supply_m2 = Column(Float, nullable=True)
    housing_starts = Column(Float, nullable=True)
    industrial_production = Column(Float, nullable=True)
    put_call_ratio = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


class OptionsFlow(Base):
    __tablename__ = "options_flow"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), nullable=False, index=True)
    analysis_date = Column(DateTime, nullable=False)
    pcr = Column(Float, nullable=True)
    unusual_volume_score = Column(Float, nullable=True)
    iv_skew = Column(Float, nullable=True)
    max_pain = Column(Float, nullable=True)
    nearest_expiry = Column(String(20), nullable=True)
    total_call_volume = Column(Integer, nullable=True)
    total_put_volume = Column(Integer, nullable=True)
    total_call_oi = Column(Integer, nullable=True)
    total_put_oi = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("ticker", "analysis_date", name="uq_options_ticker_date"),
    )


# --- Engine / Session helpers ---


def get_engine(database_url: str):
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(database_url, echo=False, connect_args=connect_args)


def get_session_factory(engine):
    return sessionmaker(bind=engine)


def init_db(database_url: str):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


if __name__ == "__main__":
    from config.settings import Settings

    settings = Settings()
    engine = init_db(settings.database_url)
    print(f"Database initialized at {settings.database_url}")
    for table in Base.metadata.sorted_tables:
        print(f"  > {table.name}")
