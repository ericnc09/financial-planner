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
