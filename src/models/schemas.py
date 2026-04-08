from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class Direction(str, Enum):
    BUY = "buy"
    SELL = "sell"


class SourceType(str, Enum):
    CONGRESSIONAL = "congressional"
    INSIDER = "insider"


class MacroRegime(str, Enum):
    EXPANSION = "expansion"
    TRANSITION = "transition"
    RECESSION = "recession"


class SmartMoneyEventSchema(BaseModel):
    ticker: str
    actor: str
    direction: Direction
    size_lower: float | None = None
    size_upper: float | None = None
    trade_date: datetime
    disclosure_date: datetime | None = None
    source_type: SourceType
    raw_payload: str | None = None


class EnrichmentSchema(BaseModel):
    pe_ratio: float | None = None
    market_cap: float | None = None
    revenue_growth_yoy: float | None = None
    eps_latest: float | None = None
    eps_growth_yoy: float | None = None
    price_at_signal: float | None = None
    momentum_30d: float | None = None
    momentum_90d: float | None = None
    rsi_14d: float | None = None
    drawdown_from_52w_high: float | None = None
    avg_volume_30d: float | None = None
    sector: str | None = None
    short_ratio: float | None = None


class MacroSnapshotSchema(BaseModel):
    snapshot_date: datetime | None = None
    yield_spread_10y2y: float | None = None
    unemployment_claims: float | None = None
    cpi_yoy: float | None = None
    fed_funds_rate: float | None = None
    regime: MacroRegime | None = None
    regime_score: float | None = None


class ConvictionResult(BaseModel):
    signal_score: float = Field(ge=0, le=1)
    fundamental_score: float = Field(ge=0, le=1)
    macro_modifier: float = Field(ge=0.5, le=1.5)
    direction_boost: float = Field(ge=0.8, le=1.2)
    conviction: float = Field(ge=0, le=1)
    passes_threshold: bool


class PositionTarget(BaseModel):
    ticker: str
    direction: Direction
    conviction: float
    weight: float
    qty: int
    sector: str | None = None
    reason: str = ""
