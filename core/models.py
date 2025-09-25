from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator

class Dist(BaseModel):
    buy: float = Field(ge=0.0)
    hold: float = Field(ge=0.0)
    sell: float = Field(ge=0.0)

class KeyLevel(BaseModel):
    level: float
    label: Optional[str] = None

class Strategy(BaseModel):
    setup_type: Optional[str] = None
    executive_summary: Optional[str] = None
    technical_detail: Optional[str] = None
    entry_zone: Optional[Tuple[Optional[float], Optional[float]]] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_reward: Optional[float] = None
    timeframe_days: Optional[int] = None
    key_levels: List[KeyLevel] = Field(default_factory=list)

class TradeSignal(BaseModel):
    symbol: str
    last_price: float
    action: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None
    analysis: Optional[str] = None
    kpis: Dict[str, Any] = Field(default_factory=dict)
    trends: Dict[str, Any] = Field(default_factory=dict)
    recommendation_distribution: Dist
    strategy: Strategy

    @field_validator("action")
    @classmethod
    def _check_action(cls, v:str):
        v = v.lower().strip()
        if v not in ("buy", "hold", "sell"):
            raise ValueError("action debe ser buy|hold|sell")
        return v
