
# -*- coding: utf-8 -*-
"""
Módulo de backtesting con señales técnicas y vetting opcional por OpenAI (determinista).
Requisitos:
    pip install pandas numpy pydantic openai==1.*
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False


# -------------------------------
# Señales técnicas básicas
# -------------------------------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

@dataclass
class Signal:
    date: pd.Timestamp
    side: str  # "buy" | "sell"
    price: float
    reason: str

def generate_signals(df: pd.DataFrame,
                     use_rsi: bool = True,
                     rsi_buy: float = 30.0,
                     rsi_sell: float = 70.0,
                     use_ma_cross: bool = True,
                     fast_ma: int = 20,
                     slow_ma: int = 50,
                     rsi_entry_mode: str = "exit_oversold"  # "enter_oversold" | "exit_oversold"
                     ) -> List[Signal]:
    close = df["Close"]
    signals: List[Signal] = []

    # RSI
    if use_rsi:
        r = rsi(close, 14)
        if rsi_entry_mode == "enter_oversold":
            cross_up = (r.shift(1) >= rsi_buy) & (r < rsi_buy)   # entrar en sobreventa (más agresivo)
        else:
            cross_up = (r.shift(1) <= rsi_buy) & (r > rsi_buy)   # salir de sobreventa (clásico)
        cross_dn = (r.shift(1) <= rsi_sell) & (r > rsi_sell)     # entrar en sobrecompra (vender)
        for idx in df.index[cross_up.fillna(False)]:
            signals.append(Signal(idx, "buy", float(close.loc[idx]), "RSI"))
        for idx in df.index[cross_dn.fillna(False)]:
            signals.append(Signal(idx, "sell", float(close.loc[idx]), "RSI"))

    # Cruce de medias
    if use_ma_cross:
        fast = close.rolling(fast_ma).mean()
        slow = close.rolling(slow_ma).mean()
        cross_up = (fast.shift(1) <= slow.shift(1)) & (fast > slow)
        cross_dn = (fast.shift(1) >= slow.shift(1)) & (fast < slow)
        for idx in df.index[cross_up.fillna(False)]:
            signals.append(Signal(idx, "buy", float(close.loc[idx]), "MA cross"))
        for idx in df.index[cross_dn.fillna(False)]:
            signals.append(Signal(idx, "sell", float(close.loc[idx]), "MA cross"))

    # Orden cronológico
    signals.sort(key=lambda s: (s.date, s.side))
    return signals


# -------------------------------
# Vetting por OpenAI (opcional)
# -------------------------------
def openai_vet_signals(symbol: str,
                       signals: List[Signal],
                       model: str = None) -> Dict[str, str]:
    """
    Devuelve un diccionario {key: "accept"|"reject"} donde key = f"{date_iso}|{side}".
    Se usa determinismo: temperature=0.0 y (si se define) seed en OPENAI_SEED.
    Si OpenAI no está disponible o falla, acepta todas.
    """
    if not _openai_available or not os.getenv("OPENAI_API_KEY"):
        return {f"{str(s.date.date())}|{s.side}": "accept" for s in signals}

    client = OpenAI()
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
    schema = {
        "type":"object",
        "additionalProperties": True,
        "properties":{}  # devolvemos mapa libre {key: "accept"/"reject"}
    }
    prompt_user = {
        "role":"user",
        "content": json_dumps_signals(symbol, signals)
    }

    kwargs = dict(
        model=model,
        messages=[
            {"role":"system","content":"Devuelve un JSON con claves '<YYYY-MM-DD>|<buy|sell>' y valores 'accept' o 'reject'."},
            prompt_user
        ],
        response_format={"type":"json_schema","json_schema":{"name":"backtest_decision","schema":schema,"strict":False}},
        temperature=0.0
    )
    seed_env = os.getenv("OPENAI_SEED")
    if seed_env:
        try:
            kwargs["seed"] = int(seed_env)
        except Exception:
            pass

    try:
        resp = client.chat.completions.create(**kwargs)
        data = resp.choices[0].message.content
        decisions = json.loads(data)
        out = {}
        for s in signals:
            k = f"{str(s.date.date())}|{s.side}"
            v = decisions.get(k, "accept")
            out[k] = "accept" if str(v).lower() == "accept" else "reject"
        return out
    except Exception:
        # Fallback: aceptar todo
        return {f"{str(s.date.date())}|{s.side}": "accept" for s in signals}

import json
def json_dumps_signals(symbol: str, signals: List[Signal]) -> str:
    compact = [
        {"date": str(s.date.date()), "side": s.side, "price": s.price, "reason": s.reason}
        for s in signals
    ]
    return json.dumps({"symbol":symbol, "signals":compact}, ensure_ascii=False)


# -------------------------------
# Backtest simple (buy/sell alternando)
# -------------------------------
def backtest(df: pd.DataFrame,
             signals: List[Signal],
             decisions_map: Optional[Dict[str,str]] = None,
             fee_bps: float = 2.0) -> pd.DataFrame:
    """
    Backtest elemental: entra/sale en cada señal aceptada; no permite overlays.
    fee_bps: comisiones en basis points por operación (2.0 => 0.02%).
    """
    cash = 1.0
    pos = 0.0
    last_price = float(df["Close"].iloc[0])
    equity_curve = []

    for idx, row in df.iterrows():
        price = float(row["Close"])
        date_iso = str(idx.date())

        # Ejecutar señal si existe en este día
        todays = [s for s in signals if str(s.date.date()) == date_iso]
        for s in todays:
            key = f"{date_iso}|{s.side}"
            if decisions_map and decisions_map.get(key, "accept") != "accept":
                continue
            # Trade
            if s.side == "buy" and cash > 0:
                pos = (cash * (1 - fee_bps/1e4)) / price
                cash = 0.0
            elif s.side == "sell" and pos > 0:
                cash = pos * price * (1 - fee_bps/1e4)
                pos = 0.0

        last_price = price
        equity = cash + pos * last_price
        equity_curve.append({"date": idx, "equity": equity})

    curve = pd.DataFrame(equity_curve).set_index("date")
    curve["ret"] = curve["equity"].pct_change().fillna(0.0)
    return curve
