from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
from .indicators import rsi, moving_average, atr, find_last_swing
from .models import TradeSignal, KeyLevel

def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    close = df["Close"]
    out = {
        "last": float(close.iloc[-1]),
        "chg_1d": float(close.pct_change().iloc[-1]),
        "chg_5d": float(close.pct_change(5).iloc[-1]),
        "vol_mean_20": float(df["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in df.columns else None,
        "ma20": float(moving_average(close, 20).iloc[-1]),
        "ma50": float(moving_average(close, 50).iloc[-1]),
        "ma200": float(moving_average(close, 200).iloc[-1]),
        "rsi14": float(rsi(close, 14).iloc[-1]),
        "atr14": float(atr(df, 14).iloc[-1])
    }
    return out

def compute_trends(df: pd.DataFrame) -> Dict[str, str]:
    close = df["Close"]
    ma20_val = float(moving_average(close, 20).iloc[-1])
    ma50_val = float(moving_average(close, 50).iloc[-1])
    ma200_val = float(moving_average(close, 200).iloc[-1])
    is_up = (ma20_val > ma50_val) and (ma50_val > ma200_val)
    is_down = (ma20_val < ma50_val) and (ma50_val < ma200_val)
    trend = "up" if is_up else ("down" if is_down else "sideways")
    return {"trend": trend}

def build_local_text_and_levels(df: pd.DataFrame, action:str, kpis:dict, trends:dict):
    c = df['Close']
    last = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    ma200 = float(c.rolling(200).mean().iloc[-1])
    rsi14 = float(rsi(c,14).iloc[-1])
    atr14 = float(atr(df,14).iloc[-1])
    swing_low  = find_last_swing(df['Low'],  window=5, mode="low")
    swing_high = find_last_swing(df['High'], window=5, mode="high")
    trend = trends.get("trend","sideways")
    sl = tp = None
    if action=="buy":
        base_sl = (last - 2*atr14) if not np.isnan(atr14) else (swing_low[1] if swing_low else last*0.97)
        sl = round(base_sl, 2); tp = round(last + 2*abs(last-sl), 2)
    elif action=="sell":
        base_sl = (last + 2*atr14) if not np.isnan(atr14) else (swing_high[1] if swing_high else last*1.03)
        sl = round(base_sl, 2); tp = round(last - 2*abs(sl-last), 2)
    txt = (f"Cierre {last:.2f}. MA20 {ma20:.2f}, MA50 {ma50:.2f}, MA200 {ma200:.2f}; "
           f"RSI14 {rsi14:.1f}. ATR14 {atr14:.2f}. Tendencia {trend}. ")
    if swing_low:  txt += f"Último swing-low {swing_low[1]:.2f} ({swing_low[0].date()}). "
    if swing_high: txt += f"Último swing-high {swing_high[1]:.2f} ({swing_high[0].date()}). "
    txt += ("Sesgo alcista; pullbacks a MA20/MA50."
            if action=='buy' else
            "Sesgo bajista; pullbacks a resistencias dinámicas."
            if action=='sell' else
            "Rango; esperar ruptura con volumen.")
    key_levels = [{"level": round(ma20,2), "label":"MA20"},
                  {"level": round(ma50,2), "label":"MA50"},
                  {"level": round(ma200,2), "label":"MA200"}]
        # Entry zone
    entry = (None, None)
    if not np.isnan(atr14) and atr14 > 0:
        if action == "buy":
            entry = (round(last - atr14, 2), round(last, 2))
        elif action == "sell":
            entry = (round(last, 2), round(last + atr14, 2))


    return dict(executive_summary="Estrategia cuantitativa con MAs/RSI/ATR.",
                technical_detail=txt, stop_loss=sl, take_profit=tp, timeframe_days=20, key_levels=key_levels, entry_zone=entry)

def fallback_signal(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict) -> TradeSignal:
    last = float(df["Close"].iloc[-1])
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    ma200 = float(df["Close"].rolling(200).mean().iloc[-1])
    rsi14 = float(kpis.get("rsi14", 50.0))
    atr14 = float(kpis.get("atr14", np.nan))
    if (rsi14 < 35 and last > ma50) or (ma20 > ma50 > ma200):
        action, dist = "buy", {"buy":0.6,"hold":0.3,"sell":0.1}
    elif (rsi14 > 65 and last < ma50) or (ma20 < ma50 < ma200):
        action, dist = "sell", {"buy":0.1,"hold":0.3,"sell":0.6}
    else:
        action, dist = "hold", {"buy":0.33,"hold":0.34,"sell":0.33}
    if not np.isnan(atr14) and atr14 > 0:
        if action == "buy":
            sl = last - 2*atr14; tp = last + 2*atr14
        elif action == "sell":
            sl = last + 2*atr14; tp = last - 2*atr14
        else:
            sl = tp = None
        rr = None if (sl is None or tp is None) else round(abs((tp-last)/(last-sl)), 2)
    else:
        sl = tp = rr = None
    extra = build_local_text_and_levels(df, action, kpis, trends)
    strat = {
        "setup_type": "tendencial",
        "executive_summary": extra["executive_summary"],
        "technical_detail": extra["technical_detail"],
        "entry_zone": (None, None),
        "stop_loss": round(sl,2) if sl is not None else extra["stop_loss"],
        "take_profit": round(tp,2) if tp is not None else extra["take_profit"],
        "risk_reward": rr,
        "timeframe_days": 20,
        "key_levels": extra["key_levels"]
    }
    return TradeSignal.model_validate({
        "symbol": symbol, "last_price": last, "action": action, "confidence": 0.55,
        "rationale": "Fallback cuantitativo local (RSI/MA/ATR).",
        "analysis": "Señal heurística basada en KPIs; niveles por swings y ATR.",
        "kpis": kpis, "trends": trends, "recommendation_distribution": dist, "strategy": strat
    })

def postprocess_signal(ts: TradeSignal, df: pd.DataFrame) -> TradeSignal:
    tech = (ts.strategy.technical_detail or "").strip().lower()
    generic = ("sin volumen ni patrones" in tech) or (len(tech) < 60)
    if generic or ts.strategy.stop_loss is None or ts.strategy.take_profit is None:
        extra = build_local_text_and_levels(df, ts.action, ts.kpis, ts.trends)
        if generic:
            ts.strategy.executive_summary = ts.strategy.executive_summary or extra["executive_summary"]
            ts.strategy.technical_detail  = extra["technical_detail"]
        if ts.strategy.stop_loss is None:   ts.strategy.stop_loss   = extra["stop_loss"]
        if ts.strategy.take_profit is None: ts.strategy.take_profit = extra["take_profit"]
        if ts.strategy.timeframe_days is None: ts.strategy.timeframe_days = extra["timeframe_days"]
        if not ts.strategy.key_levels: ts.strategy.key_levels = [KeyLevel(**k) for k in extra["key_levels"]]
        if ts.strategy.stop_loss is not None and ts.strategy.take_profit is not None:
            last = float(ts.last_price); sl, tp = ts.strategy.stop_loss, ts.strategy.take_profit
            if (last - sl) != 0:
                ts.strategy.risk_reward = round(abs((tp-last)/(last-sl)), 2)
    return ts
