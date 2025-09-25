from __future__ import annotations
import os, json
try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

from .models import TradeSignal
from .strategy import postprocess_signal, fallback_signal
from .indicators import rsi, atr
import pandas as pd

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
oa_client = OpenAI() if (_openai_available and os.getenv("OPENAI_API_KEY")) else None

def build_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "symbol": {"type":"string"},
            "last_price": {"type":"number"},
            "action": {"type":"string", "enum":["buy","hold","sell"]},
            "confidence": {"type":"number", "minimum":0, "maximum":1},
            "rationale": {"type":"string"},
            "analysis": {"type":"string"},
            "kpis": {"type":"object"},
            "trends": {"type":"object"},
            "recommendation_distribution": {
                "type":"object",
                "additionalProperties": False,
                "properties": {"buy":{"type":"number"}, "hold":{"type":"number"}, "sell":{"type":"number"}},
                "required":["buy","hold","sell"]
            },
            "strategy": {
                "type":"object",
                "additionalProperties": False,
                "properties": {
                    "setup_type":{"type":"string"},
                    "executive_summary":{"type":"string"},
                    "technical_detail":{"type":"string"},
                    "entry_zone":{"anyOf":[{"type":"array","items":[{"type":["number","null"]},{"type":["number","null"]}], "minItems":2, "maxItems":2},{"type":"null"}]},
                    "stop_loss":{"type":["number","null"]},
                    "take_profit":{"type":["number","null"]},
                    "risk_reward":{"type":["number","null"]},
                    "timeframe_days":{"type":["integer","null"]},
                    "key_levels":{"type":"array","items":{"type":"object","properties":{"level":{"type":"number"},"label":{"type":["string","null"]}}, "required":["level"]}}
                },
                "required":["setup_type","executive_summary","technical_detail","entry_zone","stop_loss","take_profit","risk_reward","timeframe_days","key_levels"]
            }
        },
        "required":["symbol","last_price","action","confidence","recommendation_distribution","strategy","kpis","trends"]
    }

def build_messages(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict):
    start_date = str(df.index[0].date())
    end_date = str(df.index[-1].date())
    sys = {"role":"system","content":(
        "Eres un analista cuantitativo. Devuelve únicamente JSON válido (estricto) según el schema. "
        "Evita generalidades. Incluye cifras exactas (2 decimales) y niveles coherentes con ATR y swings."
    )}
    last = float(df['Close'].iloc[-1])
    ma20 = float(df['Close'].rolling(20).mean().iloc[-1])
    ma50 = float(df['Close'].rolling(50).mean().iloc[-1])
    ma200 = float(df['Close'].rolling(200).mean().iloc[-1])
    rsi14 = float(rsi(df['Close'],14).iloc[-1])
    atr14 = float(atr(df,14).iloc[-1])
    feat = dict(last=round(last,2), ma20=round(ma20,2), ma50=round(ma50,2), ma200=round(ma200,2),
                rsi14=round(rsi14,2), atr14=round(atr14,2), trend=trends.get("trend","sideways"))
    user = {"role":"user","content": json.dumps({
        "task":"Generar señal BUY/HOLD/SELL con distribución y plan de trading específico.",
        "symbol": symbol, "data_window":{"start":start_date,"end":end_date},
        "kpis": kpis, "trends": trends, "recent_features": feat,
        "requirements":{"probabilities_sum_to_one": True, "avoid_generic_text": True,
                        "technical_detail_must_reference": ["RSI14","MA20","MA50","MA200","ATR14","trend"],
                        "timeframe_days_target": 20}
    }, ensure_ascii=False)}
    return [sys, user]

def ask(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict, deterministic: bool = True):
    if oa_client is None:
        ts = fallback_signal(symbol, df, kpis, trends)
        ts = postprocess_signal(ts, df)
        return ts, "fallback", {"text_enriched": True, "levels_completed": True}
    schema = build_schema()
    messages = build_messages(symbol, df, kpis, trends)
    DEFAULT_TEMP = 0.0 if deterministic else 0.4
    seed = int(os.getenv("OPENAI_SEED", "0")) if deterministic and os.getenv("OPENAI_SEED") else None
    def _one_call():
        kwargs = dict(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18"),
                      messages=messages,
                      response_format={"type":"json_schema","json_schema":{"name":"trade_signal","schema":schema,"strict":True}},
                      temperature=DEFAULT_TEMP)
        if seed is not None: kwargs["seed"] = seed
        return oa_client.chat.completions.create(**kwargs)
    try:
        comp = _one_call()
        data = json.loads(comp.choices[0].message.content)
        ts = TradeSignal.model_validate(data)
        d = ts.recommendation_distribution
        b,h,s = float(d.buy), float(d.hold), float(d.sell)
        tot = max(b+h+s, 1e-9); d.buy, d.hold, d.sell = b/tot, h/tot, s/tot
        argmax = max({"buy":d.buy,"hold":d.hold,"sell":d.sell}, key=lambda k: {"buy":d.buy,"hold":d.hold,"sell":d.sell}[k])
        if ts.action not in ("buy","hold","sell"): ts.action = argmax
        ts = postprocess_signal(ts, df)
        return ts, "openai", {"text_enriched": False, "levels_completed": False}
    except Exception:
        ts = fallback_signal(symbol, df, kpis, trends)
        ts = postprocess_signal(ts, df)
        return ts, "fallback", {"text_enriched": True, "levels_completed": True}
