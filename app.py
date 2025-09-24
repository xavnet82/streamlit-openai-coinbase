# app.py (precio actual + tendencias 3/6/12M + manejo robusto de errores)
import os
import uuid
import json
import time
from typing import Literal, Optional, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, confloat

# OpenAI SDK
from openai import OpenAI

# Coinbase Advanced Trade SDK (official)
from coinbase.rest import RESTClient

st.set_page_config(page_title="Trading Assistant (OpenAI + Coinbase)", page_icon="📈", layout="wide")
st.title("📈 Trading Assistant — OpenAI + Coinbase Advanced Trade")

# ---------------------
# Secrets & configuration
# ---------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
CB_API_KEY     = st.secrets.get("COINBASE_API_KEY", os.getenv("COINBASE_API_KEY"))
CB_API_SECRET  = st.secrets.get("COINBASE_API_SECRET", os.getenv("COINBASE_API_SECRET"))
CB_BASE_URL    = st.secrets.get("COINBASE_BASE_URL", os.getenv("COINBASE_BASE_URL", "api.coinbase.com"))
USE_SANDBOX    = bool(str(st.secrets.get("USE_SANDBOX", os.getenv("USE_SANDBOX", "false"))).lower() in ["1","true","yes"])

if not OPENAI_API_KEY:
    st.error("Falta OPENAI_API_KEY en secrets o variables de entorno.")
    st.stop()

# Instantiate clients
oa_client = OpenAI(api_key=OPENAI_API_KEY)
cb_kwargs = {}
if CB_API_KEY and CB_API_SECRET:
    cb_kwargs.update(dict(api_key=CB_API_KEY, api_secret=CB_API_SECRET))
if CB_BASE_URL:
    cb_kwargs.update(dict(base_url=CB_BASE_URL))
cb_client = RESTClient(**cb_kwargs)

# ---------------------
# JSON Schema models (STRICT)
# ---------------------
Action = Literal["buy", "sell", "hold"]

class KPIs(BaseModel):
    model_config = {"extra": "forbid"}
    rsi: confloat(ge=0, le=100) = Field(..., description="RSI 0..100")
    macd: float = Field(..., description="MACD line minus signal (approx)")
    price_change_24h_pct: float
    volume_change_24h_pct: float
    volatility_24h_pct: float

class Trends(BaseModel):
    model_config = {"extra": "forbid"}
    short_term: Literal["up","down","flat"]
    mid_term: Literal["up","down","flat"]
    long_term: Literal["up","down","flat"]

class RecommendationDist(BaseModel):
    model_config = {"extra": "forbid"}
    buy: confloat(ge=0, le=1)
    hold: confloat(ge=0, le=1)
    sell: confloat(ge=0, le=1)

class TradeSignal(BaseModel):
    model_config = {"extra": "forbid"}  # STRICT

    product_id: str
    action: Action
    confidence: confloat(ge=0, le=1)
    rationale: str
    analysis: str

    # KPIs y tendencias
    kpis: KPIs
    trends: Trends
    recommendation_distribution: RecommendationDist

    # Sizing sugerido
    size_type: Literal["quote", "base"]
    size: str
    time_horizon_days: int = Field(..., ge=0)

# Utility: enforce additionalProperties:false recursively in JSON Schema
def _enforce_no_additional_props(schema: dict):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            for v in schema.get("properties", {}).values():
                _enforce_no_additional_props(v)
        for v in schema.values():
            _enforce_no_additional_props(v)
    elif isinstance(schema, list):
        for item in schema:
            _enforce_no_additional_props(item)

# ---------------------
# Coinbase helpers (robust price + candles)
# ---------------------
def get_spot_price(pid: str) -> Optional[float]:
    """Intento robusto: get_product.price -> public market trades (mid) -> None"""
    # 1) product.price
    try:
        p = cb_client.get_product(pid)
        price = getattr(p, "price", None) or (p.get("price") if isinstance(p, dict) else None)
        if price is not None:
            return float(price)
    except Exception as e:
        st.debug(f"get_product error: {e}")
    # 2) public market trades
    try:
        resp = cb_client.get_public_market_trades(product_id=pid, limit=1)
        best_bid = getattr(resp, "best_bid", None) or (resp.get("best_bid") if isinstance(resp, dict) else None)
        best_ask = getattr(resp, "best_ask", None) or (resp.get("best_ask") if isinstance(resp, dict) else None)
        if best_bid and best_ask:
            return (float(best_bid) + float(best_ask)) / 2.0
        trades = getattr(resp, "trades", None) or (resp.get("trades") if isinstance(resp, dict) else None)
        if trades:
            last = trades[0]
            price = getattr(last, "price", None) or (last.get("price") if isinstance(last, dict) else None)
            if price: return float(price)
    except Exception as e:
        st.debug(f"get_public_market_trades error: {e}")
    return None

def _candles_to_df(candles: List[Dict]) -> pd.DataFrame:
    rows = []
    for c in candles:
        rows.append({
            "start": int(c["start"]),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["volume"]),
        })
    df = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["start"], unit="s", utc=True)
    return df

def _get_candles_try_all(pid: str, start_ts: int, end_ts: int, granularity: str="ONE_HOUR", limit: int=350) -> Optional[pd.DataFrame]:
    # 1) Público
    try:
        resp = cb_client.get_public_product_candles(product_id=pid, start=str(start_ts), end=str(end_ts), granularity=granularity, limit=limit)
        candles = getattr(resp, "candles", None) or (resp.get("candles") if isinstance(resp, dict) else None)
        if candles:
            return _candles_to_df(candles)
    except Exception as e:
        st.debug(f"get_public_product_candles error: {e}")
    # 2) Brokerage
    try:
        resp = cb_client.get_product_candles(product_id=pid, start=str(start_ts), end=str(end_ts), granularity=granularity, limit=limit)
        candles = getattr(resp, "candles", None) or (resp.get("candles") if isinstance(resp, dict) else None)
        if candles:
            return _candles_to_df(candles)
    except Exception as e:
        st.debug(f"get_product_candles error: {e}")
    return None

def get_hourly_candles_48h(pid: str) -> Optional[pd.DataFrame]:
    now = int(time.time())
    return _get_candles_try_all(pid, start_ts=now - 48*3600, end_ts=now, granularity="ONE_HOUR", limit=350)

def get_daily_candles_days(pid: str, days: int) -> Optional[pd.DataFrame]:
    now = int(time.time())
    start = now - days*86400
    return _get_candles_try_all(pid, start_ts=start, end_ts=now, granularity="ONE_DAY", limit=400)

# ---------------------
# KPIs y tendencias intra (1H) + tendencias 3/6/12M
# ---------------------
def compute_indicators(df: pd.DataFrame) -> Dict[str, float]:
    closes = df["close"].astype(float)
    window = 14
    delta = closes.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-9))
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1])

    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_diff = float((macd_line - signal).iloc[-1])

    last_close = float(closes.iloc[-1])
    v_range = (df["high"].iloc[-24:].max() - df["low"].iloc[-24:].min())
    vol_24h_pct = float(v_range / last_close * 100.0) if last_close > 0 else 0.0

    if len(closes) >= 25:
        price_change_24h_pct = float((closes.iloc[-1] / closes.iloc[-25] - 1.0) * 100.0)
    else:
        price_change_24h_pct = float((closes.iloc[-1] / closes.iloc[0] - 1.0) * 100.0)

    return {
        "rsi": round(rsi_val, 2),
        "macd": round(macd_diff, 6),
        "volatility_24h_pct": round(vol_24h_pct, 4),
        "price_change_24h_pct": round(price_change_24h_pct, 4),
    }

def compute_volume_change(df48h: pd.DataFrame) -> float:
    if len(df48h) < 48:
        return 0.0
    v_prev = float(df48h["volume"].iloc[-48:-24].sum())
    v_curr = float(df48h["volume"].iloc[-24:].sum())
    if v_prev <= 0:
        return 0.0
    return round((v_curr / v_prev - 1.0) * 100.0, 4)

def compute_trends(df48h: pd.DataFrame) -> Dict[str, str]:
    def dir_pct(series, lag):
        if len(series) <= lag: return 0.0
        return float(series.iloc[-1] / series.iloc[-lag] - 1.0)
    closes = df48h["close"]
    d6 = dir_pct(closes, 6)
    d24 = dir_pct(closes, 24)
    d48 = dir_pct(closes, 48) if len(closes) >= 49 else d24
    def to_label(x):
        th = 0.001
        return "up" if x > th else ("down" if x < -th else "flat")
    return {"short_term": to_label(d6), "mid_term": to_label(d24), "long_term": to_label(d48)}

def compute_pct_change(df: Optional[pd.DataFrame], lookback_days: int) -> Optional[float]:
    if df is None or df.empty: return None
    closes = df["close"].astype(float).reset_index(drop=True)
    if len(closes) <= lookback_days: return None
    try:
        return float((closes.iloc[-1] / closes.iloc[-lookback_days-1] - 1.0) * 100.0)
    except Exception:
        return None

# ---------------------
# OpenAI prompt (usa contexto real)
# ---------------------
def build_prompt(product_id: str, user_notes: str, price: Optional[float], kpis: Dict[str, float], trends: Dict[str, str]) -> list:
    system = (
        "Eres un analista cuantitativo experto. Devuelve una recomendación operativa SIMPLE (buy/sell/hold), "
        "análisis técnico claro y KPIs cuantificados. No uses derivados ni apalancamiento. Si la señal es débil, responde hold."
    )
    market_context = {"spot_price": price, "kpis": kpis or None, "trends": trends or None}
    user = f"""
Analiza el activo para decisión intradía/simple.
Activo (Coinbase product_id si aplica): {product_id}

TOMA ESTOS DATOS COMO FUENTE DE VERDAD (no los inventes ni los alteres):
market_context = {json.dumps(market_context, ensure_ascii=False)}

Instrucciones estrictas:
- Usa EXACTAMENTE los KPIs/tendencias de market_context cuando existan para rellenar el JSON (no estimes ni inventes).
- Si algún valor falta, puedes estimarlo conservadoramente, pero si la señal no es clara -> hold.
- Genera: action (buy/sell/hold), confidence (0..1), rationale breve, analysis extensa.
- Sizing: BUY en 'quote' (USD). SELL en 'base' (activo).
Notas del usuario: {user_notes or "-"}
"""
    return [{"role": "system", "content": system},{"role": "user", "content": user}]

def ask_openai_for_signal(product_id: str, user_notes: str, price: Optional[float], kpis: Dict[str, float], trends: Dict[str, str]):
    messages = build_prompt(product_id, user_notes, price, kpis, trends)
    schema = TradeSignal.model_json_schema()
    _enforce_no_additional_props(schema)

    completion = oa_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "trade_signal", "schema": schema, "strict": True}
        },
        temperature=0.2,
    )
    content = completion.choices[0].message.content
    data = json.loads(content)
    ts = TradeSignal.model_validate(data)
    dist = ts.recommendation_distribution
    s = float(dist.buy) + float(dist.hold) + float(dist.sell)
    if s > 0 and abs(1 - s) > 1e-6:
        dist.buy = float(dist.buy) / s
        dist.hold = float(dist.hold) / s
        dist.sell = float(dist.sell) / s
    return ts

# ---------------------
# UI
# ---------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    default_pid = st.secrets.get("DEFAULT_PRODUCT_ID", os.getenv("DEFAULT_PRODUCT_ID", "BTC-USD"))
    product_id = st.text_input("Acción/Cripto (Coinbase product_id para operar)", value=default_pid, help="Ej: BTC-USD, ETH-USD. Para activos fuera de Coinbase solo se hará el análisis.")
    user_notes = st.text_area("Notas (opcional)", value="", help="Contexto adicional para el análisis.")
    st.caption(f"Entorno Coinbase: {'SANDBOX (api-sandbox.coinbase.com)' if USE_SANDBOX or CB_BASE_URL.startswith('api-sandbox') else 'PRODUCCIÓN'}")

st.write("Introduce el activo y pulsa **Analizar con OpenAI**. La app obtiene KPIs a partir de velas 1H (48h) y calcula tendencias 3/6/12 meses con velas 1D.")

colA, colB = st.columns([1,1])
with colA:
    analyze = st.button("📊 Analizar con OpenAI", use_container_width=True)
with colB:
    st.empty()

if analyze:
    with st.spinner("Calculando contexto de mercado y llamando a OpenAI…"):
        # 1) KPIs 1H (48h)
        df48 = get_hourly_candles_48h(product_id)
        spot = get_spot_price(product_id)
        kpis = {}
        trends_intra = {"short_term":"flat","mid_term":"flat","long_term":"flat"}
        if df48 is not None and len(df48) >= 10:
            kpis = compute_indicators(df48)
            kpis["volume_change_24h_pct"] = compute_volume_change(df48)
            trends_intra = compute_trends(df48)

        # 2) Tendencias 3/6/12 meses (1D)
        df365 = get_daily_candles_days(product_id, 400)
        chg_3m = compute_pct_change(df365, 90) if df365 is not None else None
        chg_6m = compute_pct_change(df365, 180) if df365 is not None else None
        chg_12m = compute_pct_change(df365, 365) if df365 is not None else None

        try:
            signal = ask_openai_for_signal(product_id=product_id, user_notes=user_notes, price=spot, kpis=kpis, trends=trends_intra)
            st.session_state["signal"] = signal.model_dump()
            st.session_state["market"] = {
                "spot": spot,
                "kpis": kpis,
                "trends_intra": trends_intra,
                "chg_3m": chg_3m, "chg_6m": chg_6m, "chg_12m": chg_12m,
            }
        except Exception as e:
            st.error(f"Error al generar la señal: {e}")

if "signal" in st.session_state:
    sig = TradeSignal(**st.session_state["signal"])
    market = st.session_state.get("market", {})

    # Top summary
    st.subheader("Resultado del análisis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Acción", sig.action.upper())
    c2.metric("Confianza", f"{sig.confidence:.2f}")
    c3.metric("Horizonte (días)", sig.time_horizon_days)
    spot = market.get("spot", None)
    c4.metric("Precio actual", f"{spot:.4f}" if isinstance(spot, (int, float)) and spot is not None else "—")

    # Tendencias 3/6/12M
    def trend_label(p):
        if p is None: return "N/D"
        th = 0.5  # % umbral
        return "⬆️ UP" if p > th else ("⬇️ DOWN" if p < -th else "➡️ FLAT")
    t3, t6, t12 = st.columns(3)
    t3.metric("Tendencia 3M", f"{market.get('chg_3m'):.2f}%" if market.get("chg_3m") is not None else "N/D", trend_label(market.get("chg_3m")))
    t6.metric("Tendencia 6M", f"{market.get('chg_6m'):.2f}%" if market.get("chg_6m") is not None else "N/D", trend_label(market.get("chg_6m")))
    t12.metric("Tendencia 12M", f"{market.get('chg_12m'):.2f}%" if market.get("chg_12m") is not None else "N/D", trend_label(market.get("chg_12m")))

    # Panels: análisis + KPIs + ejecución
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### 📋 Análisis detallado")
        with st.expander("Ver análisis", expanded=True):
            st.write(sig.analysis)
        st.info(f"**Rationale:** {sig.rationale}")
        st.write(f"**Producto:** {sig.product_id}")

        st.markdown("#### 📈 KPIs (48h, velas 1H)")
        kcalc = market.get("kpis") or {}
        kpi_rows = [
            ["RSI (0-100)", kcalc.get("rsi")],
            ["MACD (line-signal)", kcalc.get("macd")],
            ["Δ Precio 24h (%)", kcalc.get("price_change_24h_pct")],
            ["Δ Volumen 24h (%)", kcalc.get("volume_change_24h_pct")],
            ["Volatilidad 24h (%)", kcalc.get("volatility_24h_pct")],
        ] if kcalc else [["—", "No disponible"]]
        st.dataframe(pd.DataFrame(kpi_rows, columns=["KPI", "Valor"]), use_container_width=True, hide_index=True)

    with right:
        st.markdown("#### 🧮 % de recomendación (OpenAI)")
        dist = sig.recommendation_distribution
        dist_df = pd.DataFrame({"Recomendación": ["BUY","HOLD","SELL"], "Probabilidad": [dist.buy, dist.hold, dist.sell]}).set_index("Recomendación")
        st.bar_chart(dist_df, height=220, use_container_width=True)
        st.caption("Suma ≈ 1.0")

        if sig.action != "hold":
            st.divider()
            st.warning("⚠️ Esto NO es asesoramiento financiero. Revisa la señal antes de operar.")
            if CB_API_KEY and CB_API_SECRET and (spot is not None):
                if sig.action == "buy":
                    q = st.text_input("Importe a comprar (quote / USD)", value=sig.size if sig.size_type == "quote" else "25")
                    if st.button("✅ Ejecutar BUY en Coinbase (market)", use_container_width=True):
                        try:
                            order = cb_client.market_order_buy(client_order_id=str(uuid.uuid4()), product_id=sig.product_id, quote_size=q)
                            st.success(f"Orden BUY enviada. Respuesta: {order}")
                        except Exception as e:
                            st.error(f"Error al enviar BUY: {e}")
                elif sig.action == "sell":
                    b = st.text_input("Cantidad a vender (base)", value=sig.size if sig.size_type == "base" else "0.001")
                    if st.button("✅ Ejecutar SELL en Coinbase (market)", use_container_width=True):
                        try:
                            order = cb_client.market_order_sell(client_order_id=str(uuid.uuid4()), product_id=sig.product_id, base_size=b)
                            st.success(f"Orden SELL enviada. Respuesta: {order}")
                        except Exception as e:
                            st.error(f"Error al enviar SELL: {e}")
            else:
                st.info("Para ejecutar órdenes necesitas claves de Coinbase en `secrets` y que el producto exista en Coinbase.")
        else:
            st.info("Señal: HOLD. No se propone orden para hoy.")

st.caption("© 2025 — Demo educativa. No es asesoramiento financiero.")
