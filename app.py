# app.py — Any Asset/Index Analyzer (1Y daily) + OpenAI strategy (no brokerage links)
import os
import json
from typing import Literal, Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, ValidationError, confloat, conlist
import yfinance as yf
from datetime import datetime, timedelta, timezone

# OpenAI SDK
from openai import OpenAI

# ---------- Page ----------
st.set_page_config(page_title="Any-Asset Trading Assistant (OpenAI + 1Y Daily)", page_icon="📊", layout="wide")
st.title("📊 Any-Asset Trading Assistant — 1Y Daily + OpenAI (sin ejecución)")

# ---------- Secrets ----------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
if not OPENAI_API_KEY:
    st.error("Falta OPENAI_API_KEY en .streamlit/secrets.toml o variable de entorno.")
    st.stop()
oa_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Models (STRICT) ----------
Action = Literal["buy", "sell", "hold"]

class KPIs(BaseModel):
    model_config = {"extra": "forbid"}
    rsi14: confloat(ge=0, le=100)
    macd_diff_12_26_9: float
    atr14_pct: float
    vola20d_annualized_pct: float
    ma20: float
    ma50: float
    ma200: float
    pct_from_ma50: float
    pct_from_ma200: float
    week52_high: float
    week52_low: float
    pct_from_52w_high: float
    pct_from_52w_low: float

class Trends(BaseModel):
    model_config = {"extra": "forbid"}
    pct_3m: float | None
    pct_6m: float | None
    pct_12m: float | None
    label_3m: Literal["up","down","flat"] | None
    label_6m: Literal["up","down","flat"] | None
    label_12m: Literal["up","down","flat"] | None

class RecommendationDist(BaseModel):
    model_config = {"extra": "forbid"}
    buy: confloat(ge=0, le=1)
    hold: confloat(ge=0, le=1)
    sell: confloat(ge=0, le=1)

class Strategy(BaseModel):
    model_config = {"extra": "forbid"}
    setup_type: Literal["rebote","rotura_canal","breakout","pullback","rango","tendencia","otro"]
    narrative: str
    entry_zone: Dict[str, float] | None = Field(default=None, description="e.g. {'lower': x, 'upper': y}")
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward: float | None = None
    timeframe_days: int | None = None
    key_levels: conlist(str, min_length=0) = []

class TradeSignal(BaseModel):
    model_config = {"extra": "forbid"}
    symbol: str
    last_price: float
    action: Action
    confidence: confloat(ge=0, le=1)
    rationale: str
    analysis: str
    kpis: KPIs
    trends: Trends
    recommendation_distribution: RecommendationDist
    strategy: Strategy

def _no_extra(schema: dict):
    """Recursively set additionalProperties: false for all objects."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema["additionalProperties"] = False
            for v in schema.get("properties", {}).values():
                _no_extra(v)
        for v in schema.values():
            _no_extra(v)
    elif isinstance(schema, list):
        for x in schema:
            _no_extra(x)

# ---------- Data utils ----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_1y_daily(symbol: str) -> pd.DataFrame | None:
    # Try yfinance .download first
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty and {"Open","High","Low","Close","Volume"}.issubset(df.columns):
            df = df.rename(columns=str.title)
            df = df.dropna(subset=["Close"])
            return df
    except Exception:
        pass
    # Fallback via Ticker.history with explicit dates
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty and {"Open","High","Low","Close","Volume"}.issubset(df.columns):
            df = df.rename(columns=str.title)
            df = df.dropna(subset=["Close"])
            return df
    except Exception:
        pass
    return None

def compute_kpis(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low  = df["Low"].astype(float)
    vol  = df["Volume"].astype(float)

    # MAs
    ma20  = close.rolling(20).mean()
    ma50  = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # RSI 14 (Wilder's)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    rsi = 100 - (100 / (1 + rs))
    rsi14 = float(rsi.iloc[-1])

    # MACD 12-26-9 diff
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_diff = float((macd_line - signal).iloc[-1])

    # ATR14
    tr = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]
    atr14_pct = float(atr14 / close.iloc[-1] * 100.0) if close.iloc[-1] > 0 else 0.0

    # Realized volatility 20d annualized (pct)
    ret = close.pct_change()
    vol20 = ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100.0

    # 52w stats
    wk52_high = float(close.max())
    wk52_low  = float(close.min())
    last = float(close.iloc[-1])
    pct_from_ma50  = float((last / ma50.iloc[-1] - 1.0) * 100.0) if not np.isnan(ma50.iloc[-1]) else np.nan
    pct_from_ma200 = float((last / ma200.iloc[-1] - 1.0) * 100.0) if not np.isnan(ma200.iloc[-1]) else np.nan
    pct_from_high  = float((last / wk52_high - 1.0) * 100.0) if wk52_high > 0 else 0.0
    pct_from_low   = float((last / wk52_low - 1.0) * 100.0) if wk52_low  > 0 else 0.0

    return {
        "rsi14": round(rsi14, 2),
        "macd_diff_12_26_9": round(macd_diff, 6),
        "atr14_pct": round(float(atr14_pct), 4),
        "vola20d_annualized_pct": round(float(vol20), 4) if pd.notna(vol20) else np.nan,
        "ma20": round(float(ma20.iloc[-1]), 6) if pd.notna(ma20.iloc[-1]) else np.nan,
        "ma50": round(float(ma50.iloc[-1]), 6) if pd.notna(ma50.iloc[-1]) else np.nan,
        "ma200": round(float(ma200.iloc[-1]), 6) if pd.notna(ma200.iloc[-1]) else np.nan,
        "pct_from_ma50": round(pct_from_ma50, 4) if pd.notna(pct_from_ma50) else np.nan,
        "pct_from_ma200": round(pct_from_ma200, 4) if pd.notna(pct_from_ma200) else np.nan,
        "week52_high": round(wk52_high, 6),
        "week52_low": round(wk52_low, 6),
        "pct_from_52w_high": round(pct_from_high, 4),
        "pct_from_52w_low": round(pct_from_low, 4),
    }

def compute_trends(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float).reset_index(drop=True)
    def pct_change_lookback(days):
        if len(close) <= days: return None
        return float((close.iloc[-1] / close.iloc[-days-1] - 1.0) * 100.0)
    t3  = pct_change_lookback(63)   # ~3m
    t6  = pct_change_lookback(126)  # ~6m
    t12 = pct_change_lookback(252)  # ~12m
    def lab(x):
        if x is None: return None
        if x > 1.0: return "up"
        if x < -1.0: return "down"
        return "flat"
    return {
        "pct_3m": None if t3 is None else round(t3, 2),
        "pct_6m": None if t6 is None else round(t6, 2),
        "pct_12m": None if t12 is None else round(t12, 2),
        "label_3m": lab(t3),
        "label_6m": lab(t6),
        "label_12m": lab(t12),
    }

# ---------- Prompt ----------
def build_prompt(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict) -> list:
    last_price = float(df["Close"].iloc[-1])
    context = {
        "symbol": symbol,
        "last_price": last_price,
        "kpis": kpis,
        "trends": trends,
        "meta": {
            "rows": int(len(df)),
            "start_date": str(df.index[0].date() if hasattr(df.index[0], "date") else df.index[0]),
            "end_date": str(df.index[-1].date() if hasattr(df.index[-1], "date") else df.index[-1])
        }
    }
    system = (
        "Eres un analista cuantitativo y técnico. Con el contexto dado (1 año diario), "
        "devuelve una recomendación SIMPLE (buy/sell/hold) y una **estrategia operativa concreta**. "
        "Tu análisis debe cubrir: tendencia, momentum, volatilidad, soportes/resistencias, "
        "volumen relativo, estructuras (canales/rangos/triángulos) y **sentimiento de mercado** inferido del precio y volumen. "
        "Si no hay ventaja clara, devuelve HOLD."
    )
    user = (
        "CON TEXTO CLARO y SIN inventar datos. Usa exactamente los KPIs y tendencias del contexto; "
        "no supongas precios que no estén en kpis/last_price. Devuelve JSON que cumpla el esquema estricto. "
        "Incluye estrategia con: tipo de setup (rebote, rotura de canal, breakout, pullback, rango, tendencia, u 'otro'), "
        "zona de entrada (rango), stop, take-profit, ratio riesgo/beneficio, timeframe y niveles clave. "
        f"\n\nmarket_context = {json.dumps(context, ensure_ascii=False)}"
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]

def ask_openai(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict) -> TradeSignal:
    messages = build_prompt(symbol, df, kpis, trends)
    schema = TradeSignal.model_json_schema()
    _no_extra(schema)
    completion = oa_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        messages=messages,
        response_format={"type":"json_schema","json_schema":{"name":"trade_signal","schema":schema,"strict":True}},
        temperature=1,
    )
    content = completion.choices[0].message.content
    data = json.loads(content)
    ts = TradeSignal.model_validate(data)
    # Normalize probs
    dist = ts.recommendation_distribution
    s = float(dist.buy)+float(dist.hold)+float(dist.sell)
    if s > 0 and abs(1-s) > 1e-6:
        dist.buy  = float(dist.buy)/s
        dist.hold = float(dist.hold)/s
        dist.sell = float(dist.sell)/s
    return ts

# ---------- UI ----------
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    symbol = st.text_input("Símbolo/Ticker (yfinance):", value="AAPL", help="Ejemplos: AAPL, MSFT, ^GSPC, BTC-USD, ^IXIC, ^IBEX")
    st.caption("Nota: la fuente de mercado es yfinance (Yahoo). Los símbolos deben ser compatibles con Yahoo Finance.")

st.write("La app descarga **1 año de datos diarios** del activo/índice, calcula KPIs y tendencias, y solicita a OpenAI una **recomendación y estrategia**.")

cols = st.columns([1,1])
with cols[0]:
    analyze = st.button("📊 Analizar activo", use_container_width=True)
with cols[1]:
    st.empty()

if analyze:
    with st.spinner("Descargando 1Y diario de yfinance…"):
        df = fetch_1y_daily(symbol)
        if df is None or df.empty:
            st.error("No se pudieron obtener datos para el símbolo indicado. Verifica el ticker (Yahoo Finance).")
        else:
            st.session_state["df"] = df
            st.session_state["symbol"] = symbol
            st.session_state["kpis"] = compute_kpis(df)
            st.session_state["trends"] = compute_trends(df)

            # Llamada OpenAI
            try:
                ts = ask_openai(symbol, df, st.session_state["kpis"], st.session_state["trends"])
                st.session_state["signal"] = ts.model_dump()
            except Exception as e:
                st.error(f"Error al generar la señal con OpenAI: {e}")

if "df" in st.session_state:
    df = st.session_state["df"]
    symbol = st.session_state["symbol"]
    kpis = st.session_state["kpis"]
    trends = st.session_state["trends"]
    ts = st.session_state.get("signal")

    # Top metrics
    last_price = float(df["Close"].iloc[-1])
    day_change_pct = float((df["Close"].iloc[-1]/df["Close"].iloc[-2]-1.0)*100.0) if len(df)>=2 else 0.0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Símbolo", symbol)
    c2.metric("Precio actual", f"{last_price:,.4f}")
    c3.metric("Variación 1D", f"{day_change_pct:+.2f}%")
    if trends["pct_12m"] is not None:
        c4.metric("Tendencia 12M", f"{trends['pct_12m']:+.2f}%", ("⬆️" if trends["label_12m"]=="up" else "⬇️" if trends["label_12m"]=="down" else "➡️"))

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Gráfico", "🧮 KPIs", "🧭 Estrategia (OpenAI)", "📄 Datos"])

    with tab1:
        # Plotly candlestick + MAs
        try:
            import plotly.graph_objects as go
            fig = go.Figure(data=[
                go.Candlestick(
                    x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
                    name="OHLC"
                )
            ])
            ma20 = df["Close"].rolling(20).mean()
            ma50 = df["Close"].rolling(50).mean()
            ma200 = df["Close"].rolling(200).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ma20, mode="lines", name="MA20"))
            fig.add_trace(go.Scatter(x=df.index, y=ma50, mode="lines", name="MA50"))
            fig.add_trace(go.Scatter(x=df.index, y=ma200, mode="lines", name="MA200"))
            fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"No se pudo renderizar el gráfico avanzado (plotly). Detalle: {e}")
            st.line_chart(df["Close"])

    with tab2:
        # KPIs table
        kpi_rows = [
            ("RSI 14", kpis["rsi14"]),
            ("MACD diff (12-26-9)", kpis["macd_diff_12_26_9"]),
            ("ATR14 (%)", kpis["atr14_pct"]),
            ("Vol. realizada 20d anualizada (%)", kpis["vola20d_annualized_pct"]),
            ("MA20", kpis["ma20"]),
            ("MA50", kpis["ma50"]),
            ("MA200", kpis["ma200"]),
            ("% desde MA50", kpis["pct_from_ma50"]),
            ("% desde MA200", kpis["pct_from_ma200"]),
            ("Máximo 52w", kpis["week52_high"]),
            ("Mínimo 52w", kpis["week52_low"]),
            ("% desde 52w high", kpis["pct_from_52w_high"]),
            ("% sobre 52w low", kpis["pct_from_52w_low"]),
        ]
        st.dataframe(pd.DataFrame(kpi_rows, columns=["KPI","Valor"]), use_container_width=True, hide_index=True)

        # Trends
        t1,t2,t3 = st.columns(3)
        def labicon(lbl):
            return "⬆️ UP" if lbl=="up" else ("⬇️ DOWN" if lbl=="down" else "➡️ FLAT")
        t1.metric("Tendencia 3M", f"{trends['pct_3m']:+.2f}%" if trends["pct_3m"] is not None else "N/D", labicon(trends["label_3m"]) if trends["label_3m"] else "N/D")
        t2.metric("Tendencia 6M", f"{trends['pct_6m']:+.2f}%" if trends["pct_6m"] is not None else "N/D", labicon(trends["label_6m"]) if trends["label_6m"] else "N/D")
        t3.metric("Tendencia 12M", f"{trends['pct_12m']:+.2f}%" if trends["pct_12m"] is not None else "N/D", labicon(trends["label_12m"]) if trends["label_12m"] else "N/D")

    with tab3:
        if ts is None:
            st.info("Aún no se ha generado análisis con OpenAI.")
        else:
            parsed = TradeSignal(**ts) if isinstance(ts, dict) else ts
            c1,c2,c3 = st.columns(3)
            c1.metric("Recomendación", parsed.action.upper())
            c2.metric("Confianza", f"{parsed.confidence:.2f}")
            dist = parsed.recommendation_distribution
            c3.metric("Distribución", f"B:{dist.buy:.2f} / H:{dist.hold:.2f} / S:{dist.sell:.2f}")

            st.subheader("Estrategia propuesta")
            strat = parsed.strategy
            st.write(f"**Setup**: {strat.setup_type}")
            st.write(f"**Narrativa**: {strat.narrative}")
            if strat.entry_zone:
                st.write(f"**Zona de entrada**: {strat.entry_zone}")
            st.write(f"**Stop-Loss**: {str(strat.stop_loss) if strat.stop_loss is not None else '—'}")
            st.write(f"**Take-Profit**: {str(strat.take_profit) if strat.take_profit is not None else '—'}")
            st.write(f"**R/R**: {str(strat.risk_reward) if strat.risk_reward is not None else '—'}")
            st.write(f"**Timeframe (días)**: {str(strat.timeframe_days) if strat.timeframe_days is not None else '—'}")
            if strat.key_levels:
                st.write("**Niveles clave**:")
                for lvl in strat.key_levels:
                    st.write(f"- {lvl}")
            st.markdown("---")
            st.markdown("#### Análisis detallado")
            st.write(parsed.analysis)
            st.info(f"**Rationale:** {parsed.rationale}")

    with tab4:
        st.markdown("#### Datos descargados (1Y, 1D)")
        st.dataframe(df[["Open","High","Low","Close","Volume"]], use_container_width=True)
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name=f"{symbol}_1y_daily.csv", mime="text/csv")

st.caption("© 2025 — Trabajo educativo. No es asesoramiento financiero. Fuente de mercado: Yahoo Finance (yfinance).")
