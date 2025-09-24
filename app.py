# app.py — Any Asset/Index Analyzer (1Y daily) + OpenAI strategy — Revamped UX
import os
import json
from typing import Literal, Optional, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel, Field, confloat
import yfinance as yf
from datetime import datetime, timedelta, timezone

# OpenAI SDK
from openai import OpenAI

# Plotly for advanced visuals
import plotly.graph_objects as go
import backtesting

# ---------- Page & Global Styles ----------
st.set_page_config(page_title="Any-Asset Trading Assistant (OpenAI + 1Y Daily)", page_icon="📊", layout="wide")

# Minimal, clean look & feel
st.markdown("""
<style>
:root { --primary: #2B59C3; --ok: #16a34a; --warn:#f59e0b; --err:#ef4444; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1,h2,h3 { letter-spacing: .2px; }
.small { font-size: 0.85rem; color: #6b7280; }
.card {
  border-radius: 16px; padding: 14px 16px; border: 1px solid #eee; background: white;
  box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}
.kpi-card { text-align:center; }
.kpi-title { font-size: 0.9rem; color: #6b7280; margin-bottom: 4px; }
.kpi-value { font-size: 1.2rem; font-weight: 600; }
.badge {
  display:inline-block; padding: 2px 8px; border-radius: 999px; font-size: .8rem; font-weight:600;
  background: #eef2ff; color: #3730a3; border: 1px solid #e0e7ff;
}
hr { border: none; border-top: 1px solid #eee; margin: 8px 0 16px 0;}
</style>
""", unsafe_allow_html=True)

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

class EntryZone(BaseModel):
    model_config = {"extra": "forbid"}
    lower: float
    upper: float

class RecommendationDist(BaseModel):
    model_config = {"extra": "forbid"}
    buy: confloat(ge=0, le=1)
    hold: confloat(ge=0, le=1)
    sell: confloat(ge=0, le=1)

class Strategy(BaseModel):
    model_config = {"extra": "forbid"}
    setup_type: Literal["rebote","rotura_canal","breakout","pullback","rango","tendencia","otro"]
    executive_summary: str
    technical_detail: str
    narrative: str
    entry_zone: EntryZone | None = Field(default=None, description="Rango de entrada: lower/upper")
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_reward: float | None = None
    timeframe_days: int | None = None
    key_levels: List[str] = Field(default_factory=list)

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

# ---------- Explicit Strict Schema for OpenAI ----------
def BUILD_EXPLICIT_SCHEMA():
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "symbol": {"type": "string"},
            "last_price": {"type": "number"},
            "action": {"type": "string", "enum": ["buy","sell","hold"]},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "rationale": {"type": "string"},
            "analysis": {"type": "string"},
            "kpis": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "rsi14": {"type": "number", "minimum": 0, "maximum": 100},
                    "macd_diff_12_26_9": {"type": "number"},
                    "atr14_pct": {"type": "number"},
                    "vola20d_annualized_pct": {"type": "number"},
                    "ma20": {"type": "number"},
                    "ma50": {"type": "number"},
                    "ma200": {"type": "number"},
                    "pct_from_ma50": {"type": "number"},
                    "pct_from_ma200": {"type": "number"},
                    "week52_high": {"type": "number"},
                    "week52_low": {"type": "number"},
                    "pct_from_52w_high": {"type": "number"},
                    "pct_from_52w_low": {"type": "number"}
                },
                "required": ["rsi14","macd_diff_12_26_9","atr14_pct","vola20d_annualized_pct","ma20","ma50","ma200","pct_from_ma50","pct_from_ma200","week52_high","week52_low","pct_from_52w_high","pct_from_52w_low"]
            },
            "trends": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pct_3m": {"type": ["number","null"]},
                    "pct_6m": {"type": ["number","null"]},
                    "pct_12m": {"type": ["number","null"]},
                    "label_3m": {"type": ["string","null"]},
                    "label_6m": {"type": ["string","null"]},
                    "label_12m": {"type": ["string","null"]}
                },
                "required": ["pct_3m","pct_6m","pct_12m","label_3m","label_6m","label_12m"]
            },
            "recommendation_distribution": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "buy": {"type": "number", "minimum": 0, "maximum": 1},
                    "hold": {"type": "number", "minimum": 0, "maximum": 1},
                    "sell": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["buy","hold","sell"]
            },
            "strategy": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "setup_type": {"type": "string", "enum": ["rebote","rotura_canal","breakout","pullback","rango","tendencia","otro"]},
                    "executive_summary": {"type": "string"},
                    "technical_detail": {"type": "string"},
                    "narrative": {"type": "string"},
                    "entry_zone": {
                        "anyOf": [
                            {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": { "lower": {"type": "number"}, "upper": {"type": "number"} },
                                "required": ["lower","upper"]
                            },
                            {"type": "null"}
                        ]
                    },
                    "stop_loss": {"type": ["number","null"]},
                    "take_profit": {"type": ["number","null"]},
                    "risk_reward": {"type": ["number","null"]},
                    "timeframe_days": {"type": ["integer","null"]},
                    "key_levels": { "type": "array", "items": {"type": "string"} }
                },
                "required": ["setup_type","executive_summary","technical_detail","narrative","entry_zone","stop_loss","take_profit","risk_reward","timeframe_days","key_levels"]
            }
        },
        "required": ["symbol","last_price","action","confidence","rationale","analysis","kpis","trends","recommendation_distribution","strategy"]
    }

# ---------- Data utils ----------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_1y_daily(symbol: str) -> pd.DataFrame | None:
    try:
        df = yf.download(symbol, period="1y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty and {"Open","High","Low","Close","Volume"}.issubset(df.columns):
            df = df.rename(columns=str.title).dropna(subset=["Close"])
            return df
    except Exception:
        pass
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=365)
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=False)
        if isinstance(df, pd.DataFrame) and not df.empty and {"Open","High","Low","Close","Volume"}.issubset(df.columns):
            df = df.rename(columns=str.title).dropna(subset=["Close"])
            return df
    except Exception:
        pass
    return None

def compute_kpis(df: pd.DataFrame) -> dict:
    close = df["Close"].astype(float); high = df["High"].astype(float); low  = df["Low"].astype(float)
    # MAs
    ma20  = close.rolling(20).mean(); ma50  = close.rolling(50).mean(); ma200 = close.rolling(200).mean()
    # RSI 14 (Wilder's)
    delta = close.diff(); gain = delta.clip(lower=0.0); loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean(); avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12)); rsi = 100 - (100 / (1 + rs)); rsi14 = float(rsi.iloc[-1])
    # MACD 12-26-9 diff
    ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26; signal = macd_line.ewm(span=9, adjust=False).mean(); macd_diff = float((macd_line - signal).iloc[-1])
    # ATR14%
    tr = pd.concat([(high - low),(high - close.shift()).abs(),(low  - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean().iloc[-1]; atr14_pct = float(atr14 / close.iloc[-1] * 100.0) if close.iloc[-1] > 0 else 0.0
    # Realized volatility 20d annualized (pct)
    ret = close.pct_change(); vol20 = ret.rolling(20).std().iloc[-1] * np.sqrt(252) * 100.0
    # 52w stats
    wk52_high = float(close.max()); wk52_low  = float(close.min()); last = float(close.iloc[-1])
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
    t3  = pct_change_lookback(63)
    t6  = pct_change_lookback(126)
    t12 = pct_change_lookback(252)
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
        "Devuelve JSON que cumpla el esquema estricto. **No inventes datos**: usa los KPIs y el último precio del contexto. "
        "La estrategia debe tener dos secciones claras:\n"
        "1) executive_summary: 3-5 frases accionables con qué hacer ahora, niveles (entrada/stop/TP), alternativas si falla el escenario y R/R.\n"
        "2) technical_detail: explicación técnica con soportes/resistencias, patrones, divergencias, flujo de volumen, y fundamentos técnicos de la decisión.\n"
        "Incluye además: setup_type, entry_zone (rango lower/upper o null), stop_loss, take_profit, risk_reward, timeframe_days, key_levels.\n"
        f"\n\nmarket_context = {json.dumps(context, ensure_ascii=False)}"
    )
    return [{"role":"system","content":system},{"role":"user","content":user}]

def ask_openai(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict) -> TradeSignal:
    messages = build_prompt(symbol, df, kpis, trends)
    schema = BUILD_EXPLICIT_SCHEMA()
    completion = oa_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        messages=messages,
        response_format={"type":"json_schema","json_schema":{"name":"trade_signal","schema":schema,"strict":True}},
        temperature=1,  # GPT-5: usar 1
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
    st.caption("Fuente de mercado: Yahoo Finance via yfinance.")

st.write("La app descarga **1 año de datos diarios**, calcula KPIs y tendencias, y solicita a OpenAI una **recomendación y estrategia** (con parte ejecutiva + detalle técnico).")

cols = st.columns([1,1])
with cols[0]:
    analyze = st.button("📊 Analizar activo", use_container_width=True)
with cols[1]:
    st.empty()

if analyze:
    with st.spinner("Descargando 1Y diario de yfinance…"):
        df = fetch_1y_daily(symbol)
        if df is None or df.empty:
            st.error("No se pudieron obtener datos para el símbolo indicado. Verifica el ticker.")
        else:
            st.session_state["df"] = df
            st.session_state["symbol"] = symbol
            st.session_state["kpis"] = compute_kpis(df)
            st.session_state["trends"] = compute_trends(df)
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

    # HERO DASHBOARD
    last_price = float(df["Close"].iloc[-1])
    day_change_pct = float((df["Close"].iloc[-1]/df["Close"].iloc[-2]-1.0)*100.0) if len(df)>=2 else 0.0

    st.markdown("### 🔎 Resumen ejecutivo")
    with st.container():
        c1, c2, c3, c4 = st.columns([1.2,1,1,1])
        if ts:
            parsed = TradeSignal(**ts) if isinstance(ts, dict) else ts
            dist = parsed.recommendation_distribution
            score = (dist.buy - dist.sell) * 100.0  # -100 SELL … +100 BUY
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score,
                gauge = {
                    "axis": {"range": [-100,100]},
                    "bar": {"color": "#2B59C3"},
                    "steps": [
                        {"range": [-100,-33], "color":"#fee2e2"},
                        {"range": [-33,33], "color":"#fef9c3"},
                        {"range": [33,100], "color":"#dcfce7"},
                    ]
                },
                title = {"text": f"Recomendación • {parsed.action.upper()}"},
                number = {"suffix": ""},
                delta = {"reference": 0}
            ))
            fig.update_layout(height=260, margin=dict(l=10,r=10,t=30,b=10))
            c1.plotly_chart(fig, use_container_width=True)
        else:
            c1.info("Ejecuta el análisis para ver la recomendación.")

        with c2:
            st.markdown('<div class="card kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-title">Confianza</div>', unsafe_allow_html=True)
            if ts:
                conf = parsed.confidence
                st.progress(min(max(conf,0.0),1.0), text=f"{conf*100:.1f}%")
            else:
                st.progress(0.0, text="—")
            st.markdown('<hr/>', unsafe_allow_html=True)
            st.metric("Precio actual", f"{last_price:,.4f}", f"{day_change_pct:+.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with c3:
            st.markdown('<div class="card kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-title">Entrada / Stop</div>', unsafe_allow_html=True)
            if ts and parsed.strategy.entry_zone:
                ez = parsed.strategy.entry_zone
                st.metric("Entry Zone", f"{ez.lower:,.3f} - {ez.upper:,.3f}")
            else:
                st.metric("Entry Zone", "—")
            st.metric("Stop-Loss", f"{parsed.strategy.stop_loss:,.3f}" if ts and parsed.strategy.stop_loss is not None else "—")
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="card kpi-card">', unsafe_allow_html=True)
            st.markdown('<div class="kpi-title">Objetivos</div>', unsafe_allow_html=True)
            st.metric("Take-Profit", f"{parsed.strategy.take_profit:,.3f}" if ts and parsed.strategy.take_profit is not None else "—")
            st.metric("Riesgo/Beneficio", f"{parsed.strategy.risk_reward:.2f}" if ts and parsed.strategy.risk_reward is not None else "—")
            st.markdown('</div>', unsafe_allow_html=True)

    # Quick KPI cards
    st.markdown("### 📈 KPIs clave")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("RSI14", kpis["rsi14"])
    k2.metric("MACD diff", kpis["macd_diff_12_26_9"])
    k3.metric("ATR14 %", kpis["atr14_pct"])
    k4.metric("Vol 20d %", kpis["vola20d_annualized_pct"])
    k5.metric("% vs MA50", kpis["pct_from_ma50"])
    k6.metric("% vs MA200", kpis["pct_from_ma200"])

    # Tabs
    tab1, tab2, tab_bt, tab3, tab4 = st.tabs(["📈 Gráfico", "🧭 Estrategia (OpenAI)", "🔁 Backtesting", "🧮 KPIs & Tendencias", "📄 Datos"])

    with tab1:
        try:
            fig = go.Figure(data=[
                go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="OHLC")
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
        if not ts:
            st.info("Aún no se ha generado análisis con OpenAI.")
        else:
            parsed = TradeSignal(**ts) if isinstance(ts, dict) else ts
            st.subheader(f"Recomendación: {parsed.action.upper()}  •  Confianza: {parsed.confidence:.2f}")
            dist = parsed.recommendation_distribution
            dist_df = pd.DataFrame({"label":["BUY","HOLD","SELL"], "p":[dist.buy, dist.hold, dist.sell]})
            bar = go.Figure(go.Bar(x=dist_df["label"], y=dist_df["p"]))
            bar.update_layout(height=240, yaxis=dict(range=[0,1]), margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(bar, use_container_width=True)

            st.markdown("#### 🧭 Executive summary")
            st.write(parsed.strategy.executive_summary)

            st.markdown("#### 🧪 Detalle técnico")
            st.write(parsed.strategy.technical_detail)

            st.markdown("#### 📌 Niveles & Plan")
            cols = st.columns(4)
            if parsed.strategy.entry_zone:
                cols[0].metric("Entry Zone", f"{parsed.strategy.entry_zone.lower:,.3f}–{parsed.strategy.entry_zone.upper:,.3f}")
            else:
                cols[0].metric("Entry Zone", "—")
            cols[1].metric("Stop-Loss", f"{parsed.strategy.stop_loss:,.3f}" if parsed.strategy.stop_loss is not None else "—")
            cols[2].metric("Take-Profit", f"{parsed.strategy.take_profit:,.3f}" if parsed.strategy.take_profit is not None else "—")
            cols[3].metric("Riesgo/Beneficio", f"{parsed.strategy.risk_reward:.2f}" if parsed.strategy.risk_reward is not None else "—")
            st.caption(f"Timeframe (días): {parsed.strategy.timeframe_days if parsed.strategy.timeframe_days is not None else '—'}")
            if parsed.strategy.key_levels:
                st.markdown("**Niveles clave**: " + ", ".join(parsed.strategy.key_levels))

            st.markdown("---")
            st.markdown("#### Análisis detallado")
            st.write(parsed.analysis)
            st.info(f"**Rationale:** {parsed.rationale}")
   
    with tab_bt:
      st.subheader("Simulador de estrategias (Backtesting)")
      if "df" not in st.session_state:
          st.info("Primero ejecuta un análisis para cargar datos del activo.")
      else:
          df_bt = df.copy()
          # Selector de rango temporal
          min_date = df_bt.index.min()
          max_date = df_bt.index.max()
          r = st.slider("Rango temporal",
                        min_value=min_date.to_pydatetime(),
                        max_value=max_date.to_pydatetime(),
                        value=(min_date.to_pydatetime(), max_date.to_pydatetime()))
          start_d, end_d = pd.Timestamp(r[0]), pd.Timestamp(r[1])
          df_bt = df_bt.loc[(df_bt.index >= start_d) & (df_bt.index <= end_d)].copy()
          if df_bt.empty:
              st.warning("No hay datos en el rango seleccionado.")
          else:
              # Parámetros de reglas
              st.markdown("**Reglas de señal (KPI):**")
              c1, c2, c3 = st.columns(3)
              with c1:
                  use_rsi = st.checkbox("Usar RSI", value=True)
                  rsi_buy = st.number_input("RSI BUY <", value=30.0, min_value=1.0, max_value=50.0, step=0.5)
                  rsi_sell = st.number_input("RSI SELL >", value=70.0, min_value=50.0, max_value=99.0, step=0.5)
              with c2:
                  use_ma_cross = st.checkbox("Cruce MAs", value=True)
                  fast_ma = st.number_input("MA rápida", value=20, min_value=2, max_value=100, step=1)
                  slow_ma = st.number_input("MA lenta", value=50, min_value=5, max_value=400, step=1)
              with c3:
                  sl_pct = st.number_input("Stop-Loss (%)", value=3.0, min_value=0.1, max_value=50.0, step=0.1) / 100.0
                  tp_pct = st.number_input("Take-Profit (%)", value=6.0, min_value=0.1, max_value=200.0, step=0.1) / 100.0
                  fee_bps = st.number_input("Comisión (bps)", value=5, min_value=0, max_value=100, step=1)
  
              st.markdown("---")
              oa_vet = st.checkbox("Validar/ajustar señales con OpenAI (puede consumir créditos)", value=False)
              run_bt = st.button("▶️ Ejecutar backtest", use_container_width=True)
  
              if run_bt:
                  # Generar señales base
                  sigs = backtesting.generate_signals(df_bt, use_rsi, rsi_buy, rsi_sell, use_ma_cross, int(fast_ma), int(slow_ma))
                  st.write(f"Señales generadas: {len(sigs)}")
  
                  decisions = None
                  if oa_vet:
                      try:
                          decisions = backtesting.openai_vet_signals(
                              oa_client, os.getenv("OPENAI_MODEL", OPENAI_MODEL), symbol, df_bt, sigs
                          )
                      except Exception as e:
                          st.warning(f"No se pudo consultar OpenAI para validar señales: {e}")
  
                  trades_df, eq_df = backtesting.simulate(
                      df_bt, sigs, sl_pct, tp_pct, capital=10000.0, fee_pct=fee_bps/10000.0,
                      use_openai_decisions=decisions
                  )
                  summary = backtesting.summarize(trades_df, eq_df)
  
                  # Resultados
                  m1, m2, m3, m4, m5, m6 = st.columns(6)
                  m1.metric("Trades", summary["trades"])
                  m2.metric("Win rate", f'{summary["win_rate"]*100:.1f}%')
                  m3.metric("Avg win", f'{summary["avg_win"]*100:.2f}%')
                  m4.metric("Avg loss", f'{summary["avg_loss"]*100:.2f}%')
                  m5.metric("Total return", f'{summary["total_return"]*100:.2f}%')
                  m6.metric("Max DD", f'{summary["max_drawdown"]*100:.2f}%')
                  st.caption(f"Sharpe aprox.: {summary['sharpe']:.2f}")
  
                  # Curva de equity
                  import plotly.graph_objects as go
                  if not eq_df.empty:
                      eq_fig = go.Figure(go.Scatter(x=eq_df['date'], y=eq_df['equity'], mode='lines', name='Equity'))
                      eq_fig.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10))
                      st.plotly_chart(eq_fig, use_container_width=True)
  
                  # Detalle de trades
                  if not trades_df.empty:
                      st.markdown("#### Detalle de operaciones")
                      tdf = trades_df.copy()
                      tdf["pnl_pct"] = (tdf["pnl_pct"]*100).round(2)
                      st.dataframe(tdf, use_container_width=True)
                  else:
                      st.info("No se generaron operaciones para las reglas y el rango seleccionados.")

    with tab3:
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

        t1,t2,t3 = st.columns(3)
        def labicon(lbl):
            return "⬆️ UP" if lbl=="up" else ("⬇️ DOWN" if lbl=="down" else "➡️ FLAT")
        t1.metric("Tendencia 3M", f"{trends['pct_3m']:+.2f}%" if trends["pct_3m"] is not None else "N/D", labicon(trends["label_3m"]) if trends["label_3m"] else "N/D")
        t2.metric("Tendencia 6M", f"{trends['pct_6m']:+.2f}%" if trends["pct_6m"] is not None else "N/D", labicon(trends["label_6m"]) if trends["label_6m"] else "N/D")
        t3.metric("Tendencia 12M", f"{trends['pct_12m']:+.2f}%" if trends["pct_12m"] is not None else "N/D", labicon(trends["label_12m"]) if trends["label_12m"] else "N/D")

    with tab4:
        st.markdown("#### Datos descargados (1Y, 1D)")
        st.dataframe(df[["Open","High","Low","Close","Volume"]], use_container_width=True)
        csv = df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Descargar CSV", data=csv, file_name=f"{symbol}_1y_daily.csv", mime="text/csv")

st.caption("© 2025 — Trabajo educativo. No es asesoramiento financiero. Fuente de mercado: Yahoo Finance (yfinance).")
