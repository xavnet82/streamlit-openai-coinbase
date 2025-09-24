
# Nota: este archivo ha sido rehacer para corregir
# - renderizado accidental del docstring en la cabecera (eliminado)
# - estilos del badge (background-color)
# - gráficos sin datos visibles (limpieza NaN, mínimo de velas, ejes con rangebreaks)
# - robustez en métrica de último precio
# - pequeñas incoherencias de estilo y nombres

import os
import json
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator

# OpenAI (SDK v1.x)
try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

# -------------------------------
# Configuración general de página
# -------------------------------
st.set_page_config(
    page_title="Quant Assist — Señales BUY/HOLD/SELL",
    page_icon="📈",
    layout="wide"
)

# Estilos
st.markdown(
    """
    <style>
    .badge {
        display:inline-block; padding:6px 12px; border-radius:12px;
        border:1px solid rgba(0,0,0,0.06); font-weight:700; letter-spacing:.5px;
    }
    .muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Utilidades base
# -------------------------------
def _to_scalar_float(x):
    """Convierte valores (incluyendo Series/ndarray) a float escalar; devuelve None si no es posible."""
    try:
        arr = np.asarray(x)
        if arr.size == 0:
            return None
        val = arr.reshape(-1)[-1]
        try:
            return float(val)
        except Exception:
            val2 = pd.to_numeric(pd.Series([x]), errors='coerce').iloc[0]
            return float(val2) if pd.notna(val2) else None
    except Exception:
        try:
            return float(x)
        except Exception:
            return None

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
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
    }
    return out

def compute_trends(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]
    ma20_val = float(moving_average(close, 20).iloc[-1])
    ma50_val = float(moving_average(close, 50).iloc[-1])
    ma200_val = float(moving_average(close, 200).iloc[-1])
    is_up = (ma20_val > ma50_val) and (ma50_val > ma200_val)
    is_down = (ma20_val < ma50_val) and (ma50_val < ma200_val)
    trend = "up" if is_up else ("down" if is_down else "sideways")
    return {"trend": trend}

@st.cache_data(show_spinner=False, ttl=900)
def get_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError("No hay datos de mercado para el símbolo solicitado.")
    df = df.dropna().copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()]
    df = df.sort_index()
    return df

# -------------------------------
# Modelos de datos (Pydantic)
# -------------------------------
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

# -------------------------------
# OpenAI
# -------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini-2024-07-18")
oa_client = None
if _openai_available and os.getenv("OPENAI_API_KEY"):
    oa_client = OpenAI()

def BUILD_EXPLICIT_SCHEMA() -> Dict[str, Any]:
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
                "properties": {
                    "buy":{"type":"number", "minimum":0},
                    "hold":{"type":"number", "minimum":0},
                    "sell":{"type":"number", "minimum":0}
                },
                "required":["buy","hold","sell"]
            },
            "strategy": {
                "type":"object",
                "additionalProperties": False,
                "properties": {
                    "setup_type":{"type":"string"},
                    "executive_summary":{"type":"string"},
                    "technical_detail":{"type":"string"},
                    "entry_zone":{
                        "anyOf":[
                            {"type":"array", "items":[{"type":["number","null"]},{"type":["number","null"]}], "minItems":2, "maxItems":2},
                            {"type":"null"}
                        ]
                    },
                    "stop_loss":{"type":["number","null"]},
                    "take_profit":{"type":["number","null"]},
                    "risk_reward":{"type":["number","null"]},
                    "timeframe_days":{"type":["integer","null"]},
                    "key_levels":{
                        "type":"array",
                        "items":{"type":"object","properties":{"level":{"type":"number"},"label":{"type":["string","null"]}}, "required":["level"]}
                    }
                },
                "required":["setup_type","executive_summary","technical_detail","entry_zone","stop_loss","take_profit","risk_reward","timeframe_days","key_levels"]
            }
        },
        "required":["symbol","last_price","action","confidence","recommendation_distribution","strategy","kpis","trends"]
    }

def build_prompt(symbol: str, df: pd.DataFrame, kpis: Dict[str,Any], trends: Dict[str,Any]):
    start_date = str(df.index[0].date())
    end_date = str(df.index[-1].date())
    sys = {
        "role": "system",
        "content": (
            "Eres un analista cuantitativo. Devuelve únicamente JSON válido que cumpla el schema suministrado."
        )
    }
    user = {
        "role": "user",
        "content": json.dumps({
            "task": "Generar señal BUY/HOLD/SELL con distribución de probabilidad y plan de trading.",
            "symbol": symbol,
            "data_window": {"start": start_date, "end": end_date},
            "kpis": kpis,
            "trends": trends,
            "requirements": {
                "action": ["buy","hold","sell"],
                "probabilities_sum_to_one": True,
                "sensible_levels": True
            }
        }, ensure_ascii=False)
    }
    return [sys, user]

def ask_openai(symbol: str, df: pd.DataFrame, kpis: dict, trends: dict, deterministic: bool = True) -> TradeSignal:
    if oa_client is None:
        raise RuntimeError("No dispongo de OpenAI API Key configurada en el entorno.")
    schema = BUILD_EXPLICIT_SCHEMA()
    messages = build_prompt(symbol, df, kpis, trends)

    DEFAULT_TEMP = 0.0 if deterministic else 0.4
    OPENAI_SEED = int(os.getenv("OPENAI_SEED", "0")) if deterministic and os.getenv("OPENAI_SEED") else None

    def _one_call():
        kwargs = dict(
            model=OPENAI_MODEL,
            messages=messages,
            response_format={"type":"json_schema","json_schema":{"name":"trade_signal","schema":schema,"strict":True}},
            temperature=DEFAULT_TEMP
        )
        if OPENAI_SEED is not None:
            try:
                kwargs["seed"] = OPENAI_SEED
            except Exception:
                pass
        return oa_client.chat.completions.create(**kwargs)

    last_err = None
    for _ in range(2):
        try:
            completion = _one_call()
            content = completion.choices[0].message.content
            data = json.loads(content)
            ts = TradeSignal.model_validate(data)
            # Guardrails
            dist = ts.recommendation_distribution
            b, h, s = float(dist.buy), float(dist.hold), float(dist.sell)
            tot = b + h + s
            if tot <= 0 or abs(1.0 - tot) > 1e-6:
                b, h, s = max(b,0.0), max(h,0.0), max(s,0.0)
                tot = b + h + s
                if tot == 0: b, h, s = 0.34, 0.33, 0.33; tot = 1.0
                ts.recommendation_distribution.buy  = b/tot
                ts.recommendation_distribution.hold = h/tot
                ts.recommendation_distribution.sell = s/tot
            probs = {"buy": ts.recommendation_distribution.buy,
                     "hold": ts.recommendation_distribution.hold,
                     "sell": ts.recommendation_distribution.sell}
            argmax = max(probs, key=probs.get)
            if ts.action != argmax:
                ts.action = argmax
            return ts
        except Exception as e:
            last_err = e

    # Fallback determinista
    last = float(df["Close"].iloc[-1])
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    ma200 = float(df["Close"].rolling(200).mean().iloc[-1])
    rsi14 = float(kpis.get("rsi14", 50.0))

    def decide():
        if (rsi14 < 35 and last > ma50) or (ma20 > ma50 > ma200):
            return "buy", {"buy":0.6, "hold":0.3, "sell":0.1}
        if (rsi14 > 65 and last < ma50) or (ma20 < ma50 < ma200):
            return "sell", {"buy":0.1, "hold":0.3, "sell":0.6}
        return "hold", {"buy":0.33, "hold":0.34, "sell":0.33}

    action, d = decide()
    ts_like = {
        "symbol": symbol,
        "last_price": last,
        "action": action,
        "confidence": 0.55,
        "rationale": "Fallback determinista por KPIs locales ante error del modelo.",
        "analysis": "Basada en RSI/MA y tendencia simple.",
        "kpis": kpis,
        "trends": trends,
        "recommendation_distribution": d,
        "strategy": {
            "setup_type": "tendencial",
            "executive_summary": "Estrategia simple por KPIs; tamaño de posición conservador.",
            "technical_detail": "RSI y medias móviles; sin volumen ni patrones.",
            "entry_zone": None,
            "stop_loss": None,
            "take_profit": None,
            "risk_reward": None,
            "timeframe_days": 20,
            "key_levels": []
        }
    }
    try:
        return TradeSignal.model_validate(ts_like)
    except Exception as e:
        raise RuntimeError(f"Error al generar la señal con OpenAI (y fallback): {last_err or e}")

# -------------------------------
# UI — Sidebar
# -------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    symbol = st.text_input("Símbolo/Ticker (yfinance):", value="AAPL",
                           help="Ejemplos: AAPL, MSFT, ^GSPC, BTC-USD, ^IXIC, ^IBEX")
    period = st.selectbox("Periodo", options=["6mo","1y","2y","5y","10y"], index=1)
    interval = st.selectbox("Intervalo", options=["1d","1h","1wk"], index=0)
    deterministic = st.toggle("Deterministic mode (temp=0, seed)", value=True,
                              help="Mismas entradas → misma salida (si el modelo soporta seed).")
    run_btn = st.button("Analizar", type="primary", use_container_width=True)

# -------------------------------
# Lógica principal
# -------------------------------
if "signal" not in st.session_state:
    st.session_state["signal"] = None
if "symbol" not in st.session_state:
    st.session_state["symbol"] = None

# Datos
try:
    df = get_data(symbol, period=period, interval=interval)
    kpis = compute_kpis(df)
    trends = compute_trends(df)
except Exception as e:
    st.error(f"No se pudieron cargar datos: {e}")
    st.stop()

# Ejecutar análisis
if run_btn:
    try:
        ts = ask_openai(symbol, df, kpis, trends, deterministic=deterministic)
        st.session_state["signal"] = ts.model_dump()
        st.session_state["symbol"] = symbol
    except Exception as e:
        st.warning(f"Fallo al generar nueva señal: {e}. Se mantiene la última señal válida.")

# Señal actual
raw = st.session_state.get("signal")
current = TradeSignal(**raw) if isinstance(raw, dict) else None

# -------------------------------
# UI — Cabecera / Estado
# -------------------------------
c1,c2,c3 = st.columns([2,3,2])
with c1:
    st.markdown(f"## {symbol}")
    st.caption(f"Datos: {df.index[0].date()} → {df.index[-1].date()} • Último: {df.index[-1].date()}")
with c2:
    if current:
        color = {"buy":"#16a34a","hold":"#f59e0b","sell":"#ef4444"}[current.action]
        st.markdown(f'<span class="badge" style="background-color:{color}22;border-color:{color}33;color:{color}">{current.action.upper()}</span>', unsafe_allow_html=True)
        st.caption(f"Confianza: {round(current.confidence*100,1)}%")
    else:
        st.info("Sin señal todavía. Pulsa **Analizar**.")
with c3:
    last_close_val_raw = df['Close'].iloc[-1]
    _last_close_scalar = _to_scalar_float(last_close_val_raw)
    if (_last_close_scalar is None) or (isinstance(_last_close_scalar, float) and np.isnan(_last_close_scalar)):
        st.metric(label="Precio (último)", value="—")
    else:
        st.metric(label="Precio (último)", value=f"{_last_close_scalar:,.2f}")
st.divider()

# -------------------------------
# Gráfico de precio con medias
# -------------------------------
df_plot = df[["Open","High","Low","Close"]].copy()
df_plot = df_plot.dropna()
if len(df_plot) < 30:
    st.warning("No hay suficientes datos limpios para dibujar el gráfico (mínimo 30 velas).")
else:
    ma20 = df_plot["Close"].rolling(20).mean()
    ma50 = df_plot["Close"].rolling(50).mean()
    ma200 = df_plot["Close"].rolling(200).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_plot.index, open=df_plot["Open"], high=df_plot["High"], low=df_plot["Low"], close=df_plot["Close"],
        name="Precio"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=ma20, name="MA20"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=ma50, name="MA50"))
    fig.add_trace(go.Scatter(x=df_plot.index, y=ma200, name="MA200"))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat","mon"])])  # oculta fines de semana para daily
    fig.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=30), xaxis_title="Fecha", yaxis_title="Precio", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tabs de análisis
# -------------------------------
tab1, tab2, tab3 = st.tabs(["Estrategia (OpenAI)", "Probabilidades", "KPIs"])

with tab1:
    if current:
        s = current.strategy
        st.subheader("Plan de trading")
        left, right = st.columns([2,2])
        with left:
            st.write("**Resumen ejecutivo**")
            st.write(s.executive_summary or "—")
            st.write("**Detalle técnico**")
            with st.expander("Ver detalle técnico"):
                st.write(s.technical_detail or "—")
        with right:
            st.write("**Niveles**")
            entry = s.entry_zone if s.entry_zone else (None, None)
            st.write(f"- Entry zone: {entry[0]} — {entry[1]}")
            st.write(f"- Stop-loss: {s.stop_loss if s.stop_loss is not None else '—'}")
            st.write(f"- Take-profit: {s.take_profit if s.take_profit is not None else '—'}")
            st.write(f"- R:R: {round(s.risk_reward,2) if s.risk_reward is not None else '—'}")
            st.write(f"- Horizonte: {s.timeframe_days or '—'} días")
            if s.key_levels:
                st.write("**Key levels**")
                st.table(pd.DataFrame([{"level": kl.level, "label": kl.label} for kl in s.key_levels]))
        st.write("**Rationale**")
        with st.expander("Ver explicación"):
            st.write(current.rationale or "—")
            if current.analysis:
                st.write(current.analysis)
    else:
        st.info("Sin señal cargada.")

with tab2:
    if current:
        dist = current.recommendation_distribution
        d_df = pd.DataFrame({"Clase":["BUY","HOLD","SELL"], "Probabilidad":[dist.buy, dist.hold, dist.sell]})
        bar = go.Figure()
        bar.add_trace(go.Bar(x=d_df["Clase"], y=d_df["Probabilidad"]))
        bar.update_layout(height=260, yaxis=dict(tickformat=".0%"), margin=dict(l=10,r=10,b=10,t=10))
        st.plotly_chart(bar, use_container_width=True)
        st.dataframe(d_df.assign(Probabilidad=(d_df["Probabilidad"]*100).round(2)))
    else:
        st.info("Sin señal cargada.")

with tab3:
    st.dataframe(pd.DataFrame(kpis, index=["valor"]).T)

st.caption("Aviso: Este software es de apoyo a la decisión y no constituye asesoramiento financiero.")
