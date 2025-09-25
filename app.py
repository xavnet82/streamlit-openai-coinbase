# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
from core.data import get_data
from core.strategy import compute_kpis, compute_trends
from core.openai_client import ask as ask_openai
from core.models import TradeSignal
from core.ui import header, price_chart, probs_tab

st.set_page_config(page_title="Quant Assist — Señales", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .badge {display:inline-block;padding:6px 12px;border-radius:12px;border:1px solid rgba(0,0,0,0.06);font-weight:700;letter-spacing:.5px;}
    .muted { color:#6b7280; }
    .chip { display:inline-flex; align-items:center; gap:.5rem; padding:.25rem .6rem; border-radius:9999px; background:#f3f4f6; border:1px solid #e5e7eb; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    symbol = st.text_input("Símbolo/Ticker (yfinance):", value="AAPL",
                           help="Ejemplos: AAPL, MSFT, ^GSPC, BTC-USD, ^IXIC, ^IBEX")
    period = st.selectbox("Periodo", options=["6mo","1y","2y","5y","10y"], index=1)
    interval = st.selectbox("Intervalo", options=["1d","1h","1wk"], index=0)
    deterministic = st.toggle("Deterministic mode (temp=0, seed)", value=True)
    if not os.getenv("OPENAI_API_KEY"):
        st.info("OpenAI desactivado → se usará fallback local (texto técnico enriquecido).")
    run_btn = st.button("Analizar", type="primary", use_container_width=True)

if "signal" not in st.session_state:
    st.session_state["signal"] = None
if "symbol" not in st.session_state:
    st.session_state["symbol"] = None
if "source" not in st.session_state:
    st.session_state["source"] = None
if "flags" not in st.session_state:
    st.session_state["flags"] = None

try:
    df = get_data(symbol, period=period, interval=interval)
    kpis = compute_kpis(df)
    trends = compute_trends(df)
except Exception as e:
    st.error(f"No se pudieron cargar datos: {e}")
    st.stop()

if run_btn:
    ts, source, flags = ask_openai(symbol, df, kpis, trends, deterministic=deterministic)
    st.session_state["signal"] = ts.model_dump()
    st.session_state["symbol"] = symbol
    st.session_state["source"] = source
    st.session_state["flags"] = flags

if st.session_state.get("signal") is None:
    ts, source, flags = ask_openai(symbol, df, kpis, trends, deterministic=True)
    st.session_state["signal"] = ts.model_dump()
    st.session_state["symbol"] = symbol
    st.session_state["source"] = source
    st.session_state["flags"] = flags

raw = st.session_state.get("signal")
current = TradeSignal(**raw) if isinstance(raw, dict) else None
source = st.session_state.get("source") or "fallback"
flags = st.session_state.get("flags") or {}

header(symbol, df, current, source, flags)
price_chart(df)

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
            with st.expander("Ver detalle técnico", expanded=True if flags.get("text_enriched") else False):
                st.write(s.technical_detail or "—")
        with right:
            st.write("**Niveles**")
            entry = s.entry_zone if s.entry_zone else (None, None)
            st.write(f"- Entry zone: {entry[0]} — {entry[1]}")
            st.write(f"- Stop-loss: {round(s.stop_loss,2) if s.stop_loss is not None else '—'}")
            st.write(f"- Take-profit: {round(s.take_profit,2) if s.take_profit is not None else '—'}")
            st.write(f"- R:R: {round(s.risk_reward,2) if s.risk_reward is not None else '—'}")
            st.write(f"- Horizonte: {s.timeframe_days or '—'} días")
            if s.key_levels:
                import pandas as pd
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
    probs_tab(current)

with tab3:
    st.dataframe(pd.DataFrame(kpis, index=["valor"]).T)

st.caption("Aviso: Este software es de apoyo a la decisión y no constituye asesoramiento financiero.")
