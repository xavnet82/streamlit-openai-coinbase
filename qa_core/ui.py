# qa_core/ui.py
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def header(symbol: str, df: pd.DataFrame, current, source: str, flags: dict):
    c1,c2,c3,c4 = st.columns([2,2,2,2])
    with c1:
        st.markdown(f"## {symbol}")
        try:
            st.caption(f"Datos: {df.index[0].date()} → {df.index[-1].date()} • Último: {df.index[-1].date()}")
        except Exception:
            pass
    with c2:
        if current:
            color = {"buy":"#16a34a","hold":"#f59e0b","sell":"#ef4444"}.get(current.action, "#6b7280")
            st.markdown(
                f'<span class="badge" style="background-color:{color}22;border-color:{color}33;color:{color}">'
                f'{current.action.upper()}</span>', unsafe_allow_html=True
            )
            try:
                st.caption(f"Confianza: {round(float(current.confidence)*100,1)}%")
            except Exception:
                pass
    with c3:
        try:
            last_close = float(pd.to_numeric(df["Close"].iloc[-1], errors="coerce"))
            st.metric(label="Precio (último)", value=f"{last_close:,.2f}")
        except Exception:
            st.metric(label="Precio (último)", value="—")
    with c4:
        src_txt = "OpenAI" if source == "openai" else "Fallback (local)"
        extra = []
        if flags.get("text_enriched"): extra.append("texto enriquecido")
        if flags.get("levels_completed"): extra.append("niveles completados")
        hint = (" • " + ", ".join(extra)) if extra else ""
        st.markdown(f'<span class="chip">Fuente estrategia: <b>{src_txt}</b>{hint}</span>', unsafe_allow_html=True)
    st.divider()

def price_chart(df: pd.DataFrame):
    df_plot = df[["Open","High","Low","Close"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(df_plot) < 30:
        st.warning("No hay suficientes datos limpios para dibujar el gráfico (mínimo 30 velas).")
        return
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
    fig.update_layout(height=420, margin=dict(l=10,r=10,b=10,t=30),
                      xaxis_title="Fecha", yaxis_title="Precio", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def _temperature_from_probs(buy: float, hold: float, sell: float) -> float:
    # 0 = SELL, 50 = HOLD, 100 = BUY
    import numpy as np
    t = (buy - sell + 1.0) / 2.0
    return float(np.clip(t*100.0, 0, 100))

def probs_tab(current):
    if not current:
        st.info("Sin señal cargada.")
        return
    dist = current.recommendation_distribution
    temp = _temperature_from_probs(dist.buy, dist.hold, dist.sell)

    g = go.Figure(go.Indicator(
        mode="gauge+number", value=temp, number={'suffix': "%",  'font': {'size': 14}},
        title={'text': "Temperatura (SELL↔HOLD↔BUY)",'font': {'size': 12}},
        gauge={'axis': {'range': [0, 100]},
               'threshold': {'line': {'width': 2}, 'thickness': 0.75, 'value': temp}}
    ))
    g.update_layout(height=100, margin=dict(l=10,r=10,b=10,t=25))
    st.plotly_chart(g, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    for title, val, col in [("SELL", dist.sell*100, c1), ("HOLD", dist.hold*100, c2), ("BUY", dist.buy*100, c3)]:
        with col:
            gg = go.Figure(go.Indicator(mode="gauge+number", value=val, number={'suffix': "%"},
                                        title={'text': title}, gauge={'axis': {'range':[0,100]}}))
            gg.update_layout(height=80, margin=dict(l=10,r=10,b=10,t=20))
            st.plotly_chart(gg, use_container_width=True)

    d_df = pd.DataFrame({"Clase":["BUY","HOLD","SELL"],
                         "Probabilidad":[dist.buy, dist.hold, dist.sell]})
    bar = go.Figure()
    bar.add_trace(go.Bar(x=d_df["Clase"], y=d_df["Probabilidad"]))
    bar.update_layout(height=220, yaxis=dict(tickformat=".0%"), margin=dict(l=10,r=10,b=10,t=20))
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(d_df.assign(Probabilidad=(d_df["Probabilidad"]*100).round(2)))
