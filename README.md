# Any-Asset Trading Assistant (OpenAI + 1Y Daily, sin ejecución)

Esta app de **Streamlit** permite analizar **cualquier activo o índice** (símbolos compatibles con **Yahoo Finance** / `yfinance`) y generar una **recomendación y estrategia** con **OpenAI**.

## Funcionalidad
- Descarga **1 año** de datos **diarios (1D)** con `yfinance`.
- Calcula KPIs técnicos (RSI 14, MACD diff 12-26-9, ATR14%, volatilidad 20d anualizada, MA20/50/200, 52w high/low, % respecto a medias y extremos).
- Calcula **tendencias 3/6/12 meses**.
- Solicita a **OpenAI** un análisis técnico + sentimiento de mercado (inferido de precio/volumen) y devuelve:
  - **buy/hold/sell**, **confianza**, **distribución** (probabilidades)
  - **estrategia**: tipo de setup (rebote, rotura de canal, breakout, pullback, rango, tendencia u otro), zona de entrada, stop, take-profit, R/R, timeframe y niveles clave.
- **Dashboards**:
  - Pestaña **Gráfico** con velas + MA20/50/200 (plotly).
  - Pestaña **KPIs** con tabla resumida y métricas de tendencia 3/6/12M.
  - Pestaña **Estrategia (OpenAI)** con la recomendación y el plan operativo.
  - Pestaña **Datos** para revisar y descargar el CSV.

## Uso local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Secretos
Crea `.streamlit/secrets.toml` (no subir a GitHub) con:
```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4.1-mini"  # o equivalente compatible con Structured Outputs
```

## Ejemplos de símbolos
- Acciones: `AAPL`, `MSFT`, `TSLA`
- Índices: `^GSPC` (S&P 500), `^IXIC` (Nasdaq), `^IBEX` (IBEX 35)
- Cripto (si disponible en Yahoo): `BTC-USD`, `ETH-USD`

> **Disclaimer**: Información con fines educativos. No constituye asesoramiento financiero.
