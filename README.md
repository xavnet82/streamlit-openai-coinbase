# Streamlit + OpenAI + Coinbase (UI enriquecida + KPIs reales)

Incluye:
- Obtención de **precio spot** y **velas 1H** (24–48h) desde Coinbase Advanced Trade (endpoints públicos).
- Cálculo de **KPIs** (RSI 14, MACD 12-26-9 diff, Δ Precio 24h, Δ Volumen 24h, Volatilidad 24h).
- **Tendencias** (6h/24h/48h) y **% de recomendación** (buy/hold/sell).
- Llamada a OpenAI con **Structured Outputs** (JSON Schema estricto).

Documentación relevante:
- Ticker público (best_bid/ask y trades): Get Public Market Trades. 
- Velas públicas (candles 1H): Get Public Product Candles.

## Puesta en marcha

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Configura `.streamlit/secrets.toml` con tus claves.

## Notas
- Para sandbox usa `COINBASE_BASE_URL = "api-sandbox.coinbase.com"` y `USE_SANDBOX = true`.
- Para ejecutar órdenes necesitas `COINBASE_API_KEY` y `COINBASE_API_SECRET` (CDP).
