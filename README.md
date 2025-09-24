# Any-Asset Trading Assistant — UX Edition (OpenAI + 1Y Daily)

- Descarga 1Y 1D (yfinance) para cualquier símbolo compatible.
- KPIs: RSI14, MACD diff (12-26-9), ATR14%, vol. 20d anualizada, MA20/50/200, 52w high/low y distancias.
- Tendencias 3/6/12M.
- OpenAI (temperature=1) con salida estricta (JSON Schema): buy/hold/sell + estrategia **executive_summary** + **technical_detail**.
- Dashboard con **gauge**, barra de **confianza**, **Entry/SL/TP**, **R:R** y KPIs.

## Ejecución
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Crea `.streamlit/secrets.toml` (no subir a GitHub):
```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4.1-mini"  # o modelo GPT-5 equivalente compatible con JSON Schema estricto
```
