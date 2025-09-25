# Quant Assist (modular)

Estructura de paquetes para mantener la app por bloques:
```
quant_app/
  app.py                 # Punto de entrada Streamlit
  core/
    __init__.py
    data.py              # Descarga y normalización de datos
    indicators.py        # RSI, ATR, MAs, swings
    models.py            # Pydantic schemas
    strategy.py          # Fallback + post-procesado (guardrails)
    openai_client.py     # Prompting y llamada OpenAI
    ui.py                # Componentes UI (gauges, chips, tablas)
    utils.py             # Helpers generales
```

Ejecutar:
```bash
pip install -r requirements.txt
streamlit run app.py
```
