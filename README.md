# Streamlit + OpenAI + Coinbase Advanced Trade

App de ejemplo que:
1. Solicita un **activo** (product_id de Coinbase: p.ej. `BTC-USD`).
2. Llama a **OpenAI** (modelo configurable) para obtener un **análisis detallado** y una **recomendación estructurada** (`buy/sell/hold`).
3. Si procede, permite **lanzar una orden de mercado** en Coinbase Advanced Trade (BUY con `quote_size` USD o SELL con `base_size`).

> **Aviso**: Esto es una demo educativa. No es asesoramiento financiero. Verifica siempre antes de operar.

## 🚀 Puesta en marcha (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### 🔐 Configurar secretos

Crea `.streamlit/secrets.toml` (no lo subas a GitHub) con:

```toml
# OpenAI
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-4.1-mini"  # o el modelo que prefieras soportado por Structured Outputs

# Coinbase (CDP)
COINBASE_API_KEY = "organizations/{org_id}/apiKeys/{key_id}"
# Pega tu clave privada ECDSA multilínea tal cual, por ejemplo:
# COINBASE_API_SECRET = """-----BEGIN EC PRIVATE KEY-----
# ...TU_LLAVE_PRIVADA_ECDSA...
# -----END EC PRIVATE KEY-----"""
# (Las comillas triples deben permanecer para preservar saltos de línea)

# Opcional: Sandbox
USE_SANDBOX = true
COINBASE_BASE_URL = "api-sandbox.coinbase.com" # producción = api.coinbase.com

# Por defecto
DEFAULT_PRODUCT_ID = "BTC-USD"
```

> **Sandbox** de Advanced Trade: host `api-sandbox.coinbase.com` con respuestas **estáticas** para endpoints de **Accounts y Orders** (mismo formato que producción).

## ☁️ Despliegue en Streamlit Community Cloud

1. Sube el repo a **GitHub**.
2. En https://share.streamlit.io/ crea una app con `app.py`.
3. En **Settings → Secrets**, copia las mismas claves de `secrets.toml`.
4. Despliega.

## 📦 Notas técnicas

- La app usa **Structured Outputs** (JSON Schema) para obtener un objeto `TradeSignal` válido y fácil de parsear.
- Para **BUY** se usa `quote_size` (USD en pares `*-USD`). Para **SELL** se usa `base_size` (cantidad de cripto).
- El SDK oficial `coinbase-advanced-py` maneja la firma **JWT CDP** y expone métodos como `get_product`, `market_order_buy`, `market_order_sell`.

## ⚖️ Licencia

MIT (solo para el contenido de este repo). Consulta licencias de dependencias por separado.
