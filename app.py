
# app.py
import os
import uuid
import json
import streamlit as st
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from openai import OpenAI
from coinbase.rest import RESTClient

st.set_page_config(page_title="Trading Assistant (OpenAI + Coinbase)", page_icon="📈", layout="centered")
st.title("📈 Trading Assistant — OpenAI + Coinbase Advanced Trade")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
OPENAI_MODEL   = st.secrets.get("OPENAI_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
CB_API_KEY     = st.secrets.get("COINBASE_API_KEY", os.getenv("COINBASE_API_KEY"))
CB_API_SECRET  = st.secrets.get("COINBASE_API_SECRET", os.getenv("COINBASE_API_SECRET"))
CB_BASE_URL    = st.secrets.get("COINBASE_BASE_URL", os.getenv("COINBASE_BASE_URL", "api.coinbase.com"))
USE_SANDBOX    = bool(str(st.secrets.get("USE_SANDBOX", os.getenv("USE_SANDBOX", "false"))).lower() in ["1","true","yes"])

if not OPENAI_API_KEY:
    st.error("Falta OPENAI_API_KEY en secrets o variables de entorno.")
    st.stop()

oa_client = OpenAI(api_key=OPENAI_API_KEY)

cb_kwargs = {}
if CB_API_KEY and CB_API_SECRET:
    cb_kwargs.update(dict(api_key=CB_API_KEY, api_secret=CB_API_SECRET))
if CB_BASE_URL:
    cb_kwargs.update(dict(base_url=CB_BASE_URL))
cb_client = RESTClient(**cb_kwargs)

Action = Literal["buy", "sell", "hold"]
class TradeSignal(BaseModel):
    model_config = {"extra": "forbid"}

    product_id: str
    action: Literal["buy", "sell", "hold"]
    confidence: float = Field(..., ge=0, le=1)
    rationale: str
    analysis: str
    size_type: Literal["quote", "base"]
    size: str
    time_horizon_days: int = Field(..., ge=0)

@st.cache_data(show_spinner=False)
def get_product_price(pid: str) -> Optional[float]:
    try:
        p = cb_client.get_product(pid)
        return float(getattr(p, "price", None) or p["price"])
    except Exception:
        return None

def place_market_order_buy(pid: str, quote_size: str):
    return cb_client.market_order_buy(client_order_id=str(uuid.uuid4()), product_id=pid, quote_size=quote_size)

def place_market_order_sell(pid: str, base_size: str):
    return cb_client.market_order_sell(client_order_id=str(uuid.uuid4()), product_id=pid, base_size=base_size)

def build_prompt(product_id: str, user_notes: str, price: Optional[float]) -> list:
    system = (
        "Eres un analista cuantitativo experto. Devuelve una recomendación operativa SIMPLE (buy/sell/hold) "
        "y un análisis técnico claro. No uses derivados ni apalancamiento. Si la señal es débil, responde hold."
    )
    user = f"""
Quiero un análisis DETALLADO para tomar una decisión hoy.
Activo (Coinbase product_id si aplica): {product_id}
Precio spot conocido: {price if price is not None else "desconocido"}

Instrucciones:
- Usa heurísticas técnicas ligeras (momentum, RSI/MACD genérico, tendencia, volatilidad) y menciona supuestos.
- Devuelve: action (buy/sell/hold), confidence (0..1), rationale breve y analysis extensa.
- Si no hay ventaja clara -> hold.
- Para BUY, tamaño en 'quote' (USD). Para SELL, tamaño en 'base' (activo).
Notas del usuario: {user_notes or "-"}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

def ask_openai_for_signal(product_id: str, user_notes: str, price: Optional[float]) -> TradeSignal:
    messages = build_prompt(product_id, user_notes, price)
    schema = TradeSignal.model_json_schema()

    # 👇 Requisito de Structured Outputs: schema estricto
    # (al menos en el objeto raíz; si tuvieras objetos anidados, también allí)
    schema["additionalProperties"] = False

    completion = oa_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "trade_signal",
                "schema": schema,
                "strict": True   # mantiene el modo estricto
            }
        },
        temperature=0.2,
    )
    content = completion.choices[0].message.content
    data = json.loads(content)
    return TradeSignal.model_validate(data)

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    default_pid = st.secrets.get("DEFAULT_PRODUCT_ID", os.getenv("DEFAULT_PRODUCT_ID", "BTC-USD"))
    product_id = st.text_input("Acción/Cripto (Coinbase product_id para operar)", value=default_pid, help="Ej: BTC-USD, ETH-USD. Para activos fuera de Coinbase solo se hará el análisis.")
    user_notes = st.text_area("Notas (opcional)", value="", help="Contexto adicional para el análisis.")
    st.caption(f"Entorno Coinbase: {'SANDBOX (api-sandbox.coinbase.com)' if USE_SANDBOX or CB_BASE_URL.startswith('api-sandbox') else 'PRODUCCIÓN'}")

st.write("Introduce el activo y pulsa **Analizar con OpenAI**. Si la recomendación es BUY o SELL y el producto existe en Coinbase, podrás ejecutar la orden.")

if st.button("📊 Analizar con OpenAI"):
    with st.spinner("Llamando a OpenAI…"):
        spot = get_product_price(product_id)
        try:
            signal = ask_openai_for_signal(product_id=product_id, user_notes=user_notes, price=spot)
            st.session_state["signal"] = signal.model_dump()
        except Exception as e:
            st.error(f"Error al generar la señal: {e}")

if "signal" in st.session_state:
    sig = TradeSignal(**st.session_state["signal"])
    st.subheader("Resultado del análisis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Acción", sig.action.upper())
    c2.metric("Confianza", f"{sig.confidence:.2f}")
    c3.metric("Horizonte (días)", sig.time_horizon_days)

    with st.expander("📋 Análisis detallado"):
        st.write(sig.analysis)
    st.info(f"**Rationale:** {sig.rationale}")
    st.write(f"**Producto:** {sig.product_id}")

    spot = get_product_price(sig.product_id)
    if spot is None:
        st.warning("No se ha encontrado el producto en Coinbase. La ejecución de orden está deshabilitada.")
    else:
        st.write(f"Precio spot aproximado: {spot}")

    if sig.action != "hold" and spot is not None and (CB_API_KEY and CB_API_SECRET):
        st.divider()
        st.warning("⚠️ Esto NO es asesoramiento financiero. Revisa la señal antes de operar.")
        if sig.action == "buy":
            q = st.text_input("Importe a comprar (quote / USD)", value=sig.size if sig.size_type == "quote" else "25")
            if st.button("✅ Ejecutar BUY en Coinbase (orden de mercado)"):
                try:
                    order = place_market_order_buy(sig.product_id, quote_size=q)
                    st.success(f"Orden BUY enviada ✔️ Respuesta: {order}")
                except Exception as e:
                    st.error(f"Error al enviar BUY: {e}")
        elif sig.action == "sell":
            b = st.text_input("Cantidad a vender (base)", value=sig.size if sig.size_type == "base" else "0.001")
            if st.button("✅ Ejecutar SELL en Coinbase (orden de mercado)"):
                try:
                    order = place_market_order_sell(sig.product_id, base_size=b)
                    st.success(f"Orden SELL enviada ✔️ Respuesta: {order}")
                except Exception as e:
                    st.error(f"Error al enviar SELL: {e}")
    elif sig.action == "hold":
        st.info("Señal: HOLD. No se propone orden para hoy.")
    else:
        if not (CB_API_KEY and CB_API_SECRET):
            st.warning("Faltan claves de Coinbase en secrets para poder ejecutar órdenes.")

st.caption("© 2025 — Demo educativa. No es asesoramiento financiero.")
