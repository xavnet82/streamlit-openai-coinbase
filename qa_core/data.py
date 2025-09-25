from __future__ import annotations
import pandas as pd
import yfinance as yf

def get_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise RuntimeError(f"No hay datos para {symbol} ({period}, {interval})")

    if isinstance(df.columns, pd.MultiIndex):
        try:
            if len(df.columns.levels) == 2 and len(df.columns.levels[1]) == 1:
                df.columns = df.columns.droplevel(1)
            else:
                df.columns = ["_".join([str(c) for c in tup if c]) for tup in df.columns.values]
        except Exception:
            df.columns = ["_".join([str(c) for c in tup if c]) for tup in df.columns.values]

    if isinstance(df.index, pd.DatetimeIndex):
        try:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        except Exception:
            pass
    else:
        df.index = pd.to_datetime(df.index, errors="coerce")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Close" not in df.columns:
        candidates = [c for c in df.columns if "Close" in c]
        if candidates:
            df = df.rename(columns={candidates[0]: "Close"})
    if "Close" not in df.columns:
        raise RuntimeError("No se encontró columna 'Close' tras normalización.")
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = df["Close"]

    df = df.dropna(subset=["Open","High","Low","Close"]).sort_index()
    return df
