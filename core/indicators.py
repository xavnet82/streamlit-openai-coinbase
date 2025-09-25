from __future__ import annotations
import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def moving_average(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([
        (h - l).abs(),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def find_last_swing(series: pd.Series, window:int=5, mode:str="low"):
    if len(series) < window*2+1: 
        return None
    mid = series[:-window].rolling(window*2+1, center=True).apply(
        (lambda x: 1 if (x[window] == (x.min() if mode=="low" else x.max())) else 0), raw=False
    )
    idx = mid[mid==1].index
    if len(idx)==0: 
        return None
    t = idx[-1]
    return t, float(series.loc[t])
