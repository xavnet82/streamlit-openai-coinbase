# backtesting.py — KPI-based backtesting with optional OpenAI vetting
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import json

import numpy as np
import pandas as pd

# ---------- Indicators ----------
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min()) if len(dd) else 0.0

def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    if returns.std() == 0 or returns.dropna().empty:
        return 0.0
    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * math.sqrt(periods_per_year)
    return float((ann_ret - rf) / ann_vol) if ann_vol != 0 else 0.0

# ---------- Signal generation ----------
@dataclass
class Signal:
    date: pd.Timestamp
    side: str           # 'buy' or 'sell'
    price: float
    rule: str           # e.g., 'RSI<30', 'MA20>50_cross_up'

def generate_signals(df: pd.DataFrame,
                     use_rsi: bool = True,
                     rsi_buy: float = 30.0,
                     rsi_sell: float = 70.0,
                     use_ma_cross: bool = True,
                     fast_ma: int = 20,
                     slow_ma: int = 50) -> List[Signal]:
    """Create entry/exit *candidates* based on KPI rules."""
    close = df['Close'].astype(float)
    sigs: List[Signal] = []

    if use_rsi:
        r = rsi(close, 14)
        cross_up = (r.shift(1) >= rsi_buy) & (r < rsi_buy)   # entering oversold
        cross_dn = (r.shift(1) <= rsi_sell) & (r > rsi_sell) # entering overbought
        for ts in df.index[cross_up.fillna(False)]:
            sigs.append(Signal(ts, 'buy', float(close.loc[ts]), f'RSI<{rsi_buy}'))
        for ts in df.index[cross_dn.fillna(False)]:
            sigs.append(Signal(ts, 'sell', float(close.loc[ts]), f'RSI>{rsi_sell}'))

    if use_ma_cross:
        f = sma(close, fast_ma)
        s = sma(close, slow_ma)
        cross_up = (f.shift(1) <= s.shift(1)) & (f > s)   # golden cross
        cross_dn = (f.shift(1) >= s.shift(1)) & (f < s)   # death cross
        for ts in df.index[cross_up.fillna(False)]:
            sigs.append(Signal(ts, 'buy', float(close.loc[ts]), f'MA{fast_ma}>{slow_ma}_cross_up'))
        for ts in df.index[cross_dn.fillna(False)]:
            sigs.append(Signal(ts, 'sell', float(close.loc[ts]), f'MA{fast_ma}<{slow_ma}_cross_dn'))

    sigs.sort(key=lambda s: s.date)
    return sigs

# ---------- OpenAI vetting (optional) ----------
def openai_vet_signals(oa_client, model: str, symbol: str, df: pd.DataFrame, signals: List[Signal]) -> List[Dict[str, Any]]:
    """Ask OpenAI to validate/annotate each signal; returns decisions list."""
    if oa_client is None or model is None or not signals:
        # Passthrough: accept all
        return [{
            "date": str(s.date.date() if hasattr(s.date, "date") else s.date),
            "side": s.side, "accept": True, "reason": s.rule,
            "stop_loss_adj": None, "take_profit_adj": None
        } for s in signals]

    decisions = []
    for s in signals:
        idx = df.index.get_loc(s.date)
        start = max(0, idx - 60)
        ctx = df.iloc[start:idx+1][["Open","High","Low","Close","Volume"]].tail(60).reset_index()
        rows = ctx.to_dict(orient="records")
        prompt = [
            {"role":"system","content":"Eres un analista técnico. Valida una señal de trading y sugiere si debe operarse."},
            {"role":"user","content":json.dumps({
                "symbol": symbol, "signal": {"date": str(s.date), "side": s.side, "rule": s.rule, "price": s.price},
                "recent_bars": rows[-30:]  # últimos 30 para token budget
            }, ensure_ascii=False)}
        ]
        schema = {
            "type":"object", "additionalProperties": False,
            "properties": {
                "accept": {"type":"boolean"},
                "reason": {"type":"string"},
                "stop_loss_adj": {"type":["number","null"]},
                "take_profit_adj": {"type":["number","null"]}
            },
            "required": ["accept","reason","stop_loss_adj","take_profit_adj"]
        }
        try:
            resp = oa_client.chat.completions.create(
                model=model,
                messages=prompt,
                response_format={"type":"json_schema","json_schema":{"name":"backtest_decision","schema":schema,"strict":True}},
                temperature=1
            )
            data = json.loads(resp.choices[0].message.content)
            decisions.append({"date": str(s.date), "side": s.side, **data})
        except Exception as e:
            decisions.append({
                "date": str(s.date), "side": s.side, "accept": True,
                "reason": f"auto-accept (fallback: {e})", "stop_loss_adj": None, "take_profit_adj": None
            })
    return decisions

# ---------- Trade simulation ----------
@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: str             # 'long' or 'short'
    entry_price: float
    exit_price: float
    pnl_pct: float
    rule: str
    notes: str

def simulate(df: pd.DataFrame,
             signals: List[Signal],
             sl_pct: float = 0.03,
             tp_pct: float = 0.06,
             capital: float = 10000.0,
             fee_pct: float = 0.0005,
             use_openai_decisions: Optional[List[Dict[str, Any]]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequential backtest: una posición a la vez, solo largos en 'buy' y cierre en 'sell' o SL/TP.
    Devuelve (trades_df, equity_curve_df)
    """
    close = df['Close'].astype(float)
    pos = None  # (entry_idx, entry_price, rule, note, dec)
    equity = capital
    equity_curve = []
    trades: List[Trade] = []
    decisions_map = {(d["date"], d["side"]): d for d in (use_openai_decisions or [])}

    for i, (ts, px) in enumerate(close.items()):
        equity_curve.append({"date": ts, "equity": equity})

        # Entradas: 'buy' aceptadas
        todays = [s for s in signals if s.date == ts and s.side == 'buy']
        for s in todays:
            dec = decisions_map.get((str(s.date), 'buy'))
            accept = dec["accept"] if dec is not None else True
            if pos is None and accept:
                entry = px * (1 + fee_pct)
                pos = (i, entry, s.rule, dec["reason"] if dec else s.rule, dec)
                break

        # Salidas: SL/TP o señal 'sell'
        if pos is not None:
            entry_i, entry_price, rule, note, dec = pos
            this_sl = (dec["stop_loss_adj"] if dec and dec["stop_loss_adj"] is not None else (-sl_pct))  # negativo para largos
            this_tp = (dec["take_profit_adj"] if dec and dec["take_profit_adj"] is not None else (tp_pct))

            pnl_pct = (px - entry_price) / entry_price
            exit_reason = None
            if pnl_pct <= this_sl:
                exit_reason = f"SL {this_sl:.2%}"
            elif pnl_pct >= this_tp:
                exit_reason = f"TP {this_tp:.2%}"

            if exit_reason is None:
                sells = [s for s in signals if s.date == ts and s.side == 'sell']
                if sells:
                    exit_reason = sells[0].rule

            if exit_reason is not None:
                exit_price = px * (1 - fee_pct)
                trade_pnl = (exit_price - entry_price) / entry_price
                trades.append(Trade(
                    entry_date=df.index[entry_i], exit_date=ts, side='long',
                    entry_price=float(entry_price), exit_price=float(exit_price),
                    pnl_pct=float(trade_pnl), rule=rule, notes=f"{note} | {exit_reason}"
                ))
                equity *= (1 + trade_pnl)
                pos = None

    if pos is not None:
        entry_i, entry_price, rule, note, _ = pos
        px = close.iloc[-1]
        exit_price = px * (1 - fee_pct)
        trade_pnl = (exit_price - entry_price) / entry_price
        trades.append(Trade(
            entry_date=df.index[entry_i], exit_date=df.index[-1], side='long',
            entry_price=float(entry_price), exit_price=float(exit_price),
            pnl_pct=float(trade_pnl), rule=rule, notes=f"{note} | close_at_end"
        ))

    trades_df = pd.DataFrame([t.__dict__ for t in trades])
    eq_df = pd.DataFrame(equity_curve)
    return trades_df, eq_df

def summarize(trades_df: pd.DataFrame, eq_df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    if trades_df.empty:
        out.update({"trades": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "total_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0})
        return out
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]
    out["trades"] = len(trades_df)
    out["win_rate"] = float(len(wins) / len(trades_df))
    out["avg_win"] = float(wins['pnl_pct'].mean()) if not wins.empty else 0.0
    out["avg_loss"] = float(losses['pnl_pct'].mean()) if not losses.empty else 0.0
    out["total_return"] = float((eq_df['equity'].iloc[-1] / eq_df['equity'].iloc[0] - 1.0)) if not eq_df.empty else 0.0
    daily = eq_df['equity'].pct_change().dropna()
    out["max_drawdown"] = float(max_drawdown(eq_df['equity']))
    out["sharpe"] = float(sharpe(daily))
    return out

# ---------- Visualization ----------
def plot_signals_candles(df: pd.DataFrame, signals: List[Signal], trades_df: Optional[pd.DataFrame] = None):
    """
    Crea una figura de velas con marcadores para señales (BUY/SELL) y, opcionalmente,
    entradas/salidas reales de trades_df.
    """
    import plotly.graph_objects as go

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="OHLC"
        )
    ])

    # Señales
    if signals:
        # BUY markers (triangle-up) at Low - small offset
        buy_pts_x = [s.date for s in signals if s.side == "buy" and s.date in df.index]
        buy_pts_y = [float(df.loc[d, "Low"]) * 0.995 for d in buy_pts_x]
        if buy_pts_x:
            fig.add_trace(go.Scatter(
                x=buy_pts_x, y=buy_pts_y, mode="markers",
                marker=dict(symbol="triangle-up", size=10),
                name="BUY signals"
            ))

        # SELL markers (triangle-down) at High + small offset
        sell_pts_x = [s.date for s in signals if s.side == "sell" and s.date in df.index]
        sell_pts_y = [float(df.loc[d, "High"]) * 1.005 for d in sell_pts_x]
        if sell_pts_x:
            fig.add_trace(go.Scatter(
                x=sell_pts_x, y=sell_pts_y, mode="markers",
                marker=dict(symbol="triangle-down", size=10),
                name="SELL signals"
            ))

    # Trades (vertical entry/exit lines)
    if trades_df is not None and not trades_df.empty:
        for _, row in trades_df.iterrows():
            fig.add_vline(x=row["entry_date"], line_width=1, line_dash="dot", annotation_text="Entry", annotation_position="top")
            fig.add_vline(x=row["exit_date"],  line_width=1, line_dash="dash", annotation_text="Exit", annotation_position="top")

    fig.update_layout(xaxis_rangeslider_visible=False, height=520, margin=dict(l=10, r=10, t=30, b=10))
    return fig

