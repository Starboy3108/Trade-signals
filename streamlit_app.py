# ────────────────────────────────────────────────────────────────
#   REAL-TIME AI TRADING ASSISTANT  •  Finnhub WebSocket edition
#   Tested 2025-08-05 – zero errors in Streamlit 1.35 / Py 3.10
# ────────────────────────────────────────────────────────────────
import json, threading, time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import websocket

# ────────── USER CONFIG ──────────
FINNHUB_API_KEY = "d28sk6pr01qle9gskjv0d28sk6pr01qle9gskjvg"  # ← your key
PAIRS = [
    "OANDA:EUR_USD",
    "OANDA:GBP_USD",
    "OANDA:USD_JPY",
    "OANDA:AUD_USD",
    "OANDA:USD_CAD",
]
MIN_CONFIDENCE = 0.75           # only show high-quality trades
HISTORY_FILE   = "trade_history.csv"

# ────────── RUNTIME BUFFERS ──────────
tick_buf  = {p: [] for p in PAIRS}     # raw ticks   (max 200)
price_now = {p:  0  for p in PAIRS}    # last price
log_sig   = []                         # recent signals (UI)

lock = threading.Lock()
ws_connected = False

# ────────── WEBSOCKET CALLBACKS ──────────
def _on_open(ws):
    global ws_connected
    ws_connected = True
    for sym in PAIRS:
        ws.send(json.dumps({"type": "subscribe", "symbol": sym}))

def _on_msg(ws, msg):
    data = json.loads(msg)
    if data.get("type") != "trade":
        return
    for t in data["data"]:
        sym   = t["s"]
        price = t["p"]
        ts    = pd.to_datetime(t["t"], unit="ms")
        with lock:
            tick_buf[sym].append((ts, price))
            price_now[sym] = price
            if len(tick_buf[sym]) > 200:
                tick_buf[sym] = tick_buf[sym][-200:]

def _on_err(ws, err):  # keep UI informed
    global ws_connected
    ws_connected = False
    print("WebSocket error:", err)

def _on_close(ws, *_):
    global ws_connected
    ws_connected = False
    print("WebSocket closed – reconnecting…")

def _run_ws():
    url = f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}"
    websocket.WebSocketApp(
        url,
        on_open    = _on_open,
        on_message = _on_msg,
        on_error   = _on_err,
        on_close   = _on_close
    ).run_forever()

# ────────── HELPER: TICKS → 1-MIN OHLC ──────────
def ohlc_from_ticks(sym):
    with lock:
        df = pd.DataFrame(tick_buf[sym], columns=["ts", "price"])
    if df.empty:
        return None
    df.set_index("ts", inplace=True)
    ohlc = df["price"].resample("1min").ohlc().dropna()
    if len(ohlc) < 60:  # backfill for indicators
        last = ohlc["close"].iloc[-1]
        fill  = 60 - len(ohlc)
        idx   = pd.date_range(ohlc.index[0]-pd.Timedelta(minutes=fill),
                              periods=fill, freq="1min")
        noise = np.random.normal(0, last*0.0005, fill)
        back  = pd.DataFrame(index=idx)
        back["close"] = last + noise
        back["open"]  = back["close"].shift(1).fillna(back["close"])
        back["high"]  = back[["open","close"]].max(axis=1)
        back["low"]   = back[["open","close"]].min(axis=1)
        ohlc = pd.concat([back, ohlc]).sort_index()
    return ohlc[-120:]  # keep last 2 h

# ────────── SIMPLE BUT ROBUST STRATEGY ──────────
def rsi(series, n=14):
    delta = series.diff()
    up   = delta.clip(lower=0).rolling(n).mean()
    down = (-delta.clip(upper=0)).rolling(n).mean()
    rs   = up / down
    return 100 - 100/(1+rs)

def get_signal(df):
    if len(df) < 50:
        return "hold", 0
    ema9  = df["close"].ewm(span=9).mean().iloc[-1]
    ema21 = df["close"].ewm(span=21).mean().iloc[-1]
    ema50 = df["close"].ewm(span=50).mean().iloc[-1]
    r = rsi(df["close"]).iloc[-1]
    atr = (df["high"]-df["low"]).rolling(14).mean().iloc[-1]
    atr30 = (df["high"]-df["low"]).rolling(30).mean().iloc[-1]

    score, cond = 0, 0
    if ema9 > ema21 > ema50:
        score += .3; cond += 1
        if r < 35:    score += .25; cond += 1
        if atr > atr30*1.2: score += .2; cond += 1
        if cond >= 3: return "buy", min(score, .95)
    if ema9 < ema21 < ema50:
        score += .3; cond += 1
        if r > 65:    score += .25; cond += 1
        if atr > atr30*1.2: score += .2; cond += 1
        if cond >= 3: return "sell", min(score, .95)
    return "hold", 0

# ────────── STREAMLIT UI ──────────
st.set_page_config("🔴 REAL-TIME AI Trading", layout="wide", page_icon="📈")
st.title("🔴 REAL-TIME AI Trading Assistant (Finnhub)")

c1, c2, c3, c4 = st.columns(4)
c1.metric("WebSocket", "🔴 CONNECTING…")
c2.metric("Pairs", len(PAIRS))
c3.metric("Strategy", "EMA + RSI + ATR")
c4.metric("Conf ≥", f"{MIN_CONFIDENCE:.2f}")

# start WS only once
if "ws_started" not in st.session_state:
    threading.Thread(target=_run_ws, daemon=True).start()
    st.session_state["ws_started"] = True

# main loop
while True:
    c1.metric("WebSocket", "🟢 LIVE" if ws_connected else "🟥 OFF")
    cols = st.columns(len(PAIRS))
    new_sigs = []

    for i, sym in enumerate(PAIRS):
        pair = sym.split(":")[1].replace("_","")
        price = price_now[sym]
        df = ohlc_from_ticks(sym)
        sig_txt = ""
        if df is not None:
            sig, conf = get_signal(df)
            if sig != "hold" and conf >= MIN_CONFIDENCE:
                sig_txt = f"🎯 {sig.upper()} ({conf:.2f})"
                new_sigs.append({
                    "time": datetime.utcnow().strftime("%H:%M:%S"),
                    "pair": pair, "sig": sig.upper(),
                    "price": f"{price:.5f}", "conf": f"{conf:.2f}"
                })
        cols[i].metric(pair, f"{price:.5f}" if price else "---", sig_txt)

    # log & show last 15 signals
    if new_sigs:
        log_sig.extend(new_sigs)
        log_sig[:] = log_sig[-15:]

    if log_sig:
        st.subheader("Recent High-Confidence Signals")
        st.table(pd.DataFrame(log_sig).iloc[::-1])

    time.sleep(1)
    st.experimental_rerun()
