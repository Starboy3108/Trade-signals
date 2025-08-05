# streamlit_app.py  ── Real-Time FOREX AI with Twelve Data
import json
import threading
import time
from datetime import datetime, timezone
import random
import requests

import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────── CONFIG ─────────────────────────
API_KEY = "9618464db4744b20b4df32148ff81ff4"   # Twelve Data API key
PAIRS   = ["EUR/USD", "GBP/USD", "USD/JPY"]     # 3 core forex pairs
MIN_CONFIDENCE = 0.75                           # AI signal threshold
REFRESH_SEC = 5                                 # data poll interval

# ──────────────── GLOBAL BUFFERS & FLAGS ────────────────
price_latest = {p: 0.0 for p in PAIRS}
bar_buffer   = {p: pd.DataFrame() for p in PAIRS}
log_signals  = []
api_connected = False
lock = threading.Lock()

# ──────────────── DATA FETCH THREAD ────────────────
def fetch_prices():
    """Fetch live prices from Twelve Data every REFRESH_SEC seconds."""
    global api_connected
    url = "https://api.twelvedata.com/price"
    while True:
        ok = False
        try:
            for pair in PAIRS:
                resp = requests.get(
                    url,
                    params={"symbol": pair, "apikey": API_KEY},
                    timeout=7
                )
                if resp.status_code == 200:
                    j = resp.json()
                    if "price" in j:
                        with lock:
                            price_latest[pair] = float(j["price"])
                        ok = True
                time.sleep(0.3)  # stay below free-tier limits
        except Exception as e:
            print(f"[FETCH] {e}")
        api_connected = ok
        time.sleep(max(REFRESH_SEC - 1, 1))

# ──────────────── OHLC BUILDER THREAD ────────────────
def build_ohlc():
    """Convert streaming prices into 1-minute OHLC bars."""
    while True:
        now_min = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        with lock:
            for pair, price in price_latest.items():
                if price == 0:
                    continue
                # synthetic high/low around last price to build bar
                spread = price * 0.0006
                bar = pd.DataFrame(
                    {
                        "open":   [price + random.uniform(-spread, spread)],
                        "high":   [price + random.uniform(0,  spread)],
                        "low":    [price - random.uniform(0,  spread)],
                        "close":  [price],
                        "volume": [random.randint(800, 4000)],
                    },
                    index=[now_min]
                )
                if bar_buffer[pair].empty:
                    bar_buffer[pair] = bar
                elif now_min not in bar_buffer[pair].index:
                    bar_buffer[pair] = pd.concat(
                        [bar_buffer[pair], bar]
                    ).iloc[-300:]  # keep last 300 bars
        time.sleep(60)  # build once per minute

# ──────────────── INDICATORS & STRATEGY ────────────────
def rsi(series, length=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(length).mean()
    loss  = -delta.clip(upper=0).rolling(length).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def classify(df):
    """Return (signal, confidence) using EMA + RSI + ATR + momentum."""
    if len(df) < 60:          # need at least 60 bars
        return "hold", 0.0
    df = df.copy()
    df["ema9"]  = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"]   = rsi(df["close"])
    df["atr"]   = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_avg"] = df["atr"].rolling(30).mean()

    i = -1  # last row
    trend_up   = df["ema9"][i] > df["ema21"][i] > df["ema50"][i]
    trend_down = df["ema9"][i] < df["ema21"][i] < df["ema50"][i]
    rsi_low  = df["rsi"][i] < 35
    rsi_high = df["rsi"][i] > 65
    high_vol = df["atr"][i] > df["atr_avg"][i] * 1.2
    momentum = (df["close"][i] - df["close"][i-5]) / df["close"][i-5]
    strong_mom = abs(momentum) > 0.001

    score = 0
    if trend_up:   score += 0.25
    if trend_down: score += 0.25
    if (trend_up and rsi_low) or (trend_down and rsi_high): score += 0.2
    if high_vol:   score += 0.15
    if strong_mom: score += 0.15

    signal = "hold"
    if trend_up and score >= MIN_CONFIDENCE and momentum > 0:
        signal = "buy"
    elif trend_down and score >= MIN_CONFIDENCE and momentum < 0:
        signal = "sell"

    return signal, round(min(score, 0.95), 2)

# ──────────────── STREAMLIT UI ────────────────
st.set_page_config("LIVE FOREX AI", ":chart_with_upwards_trend:", "wide")
st.title("🔴  LIVE FOREX AI DASHBOARD  🔴")

# kick-off threads once
if "started" not in st.session_state:
    threading.Thread(target=fetch_prices, daemon=True).start()
    threading.Thread(target=build_ohlc,    daemon=True).start()
    st.session_state["started"] = True

# status header
c1, c2, c3, c4 = st.columns(4)
c1.metric("API status", "CONNECTED" if api_connected else "CONNECTING…")
c2.metric("Pairs", len(PAIRS))
c3.metric("Refresh (s)", REFRESH_SEC)
c4.metric("Min conf.", f"{MIN_CONFIDENCE:.2f}")

# live metrics
cols = st.columns(len(PAIRS))
signals_now = []

for i, pair in enumerate(PAIRS):
    price = price_latest[pair]
    df = bar_buffer[pair]
    if not df.empty:
        sig, conf = classify(df)
    else:
        sig, conf = "hold", 0.0

    delta_text = f"{sig.upper()} {conf:.2f}" if sig != "hold" else ""
    cols[i].metric(pair, f"{price:.5f}" if price else "---", delta_text)

    if sig != "hold" and conf >= MIN_CONFIDENCE:
        signals_now.append({
            "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
            "pair": pair,
            "signal": sig.upper(),
            "confidence": conf,
            "price": round(price, 5)
        })

# log & display recent signals
if signals_now:
    log_signals.extend(signals_now)
    log_signals = log_signals[-30:]

if log_signals:
    st.subheader("Latest high-confidence signals")
    st.table(pd.DataFrame(log_signals[::-1]))

# footer auto-refresh every 3 s
time.sleep(3)
st.experimental_rerun()
