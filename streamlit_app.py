# streamlit_app.py - LIVE AI Trading with Finnhub WebSocket
import json
import threading
import time
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import websocket

# Configuration
FINNHUB_API_KEY = "d28sk6pr01qle9gskjv0d28sk6pr01qle9gskjvg"
PAIRS = ["OANDA:EUR_USD", "OANDA:GBP_USD", "OANDA:USD_JPY", "OANDA:AUD_USD", "OANDA:USD_CAD"]
MIN_CONFIDENCE = 0.75

# Data buffers
price_latest = {p: 0 for p in PAIRS}
tick_buffer = {p: [] for p in PAIRS}
bar_buffer = {p: pd.DataFrame() for p in PAIRS}
log_signals = []
lock = threading.Lock()
websocket_connected = False

# WebSocket callbacks
def on_open(ws):
    global websocket_connected
    websocket_connected = True
    for p in PAIRS:
        ws.send(json.dumps({"type": "subscribe", "symbol": p}))

def on_message(ws, message):
    global price_latest
    try:
        data = json.loads(message)
        if data.get("type") != "trade":
            return
        for trade in data.get("data", []):
            sym = trade.get("s")
            price = trade.get("p")
            ts = datetime.utcfromtimestamp(trade.get("t") / 1000)
            with lock:
                tick_buffer[sym].append((ts, price))
                price_latest[sym] = price
                if len(tick_buffer[sym]) > 200:
                    tick_buffer[sym] = tick_buffer[sym][-200:]
    except Exception as e:
        print(f"Tick error: {e}")

def on_error(ws, error):
    global websocket_connected
    websocket_connected = False
    print(f"WebSocket error: {error}")

def on_close(ws, *_):
    global websocket_connected
    websocket_connected = False
    print("WebSocket closed")

def start_websocket():
    ws = websocket.WebSocketApp(
        f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()

def build_ohlc(symbol):
    while True:
        try:
            now = datetime.utcnow().replace(second=0, microsecond=0)
            with lock:
                ticks = tick_buffer[symbol]
                old_ticks = [t for t in ticks if t[0] < now]
                tick_buffer[symbol] = [t for t in ticks if t[0] >= now]
                if not old_ticks:
                    time.sleep(1)
                    continue
                df = pd.DataFrame(old_ticks, columns=["dt", "price"])
                df.set_index("dt", inplace=True)
                o = df["price"].iloc[0]
                h = df["price"].max()
                l = df["price"].min()
                c = df["price"].iloc[-1]
                v = len(df)
                bar = pd.DataFrame({
                    "open": [o], "high": [h], "low": [l], "close": [c], "volume": [v]
                }, index=[now - pd.Timedelta(minutes=1)])
                
                if bar_buffer[symbol].empty:
                    bar_buffer[symbol] = bar
                else:
                    bar_buffer[symbol] = pd.concat([bar_buffer[symbol], bar]).iloc[-600:]
        except Exception as e:
            print(f"OHLC error: {e}")
        time.sleep(1)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def get_signal(df, weights):
    if len(df) < 50:
        return "hold", 0.0
        
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["rsi"] = rsi(df["close"])
    df["atr"] = (df["high"] - df["low"]).rolling(window=14).mean()
    df["atr_avg"] = df["atr"].rolling(window=30).mean()

    # Latest values
    i = len(df) - 1
    score = 0
    conditions = 0

    # Trend analysis
    if df["ema9"].iloc[i] > df["ema21"].iloc[i] > df["ema50"].iloc[i]:
        score += 0.3 * weights.get("trend", 1.2)
        conditions += 1
        trend_direction = "up"
    elif df["ema9"].iloc[i] < df["ema21"].iloc[i] < df["ema50"].iloc[i]:
        score += 0.3 * weights.get("trend", 1.2)
        conditions += 1
        trend_direction = "down"
    else:
        trend_direction = "neutral"

    # RSI momentum
    if df["rsi"].iloc[i] < 35:
        score += 0.25 * weights.get("momentum", 1.1)
        conditions += 1
        rsi_condition = "oversold"
    elif df["rsi"].iloc[i] > 65:
        score += 0.25 * weights.get("momentum", 1.1)
        conditions += 1
        rsi_condition = "overbought"
    else:
        rsi_condition = "neutral"

    # Volatility
    if df["atr"].iloc[i] > df["atr_avg"].iloc[i] * 1.2:
        score += 0.2 * weights.get("volatility", 1.0)
        conditions += 1

    # Momentum confirmation
    if i >= 5:
        momentum = (df["close"].iloc[i] - df["close"].iloc[i-5]) / df["close"].iloc[i-5]
        if abs(momentum) > 0.001:
            score += 0.25
            conditions += 1

    # Generate signals
    if score >= 0.75 and conditions >= 3:
        if trend_direction == "up" and rsi_condition == "oversold":
            return "buy", min(score, 0.95)
        elif trend_direction == "down" and rsi_condition == "overbought":
            return "sell", min(score, 0.95)
    
    return "hold", 0.0

# Strategy weights
strategy_weights = {
    "trend": 1.2,
    "momentum": 1.1,
    "volatility": 1.0,
}

# Streamlit UI
st.set_page_config(page_title="🔴 LIVE AI Trading", layout="wide", page_icon="🔴")
st.title("🔴 LIVE AI Trading Dashboard - Finnhub Stream")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
col1.metric("WebSocket", "🟢 Connected" if websocket_connected else "🔴 Connecting...")
col2.metric("Pairs", len(PAIRS))
col3.metric("Strategy", "EMA + RSI + Momentum")
col4.metric("Min Confidence", f">= {MIN_CONFIDENCE:.2f}")

# Start background threads
if "started" not in st.session_state:
    for pair in PAIRS:
        threading.Thread(target=build_ohlc, args=(pair,), daemon=True).start()
    threading.Thread(target=start_websocket, daemon=True).start()
    st.session_state["started"] = True
    st.success("🚀 Real-time WebSocket connection initiated!")

# Main price display
cols = st.columns(len(PAIRS))

# Main loop
current_signals = []
for i, pair in enumerate(PAIRS):
    if not bar_buffer[pair].empty:
        df = bar_buffer[pair].copy()
        signal, confidence = get_signal(df, strategy_weights)
        price = price_latest.get(pair, 0)
        pair_name = pair.split(":")[1].replace("_", "")
        
        # Display signal
        signal_text = ""
        if signal != "hold" and confidence >= MIN_CONFIDENCE:
            signal_text = f"🎯 {signal.upper()} ({confidence:.2f})"
            current_signals.append({
                "time": datetime.utcnow().strftime("%H:%M:%S"),
                "pair": pair_name,
                "signal": signal.upper(),
                "confidence": f"{confidence:.2f}",
                "price": f"{price:.5f}"
            })
        
        cols[i].metric(
            pair_name,
            f"{price:.5f}" if price > 0 else "Loading...",
            signal_text
        )
    else:
        pair_name = pair.split(":")[1].replace("_", "")
        cols[i].metric(pair_name, "Connecting...", "")

# Log high-confidence signals
if current_signals:
    log_signals.extend(current_signals)
    if len(log_signals) > 20:
        log_signals = log_signals[-20:]

# Display recent signals
if log_signals:
    st.subheader("🎯 Recent High-Confidence Signals")
    signals_df = pd.DataFrame(log_signals[-10:])  # Last 10 signals
    st.dataframe(signals_df, use_container_width=True, hide_index=True)

# Connection status
st.sidebar.subheader("📊 Connection Status")
if websocket_connected:
    st.sidebar.success("✅ Finnhub WebSocket Connected")
    with lock:
        total_ticks = sum(len(ticks) for ticks in tick_buffer.values())
    st.sidebar.info(f"Real-time ticks: {total_ticks}")
else:
    st.sidebar.warning("⚠️ Connecting to WebSocket...")

st.sidebar.subheader("🔧 System Info")
st.sidebar.info("Data Source: Finnhub Real-time")
st.sidebar.info("Strategy: Multi-confirmation AI")
st.sidebar.info("Update Rate: Sub-second")

# Auto-refresh every 1 second
time.sleep(1)
st.rerun()
