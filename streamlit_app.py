# streamlit_app.py - Enhanced with auto-reconnection
import json
import threading
import time
from datetime import datetime
import random

import numpy as np
import pandas as pd
import streamlit as st
import websocket

# Configuration
FINNHUB_API_KEY = "d28sk6pr01qle9gskjv0d28sk6pr01qle9gskjvg"
PAIRS = ["OANDA:EUR_USD", "OANDA:GBP_USD", "OANDA:USD_JPY"]  # Reduced for stability
MIN_CONFIDENCE = 0.75

# Data buffers
price_latest = {p: 0 for p in PAIRS}
tick_buffer = {p: [] for p in PAIRS}
bar_buffer = {p: pd.DataFrame() for p in PAIRS}
log_signals = []
lock = threading.Lock()
websocket_connected = False
connection_attempts = 0
max_attempts = 5

# WebSocket with auto-reconnection
def on_open(ws):
    global websocket_connected, connection_attempts
    websocket_connected = True
    connection_attempts = 0
    st.success("🟢 WebSocket Connected!")
    
    # Subscribe to fewer pairs for stability
    for p in PAIRS:
        ws.send(json.dumps({"type": "subscribe", "symbol": p}))
        time.sleep(0.1)  # Small delay between subscriptions

def on_message(ws, message):
    global price_latest
    try:
        data = json.loads(message)
        if data.get("type") == "ping":
            # Respond to ping to keep connection alive
            ws.send(json.dumps({"type": "pong"}))
            return
            
        if data.get("type") != "trade":
            return
            
        for trade in data.get("data", []):
            sym = trade.get("s")
            price = trade.get("p")
            ts = datetime.utcfromtimestamp(trade.get("t") / 1000)
            
            if sym in tick_buffer:
                with lock:
                    tick_buffer[sym].append((ts, price))
                    price_latest[sym] = price
                    if len(tick_buffer[sym]) > 100:  # Reduced buffer size
                        tick_buffer[sym] = tick_buffer[sym][-100:]
    except Exception as e:
        print(f"Message error: {e}")

def on_error(ws, error):
    global websocket_connected
    websocket_connected = False
    print(f"WebSocket error: {error}")
    ws.close()

def on_close(ws, *_):
    global websocket_connected, connection_attempts
    websocket_connected = False
    print("WebSocket closed - attempting reconnect...")
    
    # Auto-reconnect with exponential backoff
    if connection_attempts < max_attempts:
        connection_attempts += 1
        wait_time = min(30, 2 ** connection_attempts)
        print(f"Reconnecting in {wait_time} seconds... (attempt {connection_attempts})")
        time.sleep(wait_time)
        start_websocket()

def start_websocket():
    global websocket_connected
    try:
        ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={FINNHUB_API_KEY}",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Set a connection timeout
        ws.run_forever(ping_interval=30, ping_timeout=10)
        
    except Exception as e:
        print(f"WebSocket startup error: {e}")
        websocket_connected = False

def build_ohlc(symbol):
    while True:
        try:
            now = datetime.utcnow().replace(second=0, microsecond=0)
            with lock:
                ticks = tick_buffer[symbol]
                old_ticks = [t for t in ticks if t[0] < now]
                tick_buffer[symbol] = [t for t in ticks if t[0] >= now]
                
                if old_ticks:
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
                        bar_buffer[symbol] = pd.concat([bar_buffer[symbol], bar]).iloc[-200:]
        except Exception as e:
            print(f"OHLC error: {e}")
        time.sleep(1)

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)

def get_signal(df):
    if len(df) < 30:  # Reduced minimum bars needed
        return "hold", 0.0
        
    # Calculate indicators
    df["ema9"] = df["close"].ewm(span=9).mean()
    df["ema21"] = df["close"].ewm(span=21).mean()
    df["rsi"] = rsi(df["close"])
    
    # Latest values
    i = len(df) - 1
    ema9 = df["ema9"].iloc[i]
    ema21 = df["ema21"].iloc[i]
    rsi_val = df["rsi"].iloc[i]
    
    # Simple but effective signal logic
    score = 0
    conditions = 0
    
    # Trend analysis
    if ema9 > ema21:
        score += 0.4
        conditions += 1
        if rsi_val < 35:  # Oversold in uptrend
            score += 0.4
            conditions += 1
            
        if conditions >= 2:
            return "buy", min(score, 0.95)
    
    elif ema9 < ema21:
        score += 0.4
        conditions += 1
        if rsi_val > 65:  # Overbought in downtrend
            score += 0.4
            conditions += 1
            
        if conditions >= 2:
            return "sell", min(score, 0.95)
    
    return "hold", 0.0

# Alternative data source (fallback)
def get_simulated_price(pair):
    """Fallback price simulation if WebSocket fails"""
    base_prices = {"OANDA:EUR_USD": 1.0850, "OANDA:GBP_USD": 1.2750, "OANDA:USD_JPY": 150.25}
    base = base_prices.get(pair, 1.0)
    return base + random.uniform(-0.01, 0.01)

# Streamlit UI
st.set_page_config(page_title="🔴 LIVE AI Trading", layout="wide", page_icon="🔴")
st.title("🔴 LIVE AI Trading Dashboard - Enhanced Connection")

# Status indicators
col1, col2, col3, col4 = st.columns(4)
connection_status = "🟢 CONNECTED" if websocket_connected else f"🔴 CONNECTING... (Attempt {connection_attempts})"
col1.metric("WebSocket", connection_status)
col2.metric("Pairs", len(PAIRS))
col3.metric("Strategy", "EMA + RSI")
col4.metric("Confidence", f">= {MIN_CONFIDENCE:.2f}")

# Connection troubleshooting info
if not websocket_connected:
    st.warning("""
    ⚠️ **WebSocket Connection Issues?**
    
    **Common fixes:**
    - Close other browser tabs using the same API key
    - Wait 30 seconds and refresh the page
    - Check if your API key is active (not sandbox)
    - Try a different network/WiFi connection
    
    **The app will auto-reconnect when possible.**
    """)

# Start background threads
if "started" not in st.session_state:
    for pair in PAIRS:
        threading.Thread(target=build_ohlc, args=(pair,), daemon=True).start()
    threading.Thread(target=start_websocket, daemon=True).start()
    st.session_state["started"] = True

# Main price display
cols = st.columns(len(PAIRS))

current_signals = []
for i, pair in enumerate(PAIRS):
    # Get price (live or simulated)
    if websocket_connected:
        price = price_latest.get(pair, 0)
        data_source = "LIVE"
    else:
        price = get_simulated_price(pair)
        data_source = "SIM"
    
    # Generate signal if we have OHLC data
    signal, confidence = "hold", 0.0
    if not bar_buffer[pair].empty and len(bar_buffer[pair]) >= 30:
        df = bar_buffer[pair].copy()
        signal, confidence = get_signal(df)
    
    # Display
    pair_name = pair.split(":")[1].replace("_", "")
    signal_text = ""
    
    if signal != "hold" and confidence >= MIN_CONFIDENCE:
        signal_text = f"🎯 {signal.upper()} ({confidence:.2f}) [{data_source}]"
        current_signals.append({
            "time": datetime.utcnow().strftime("%H:%M:%S"),
            "pair": pair_name,
            "signal": signal.upper(),
            "confidence": f"{confidence:.2f}",
            "price": f"{price:.5f}",
            "source": data_source
        })
    
    cols[i].metric(
        pair_name,
        f"{price:.5f}" if price > 0 else "Loading...",
        signal_text
    )

# Log signals
if current_signals:
    log_signals.extend(current_signals)
    if len(log_signals) > 15:
        log_signals = log_signals[-15:]

# Display recent signals
if log_signals:
    st.subheader("🎯 Recent High-Confidence Signals")
    signals_df = pd.DataFrame(log_signals[-10:])
    st.dataframe(signals_df, use_container_width=True, hide_index=True)

# Connection status sidebar
st.sidebar.subheader("📊 Connection Status")
if websocket_connected:
    st.sidebar.success("✅ Finnhub WebSocket Active")
    with lock:
        total_ticks = sum(len(ticks) for ticks in tick_buffer.values())
    st.sidebar.info(f"Live ticks received: {total_ticks}")
else:
    st.sidebar.warning("⚠️ WebSocket Connecting...")
    st.sidebar.info("Using simulated prices until connected")

st.sidebar.subheader("🔧 Troubleshooting")
st.sidebar.markdown("""
**If connection fails:**
1. **Close other tabs** with same API key
2. **Wait 1 minute** and refresh
3. **Check internet** connection
4. **Verify API key** is active

**Auto-reconnect** is active with exponential backoff.
""")

# Auto-refresh
time.sleep(2)
st.rerun()
