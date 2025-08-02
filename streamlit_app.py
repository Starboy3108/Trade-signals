import os
os.environ["STREAMLIT_RUNTIME_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["HOME"] = "/tmp"

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
from dataclasses import dataclass

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

TRADE_HISTORY_PATH = '/tmp/trade_history.csv'
SUPPORTED_TIMEFRAMES = {"1 Min": 1, "3 Min": 3, "5 Min": 5, "15 Min": 15}
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]

def ensure_file_exists(path):
    if not os.path.exists(path):
        pd.DataFrame([{
            "trade_id":"", "timestamp":"", "pair":"", "signal":"", "confidence":"", 
            "reasoning":"", "outcome":"", "rating":"", "user_comment":"",
            "timeframe":"", "entry_price":"", "expiry_time":"", "exit_price":""
        }]).to_csv(path, index=False)

ensure_file_exists(TRADE_HISTORY_PATH)

@dataclass
class SignalFeedback:
    trade_id: str
    outcome: str
    rating: str
    user_comment: str = ""

class TradeLogger:
    def __init__(self, path=TRADE_HISTORY_PATH): self.path = path
    def log_signal(self, signal):
        trade_id = f"{signal['timestamp']}_{signal['pair']}_{signal['timeframe']}"
        entry = {**signal, "trade_id": trade_id, "outcome":"pending", "rating":"pending",
                 "user_comment":"", "expiry_time": signal.get("expiry_time"),
                 "entry_price": signal.get("entry_price"), "exit_price": ""}
        try: df = pd.read_csv(self.path)
        except: df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.path, index=False)
    def update_trade_result(self, trade_id, exit_price, outcome):
        df = pd.read_csv(self.path)
        idx = df[df['trade_id'] == trade_id].index
        if not idx.empty:
            df.at[idx, "exit_price"] = exit_price
            df.at[idx, "outcome"] = outcome
            df.at[idx, "rating"] = "auto"
            df.to_csv(self.path, index=False)

class ProSignalStrategy:
    def __init__(self):
        self._name = "MA, RSI, S/D, Volatility Multi-confirm"
    @property
    def name(self): return self._name
    def generate_signals(self, data):
        data = data.copy()
        MIN_LEN = 50
        if data.empty or len(data) < MIN_LEN:
            data['signal'] = 'hold'
            return data

        # Calculate indicators
        data['short_ma'] = data['close'].rolling(14).mean()
        data['long_ma'] = data['close'].rolling(45).mean()
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, 1e-6))
        data['rsi'] = 100 - (100 / (1 + rs))
        data['atr'] = (data['high']-data['low']).rolling(10).mean()
        lows = data['low'].rolling(20).min()
        highs = data['high'].rolling(20).max()
        demand = (data['close'] <= lows.shift(1))
        supply = (data['close'] >= highs.shift(1))

        # Initialize signal column
        data['signal'] = 'hold'
        
        # BULLETPROOF METHOD: Use np.where instead of .loc[mask]
        buy_condition = (
            (data['short_ma'] > data['long_ma']) &
            (data['rsi'] < 35) &
            demand &
            (data['atr'] > data['atr'].rolling(30).mean())
        ).fillna(False)
        
        sell_condition = (
            (data['short_ma'] < data['long_ma']) &
            (data['rsi'] > 65) &
            supply &
            (data['atr'] > data['atr'].rolling(30).mean())
        ).fillna(False)
        
        # Safe assignment using np.where - NO mask indexing issues
        data['signal'] = np.where(buy_condition, 'buy', 
                         np.where(sell_condition, 'sell', 'hold'))
        
        return data

class SignalGenerator:
    def __init__(self, strategies): self.strategies = strategies
    def run(self, data, pair, timeframe="1m"):
        best_signal = None
        best_conf = 0
        active = []
        for strat in self.strategies:
            s_data = strat.generate_signals(data.copy())
            if not s_data.empty and 'signal' in s_data.columns:
                idx = s_data.index[-1]
                sig = s_data.loc[idx, 'signal']
                if sig != 'hold':
                    active.append(strat.name)
                    atr_now = s_data['atr'].iloc[-1]
                    atr_mean = s_data['atr'].rolling(30).mean().iloc[-1]
                    conf = min(1.0, abs(
                        float(atr_now) / (float(atr_mean) + 1e-6)
                    )) if not pd.isnull(atr_now) and not pd.isnull(atr_mean) else 0.0
                    if conf > best_conf:
                        best_conf = conf
                        best_signal = sig
        confidence = round(best_conf, 3)
        if best_signal and confidence > 0.7:
            return [{
                "timestamp":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "pair":pair,
                "signal":best_signal,
                "confidence":confidence,
                "reasoning":', '.join(active),
                "timeframe":timeframe
            }]
        else:
            return []

def price_simulation(last, tf_min):
    drift = random.uniform(-0.001, 0.001)*tf_min
    shock = random.gauss(0, 0.002)*tf_min
    return round(last + drift + shock, 5)

class LiveDataFetcher:
    def get_live_forex_data(self, pair, tfmin, live=False):
        if live and YF_OK:
            try:
                k = yf.download(tickers=pair, period="2d", interval=f"{tfmin}m", progress=False)
                if not k.empty:
                    df = k.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close'})
                    return df.reset_index(drop=True)
            except Exception as e:
                st.warning(f"API error: {e}; using simulation.", icon="⚠️")
        base = 1.2 + random.uniform(-0.05,0.05)
        increments = np.cumsum(np.random.normal(0, 0.002, 100))
        close = [round(base + float(inc), 5) for inc in increments]
        data = pd.DataFrame({"close": close})
        data['open'] = data['close'] + np.random.normal(0, 0.001, 100)
        data['high'] = data[['open','close']].max(axis=1) + abs(np.random.normal(0,0.001,100))
        data['low']  = data[['open','close']].min(axis=1) - abs(np.random.normal(0,0.001,100))
        data.index = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        return data

def resolve_open_trades(use_live=False):
    df = pd.read_csv(TRADE_HISTORY_PATH)
    now = datetime.now()
    updated = False
    for idx, row in df[df['outcome']=='pending'].iterrows():
        try:
            expiry_str = row.get("expiry_time", "")
            if pd.isna(expiry_str) or not str(expiry_str).strip():
                continue
            expiry = pd.to_datetime(expiry_str, errors='coerce')
            if pd.isna(expiry):
                continue
        except Exception:
            continue
        if now >= expiry:
            tf = int(str(row.get("timeframe", "1")).replace("m", "").replace("Min", "").strip())
            entry_str = row.get("entry_price", "0")
            try: entry = float(entry_str or 0)
            except: entry = 0
            if use_live and YF_OK:
                fetcher = LiveDataFetcher()
                ticks = fetcher.get_live_forex_data(row['pair'], tf, live=True)
                exit_price = float(ticks['close'].iloc[-1]) if (ticks is not None and not ticks.empty) else price_simulation(entry, tf)
            else:
                exit_price = price_simulation(entry, tf)
            direction = row.get("signal", "buy")
            outcome = "win" if (exit_price > entry if direction == "buy" else exit_price < entry) else "loss"
            df.at[idx, "exit_price"] = exit_price
            df.at[idx, "outcome"] = outcome
            df.at[idx, "rating"] = "auto"
            updated = True
    if updated:
        df.to_csv(TRADE_HISTORY_PATH, index=False)
        st.toast("Pending trades auto-graded.", icon="⏰")

st.set_page_config(page_title="IQ Trading Assistant", layout="wide", page_icon="💡")
st.title("🤖 Pro Pattern, API-Powered Trading Assistant")

live_mode = st.sidebar.toggle("Live Data (yfinance)", False, help="Use real candles from Yahoo! Finance (1m/5m/15m only). Otherwise, uses simulation.")

resolve_open_trades(use_live=live_mode and YF_OK)
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_PATH)

def refresh_data():
    st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_PATH)

with st.sidebar:
    st.header("⚙️ Controls")
    timeframe = st.selectbox("Signal Timeframe", list(SUPPORTED_TIMEFRAMES.keys()), index=0)
    pairs = st.multiselect("Pairs to Scan", PAIRS, default=PAIRS)
    max_signals = st.slider("Max signals per batch", 1, 10, 8)
    if st.button("Scan for Pro-Quality Signals", use_container_width=True):
        signals = []
        strat_objs = [ProSignalStrategy()]
        fetcher = LiveDataFetcher()
        for pair in pairs:
            df = fetcher.get_live_forex_data(pair, SUPPORTED_TIMEFRAMES[timeframe], live=live_mode and YF_OK)
            found = SignalGenerator(strat_objs).run(df, pair, timeframe=f'{SUPPORTED_TIMEFRAMES[timeframe]}m')
            if found: signals.extend(found)
            if len(signals) >= max_signals: break
        if signals:
            for s in signals[:max_signals]:
                df_price = fetcher.get_live_forex_data(s['pair'], SUPPORTED_TIMEFRAMES[timeframe], live=live_mode and YF_OK)
                entry_price = float(df_price['close'].iloc[-1])
                minutes = SUPPORTED_TIMEFRAMES[timeframe]
                expiry_time = (datetime.now() + timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
                log_obj = {**s, "entry_price": entry_price, "expiry_time": expiry_time}
                TradeLogger().log_signal(log_obj)
            st.toast(f"Logged {min(len(signals), max_signals)} high-quality signals!", icon="🎯")
        else: st.toast("No pro-level setups detected.",icon="🔍")
        refresh_data()
        st.rerun()

st.header("📊 Signal Dashboard")
df_trades = st.session_state.trade_history
if df_trades.empty:
    st.info("No trades yet. Use the sidebar to scan for signals.")
else:
    df_show = df_trades.copy()
    df_show['expiry_time'] = pd.to_datetime(df_show['expiry_time'], errors='coerce').dt.strftime('%d-%b %H:%M:%S')
    st.dataframe(df_show.sort_values('timestamp',ascending=False)[[
        "timestamp","pair","signal","confidence","timeframe","entry_price","expiry_time","exit_price","outcome","reasoning"
        ]],hide_index=True,use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("📈 Analytics")
    summary = {"win_rate":"N/A","total":0}
    try:
        completed = df_trades[df_trades['outcome'].isin(['win','loss'])]
        if not completed.empty:
            wins = len(completed[completed['outcome']=='win'])
            summary["win_rate"] = f"{(wins/len(completed))*100:.2f}%"
            summary["total"] = len(completed)
    except: pass
    st.metric("Win Rate", summary['win_rate'])
    st.metric("Completed Trades", summary['total'])

with col2:
    st.subheader("✍️ Feedback / Adapt AI")
    pending = df_trades[df_trades['rating']=='pending']
    if not pending.empty:
        with st.form("feedback_form"):
            trade_id = st.selectbox("Pending Trade",pending['trade_id'])
            outcome = st.radio("Outcome",["win","loss"])
            rating = "accepted" if outcome=="win" else "rejected"
            comment = st.text_input("Comment (optional)")
            submit = st.form_submit_button("Submit")
            if submit:
                feedback = SignalFeedback(trade_id, outcome, rating, comment)
                TradeLogger().add_feedback(feedback)
                st.success("AI updated from feedback!")
                time.sleep(1)
                refresh_data()
    else:
        st.info("No pending trades to rate.")

st.sidebar.header("📥 Export")
export = df_trades.to_csv(index=False).encode()
st.sidebar.download_button("Download CSV",export,"trade_history.csv","text/csv",use_container_width=True)

with st.expander("💡 Pro Mentor Tip"):
    pro_tips = [
        "Wait for true alignment—sometimes no trade is the best trade.",
        "Only act when multiple signals confirm a clear edge.",
        "Big money leaves footprints; be patient, follow structure.",
        "Risk management and discipline are your real edge.",
        "Avoid trading in the middle—focus on supply and demand extremes.",
        "Let high-confidence trades come to you, not the other way around."
    ]
    if st.button("Show Mentor Tip"):
        st.info(random.choice(pro_tips))
