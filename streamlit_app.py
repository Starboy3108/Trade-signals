import os
os.environ["STREAMLIT_RUNTIME_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["HOME"] = "/tmp"

import streamlit as st
import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from collections import defaultdict

# --- Try to import yfinance for live candles ---
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

TRADE_HISTORY_PATH = '/tmp/trade_history.csv'
WEIGHTS_PATH = '/tmp/strategy_weights.json'
EVOLUTION_LOG_PATH = '/tmp/algorithm_evolution_log.md'
SUPPORTED_TIMEFRAMES = {"1 Min": 1, "3 Min": 3, "5 Min": 5, "15 Min": 15}
PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]  # yfinance FX tickers

def ensure_file_exists(path, template=None):
    if not os.path.exists(path):
        if path.endswith('.csv'):
            pd.DataFrame(template if template else []).to_csv(path, index=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f: json.dump({}, f)
        elif path.endswith('.md'):
            with open(path, 'w') as f: f.write("# Algorithm Evolution Log\n")

ensure_file_exists(TRADE_HISTORY_PATH, [{
    "trade_id":"", "timestamp":"", "pair":"", "signal":"", "confidence":"", 
    "reasoning":"", "outcome":"", "rating":"", "user_comment":"",
    "timeframe":"", "entry_price":"", "expiry_time":"", "exit_price":""
}])
ensure_file_exists(WEIGHTS_PATH, [{}])
ensure_file_exists(EVOLUTION_LOG_PATH, [{}])

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
        entry = {**signal, "trade_id": trade_id, "outcome":"pending", "rating":"pending", "user_comment":"", "expiry_time": signal.get("expiry_time"), "entry_price": signal.get("entry_price"), "exit_price": ""}
        try: df = pd.read_csv(self.path)
        except: df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.path, index=False)
    def update_trade_result(self, trade_id, exit_price, outcome):
        df = pd.read_csv(self.path)
        idx = df[df['trade_id'] == trade_id].index
        if not idx.empty:
            df.at[idx[0], "exit_price"] = exit_price
            df.at[idx[0], "outcome"] = outcome
            df.at[idx[0], "rating"] = "auto"
            df.to_csv(self.path, index=False)

class MovingAverageCrossover:
    @property
    def name(self): return f"MA_Crossover(20,50)"
    def generate_signals(self, data):
        data['short_ma'] = data['close'].rolling(20).mean()
        data['long_ma'] = data['close'].rolling(50).mean()
        data['signal'] = np.where(data['short_ma'] > data['long_ma'], 'buy', 'sell')
        data['signal'] = data['signal'].shift(1).fillna('hold')
        return data

class RSIStrategy:
    @property
    def name(self): return "RSI(14,70,30)"
    def generate_signals(self, data):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss.replace(0, 1e-6))
        data['rsi'] = 100 - (100 / (1 + rs))
        data['signal'] = 'hold'
        data.loc[(data['rsi'].shift(1) < 30) & (data['rsi'] >= 30), 'signal'] = 'buy'
        data.loc[(data['rsi'].shift(1) > 70) & (data['rsi'] <= 70), 'signal'] = 'sell'
        return data

class SignalGenerator:
    def __init__(self, strategies, weights_path=WEIGHTS_PATH):
        self.strategies, self.weights_path = strategies, weights_path
    def run(self, data, pair, timeframe="1m"):
        try: weights = defaultdict(lambda: 1.0, json.load(open(self.weights_path)))
        except: weights = defaultdict(lambda: 1.0)
        buy, sell, active = 0.0, 0.0, []
        for strat in self.strategies:
            s_data = strat.generate_signals(data.copy())
            if not s_data.empty and 'signal' in s_data.columns:
                sig = s_data['signal'].iloc[-1]
                if sig != 'hold':
                    w = weights.get(strat.name,1.0)
                    if sig=='buy': buy+=w
                    else: sell+=w
                    active.append(strat.name)
        confidence = round(max(buy, sell)/(buy+sell+1e-9),3) if (buy+sell)>0 else 0
        if buy>sell and confidence>0.6:
            return [{"timestamp":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"pair":pair,"signal":"buy","confidence":confidence,"reasoning":', '.join(active), "timeframe":timeframe}]
        elif sell>buy and confidence>0.6:
            return [{"timestamp":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"pair":pair,"signal":"sell","confidence":confidence,"reasoning":', '.join(active), "timeframe":timeframe}]
        else: return []

def price_simulation(last, tf_min):
    drift = random.uniform(-0.001, 0.001)*tf_min
    shock = random.gauss(0, 0.002)*tf_min
    return round(last + drift + shock, 5)

class LiveDataFetcher:
    def get_live_forex_data(self, pair, tfmin, live=False):
        # Live data: yfinance
        if live and YF_OK:
            try:
                interval = f"{tfmin}m"
                k = yf.download(tickers=pair, period="2d", interval=interval, progress=False)
                if not k.empty:
                    df = k.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close'})
                    return df.reset_index(drop=True)
            except Exception as e:
                st.warning(f"API error: {e}; using simulation.", icon="⚠️")
        base = 1.2 + random.uniform(-0.05,0.05)
        increments = np.cumsum(np.random.normal(0, 0.002, 100))
        close = [round(base + float(inc),5) for inc in increments]
        data = pd.DataFrame({"close": close})
        data['open'] = data['close'] + np.random.normal(0,0.001,100)
        data['high'] = data[['open','close']].max(axis=1) + abs(np.random.normal(0,0.001,100))
        data['low']  = data[['open','close']].min(axis=1) - abs(np.random.normal(0,0.001,100))
        data.index = pd.date_range(end=datetime.now(), periods=100, freq='1min')
        return data

class AnalyticsEngine:
    def __init__(self, log_path=TRADE_HISTORY_PATH): self.log_path = log_path
    def get_summary(self):
        try: df = pd.read_csv(self.log_path)
        except: return {"win_rate":"N/A", "total":0}
        completed = df[df['outcome'].isin(['win','loss'])]
        if completed.empty: return {"win_rate":"0.00%","total":0}
        wins = len(completed[completed['outcome']=='win'])
        total = len(completed)
        return {"win_rate":f"{(wins/total)*100:.2f}%","total":total}

def resolve_open_trades(use_live=False):
    df = pd.read_csv(TRADE_HISTORY_PATH)
    now = datetime.now()
    updated = False
    for idx,row in df[df['outcome']=='pending'].iterrows():
        expiry = pd.to_datetime(row["expiry_time"])
        if now >= expiry:
            tf = int(row["timeframe"].replace("m","").replace("Min",""))
            entry = float(row["entry_price"])
            if use_live and YF_OK:
                fetcher = LiveDataFetcher()
                ticks = fetcher.get_live_forex_data(row['pair'], tf, live=True)
                exit_price = float(ticks['close'].iloc[-1]) if ticks is not None and not ticks.empty else price_simulation(entry, tf)
            else:
                exit_price = price_simulation(entry, tf)
            direction = row["signal"]
            outcome = "win" if (exit_price > entry if direction == "buy" else exit_price < entry) else "loss"
            df.at[idx, "exit_price"] = exit_price
            df.at[idx, "outcome"] = outcome
            df.at[idx, "rating"] = "auto"
            updated = True
    if updated:
        df.to_csv(TRADE_HISTORY_PATH, index=False)
        st.toast("Pending trades auto-graded.", icon="⏰")

st.set_page_config(page_title="IQ Trading Assistant", layout="wide", page_icon="💡")
st.title("🤖 High-IQ, API-Powered Trading Assistant")

live_mode = st.sidebar.toggle("Live Data (yfinance)", False, help="Use real candles from Yahoo! Finance (1m/5m/15m). Otherwise, fast simulation.")

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
    if st.button("Scan for High-Quality Signals", use_container_width=True):
        signals = []
        strat_objs = [MovingAverageCrossover(), RSIStrategy()]
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
            st.toast(f"Logged {min(len(signals), max_signals)} signals!", icon="🎯")
        else: st.toast("No high-confidence setups detected.",icon="🔍")
        refresh_data()
        st.rerun()

st.header("📊 Signal Dashboard")
df_trades = st.session_state.trade_history
if df_trades.empty:
    st.info("No trades yet. Use the sidebar to scan for signals.")
else:
    df_show = df_trades.copy()
    df_show['expiry_time'] = pd.to_datetime(df_show['expiry_time'], errors='coerce').dt.strftime('%d-%b %H:%M:%S')
    st.dataframe(df_show.sort_values('timestamp',ascending=False)[["timestamp","pair","signal","confidence","timeframe","entry_price","expiry_time","exit_price","outcome","reasoning"]],hide_index=True,use_container_width=True)

col1,col2 = st.columns(2)
with col1:
    st.subheader("📈 Analytics")
    summary = AnalyticsEngine().get_summary()
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

with st.expander("📜 Algorithm Evolution Log"):
    if os.path.exists(EVOLUTION_LOG_PATH):
        with open(EVOLUTION_LOG_PATH) as f:
            st.markdown(f.read())
    else:
        st.warning("No evolution log found.")

st.sidebar.header("📥 Export")
export = df_trades.to_csv(index=False).encode()
st.sidebar.download_button("Download CSV",export,"trade_history.csv","text/csv",use_container_width=True)
