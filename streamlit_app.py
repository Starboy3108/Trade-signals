# streamlit_app.py
"""
===============================================================================
AI Trading Signal Assistant – Hugging Face Spaces Optimized (2025, v7)
===============================================================================
1. Paste this code into streamlit_app.py in Hugging Face Spaces/Streamlit Cloud.
2. Add requirements.txt: streamlit, pandas, numpy
3. All data files and Streamlit configs go to /tmp/ (NO permission errors!)
===============================================================================
"""

import os
os.environ["STREAMLIT_RUNTIME_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["HOME"] = "/tmp"

import streamlit as st
import pandas as pd
import numpy as np
import json
import random
import time
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

# === ALWAYS use /tmp/ for files on Spaces/Cloud ===
TRADE_HISTORY_PATH = '/tmp/trade_history.csv'
WEIGHTS_PATH = '/tmp/strategy_weights.json'
EVOLUTION_LOG_PATH = '/tmp/algorithm_evolution_log.md'

def ensure_file_exists(path, template=None):
    if not os.path.exists(path):
        if path.endswith('.csv'):
            pd.DataFrame(template if template else []).to_csv(path, index=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f: json.dump({}, f)
        elif path.endswith('.md'):
            with open(path, 'w') as f: f.write("# Algorithm Evolution Log\n")

# ---- First-run autoinit ----
ensure_file_exists(TRADE_HISTORY_PATH,
    [{
        "trade_id":"", "timestamp":"", "pair":"", "signal":"", "confidence":"",
        "reasoning":"", "outcome":"", "rating":"", "user_comment":""
    }]
)
ensure_file_exists(WEIGHTS_PATH, [{}])
ensure_file_exists(EVOLUTION_LOG_PATH, [{}])

@dataclass
class SignalFeedback:
    trade_id: str
    outcome: str  # 'win' or 'loss'
    rating: str   # 'accepted' or 'rejected'
    user_comment: str = ""

class TradeLogger:
    def __init__(self, path=TRADE_HISTORY_PATH): self.path=path
    def log_signal(self, signal):
        trade_id = f"{signal['timestamp']}_{signal['pair']}"
        entry = {**signal, "trade_id": trade_id, "outcome":"pending", "rating":"pending", "user_comment":""}
        try: df = pd.read_csv(self.path)
        except: df = pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
        df.to_csv(self.path, index=False)
    def add_feedback(self, feedback: SignalFeedback):
        df = pd.read_csv(self.path)
        idx = df[df['trade_id']==feedback.trade_id].index
        if not idx.empty:
            df.loc[idx[0], ['outcome','rating','user_comment']] = [feedback.outcome, feedback.rating, feedback.user_comment]
            df.to_csv(self.path, index=False)

# ---- Example Strategies ----
class BaseStrategy:
    @property
    def name(self): raise NotImplementedError
    def generate_signals(self, data): raise NotImplementedError

class MovingAverageCrossover(BaseStrategy):
    def __init__(self, short=20, long=50): self._name=f"MA_Crossover({short},{long})"
    @property
    def name(self): return self._name
    def generate_signals(self, data):
        data['short_ma'] = data['close'].rolling(20).mean()
        data['long_ma'] = data['close'].rolling(50).mean()
        data['signal'] = np.where(data['short_ma'] > data['long_ma'], 'buy', 'sell')
        data['signal'] = data['signal'].shift(1).fillna('hold')
        return data

class RSIStrategy(BaseStrategy):
    def __init__(self, period=14, overbought=70, oversold=30): self._name=f"RSI({period},{overbought},{oversold})"
    @property
    def name(self): return self._name
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
    def __init__(self, strategies, weights_path=WEIGHTS_PATH): self.strategies, self.weights_path = strategies, weights_path
    def run(self, data, pair):
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
        if buy>sell: return [{"timestamp":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"pair":pair,"signal":"buy","confidence":round(buy/(buy+sell+1e-9),3),"reasoning":', '.join(active)}]
        elif sell>buy: return [{"timestamp":datetime.now().strftime('%Y-%m-%d %H:%M:%S'),"pair":pair,"signal":"sell","confidence":round(sell/(buy+sell+1e-9),3),"reasoning":', '.join(active)}]
        else: return []

class SelfLearningAgent:
    def __init__(self, log_path=TRADE_HISTORY_PATH, weights_path=WEIGHTS_PATH, evolution_log=EVOLUTION_LOG_PATH):
        self.log_path, self.weights_path, self.evolution_log = log_path, weights_path, evolution_log
    def update_weights(self):
        df = pd.read_csv(self.log_path)
        completed = df[df['rating'].isin(['accepted','rejected'])].copy()
        if completed.empty: return
        try: weights = defaultdict(lambda: 1.0, json.load(open(self.weights_path)))
        except: weights = defaultdict(lambda: 1.0)
        entries = []
        for _, row in completed.iterrows():
            strats = str(row.get('reasoning','')).split(',')
            for name in map(str.strip, strats):
                if not name: continue
                delta = 0.1 if row['rating']=='accepted' else -0.1
                weights[name] = max(0.1, weights.get(name,1.0)+delta)
                entries.append(f"- **{name}** weight now `{weights[name]:.2f}` (Feedback: {row['rating']})")
        with open(self.weights_path,'w') as f: json.dump(dict(weights), f, indent=4)
        with open(self.evolution_log,'a') as f: f.write(f"\n### {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"+"\n".join(entries))
        df.loc[completed.index, 'rating'] = 'processed'
        df.to_csv(self.log_path, index=False)

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

class LiveDataFetcher:
    def get_live_forex_data(self, pair):
        st.info(f"Simulated random-walk data for: {pair}",icon="🧪")
        base = 1.2 + random.uniform(-0.05, 0.05)
        series = [base + np.random.normal(0, 0.005) for _ in range(100)]
        now = datetime.now()
        df = pd.DataFrame({"close": series})
        df['open'] = df['close'] + np.random.normal(0,0.001,100)
        df['high'] = df[['open','close']].max(axis=1) + abs(np.random.normal(0,0.001,100))
        df['low']  = df[['open','close']].min(axis=1) - abs(np.random.normal(0,0.001,100))
        df.index = pd.date_range(end=now,periods=len(df),freq='1min')
        return df

if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_PATH)
def refresh_data():
    st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_PATH)

st.set_page_config(page_title="AI Trading Assistant", layout="wide", page_icon="📈")
st.title("🚀 AI Trading Assistant")

with st.sidebar:
    st.header("⚙️ Controls")
    run_mode = st.radio("Mode",["SIMULATION","LIVE (Demo)"])
    pairs = st.multiselect("Pairs to Monitor", ["EUR/USD", "GBP/USD", "USD/JPY"],default=["EUR/USD"])
    if st.button("Generate Next Signal", use_container_width=True):
        if pairs:
            with st.spinner("Analyzing..."):
                strategies = [MovingAverageCrossover(), RSIStrategy()]
                pair = random.choice(pairs)
                data = LiveDataFetcher().get_live_forex_data(pair)
                signals = SignalGenerator(strategies).run(data, pair)
                if signals:
                    TradeLogger().log_signal(signals[0])
                    st.toast(f"Signal found for {pair}!",icon="🎯")
                else:
                    st.toast("No high-confidence setup.",icon="🧐")
                refresh_data()
    else:
        st.info("Tap to generate the next simulated trade signal.",icon="🔁")

st.header("📊 Signal Dashboard")
st.dataframe(st.session_state.trade_history.sort_values('timestamp',ascending=False),hide_index=True,use_container_width=True)

col1,col2 = st.columns(2)
with col1:
    st.subheader("📈 Analytics")
    summary = AnalyticsEngine().get_summary()
    st.metric("Win Rate", summary['win_rate'])
    st.metric("Completed Trades", summary['total'])

with col2:
    st.subheader("✍️ Feedback / Adapt AI")
    pending = st.session_state.trade_history[st.session_state.trade_history['rating']=='pending']
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
                SelfLearningAgent().update_weights()
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
export = st.session_state.trade_history.to_csv(index=False).encode()
st.sidebar.download_button("Download CSV",export,"trade_history.csv","text/csv",use_container_width=True)
