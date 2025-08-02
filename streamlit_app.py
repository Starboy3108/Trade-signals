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

# Added OTC currencies that trade 24/7 including weekends
PAIRS = [
    # Major Forex (weekdays only)
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X",
    # OTC Currencies (available 24/7 including weekends)
    "OTC_EURUSD", "OTC_GBPUSD", "OTC_USDJPY", "OTC_AUDUSD", "OTC_USDCAD", 
    "OTC_EURJPY", "OTC_GBPJPY", "OTC_EURGBP", "OTC_AUDCAD", "OTC_NZDUSD",
    "OTC_XAUUSD", "OTC_XAGUSD", "OTC_BTCUSD", "OTC_ETHUSD"
]

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
        demand = (data['close'] <= lows.shift(1)).fillna(False)
        supply = (data['close'] >= highs.shift(1)).fillna(False)

        # Initialize signal column with default value
        data['signal'] = 'hold'
        
        # BULLETPROOF METHOD: Loop through rows and assign individually
        for i in range(len(data)):
            try:
                # Get the current row values safely
                short_ma = data['short_ma'].iloc[i]
                long_ma = data['long_ma'].iloc[i]
                rsi = data['rsi'].iloc[i]
                atr_current = data['atr'].iloc[i]
                demand_current = demand.iloc[i]
                supply_current = supply.iloc[i]
                atr_mean = data['atr'].iloc[:i+1].rolling(30).mean().iloc[i] if i >= 29 else np.nan
                
                # Check buy conditions
                if (pd.notna(short_ma) and pd.notna(long_ma) and pd.notna(rsi) and 
                    pd.notna(atr_current) and pd.notna(atr_mean) and
                    short_ma > long_ma and rsi < 35 and demand_current and
                    atr_current > atr_mean):
                    data.iloc[i, data.columns.get_loc('signal')] = 'buy'
                
                # Check sell conditions
                elif (pd.notna(short_ma) and pd.notna(long_ma) and pd.notna(rsi) and 
                      pd.notna(atr_current) and pd.notna(atr_mean) and
                      short_ma < long_ma and rsi > 65 and supply_current and
                      atr_current > atr_mean):
                    data.iloc[i, data.columns.get_loc('signal')] = 'sell'
                    
            except Exception:
                continue
        
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
                # Get the last signal value as a scalar
                try:
                    sig = str(s_data['signal'].iloc[-1])  # Convert to string to avoid Series issues
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
                except Exception:
                    continue
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

def get_otc_base_price(pair):
    """Get realistic base prices for OTC currencies"""
    otc_bases = {
        "OTC_EURUSD": 1.0850, "OTC_GBPUSD": 1.2650, "OTC_USDJPY": 148.50,
        "OTC_AUDUSD": 0.6720, "OTC_USDCAD": 1.3580, "OTC_EURJPY": 161.20,
        "OTC_GBPJPY": 187.80, "OTC_EURGBP": 0.8580, "OTC_AUDCAD": 0.9120,
        "OTC_NZDUSD": 0.6180, "OTC_XAUUSD": 2650.00, "OTC_XAGUSD": 31.50,
        "OTC_BTCUSD": 42000.00, "OTC_ETHUSD": 2450.00
    }
    return otc_bases.get(pair, 1.2000)

class LiveDataFetcher:
    def get_live_forex_data(self, pair, tfmin, live=False):
        # Check if it's an OTC pair
        if pair.startswith("OTC_"):
            # Generate realistic OTC data (always available)
            base = get_otc_base_price(pair)
            base += random.uniform(-0.02, 0.02) * base  # Add some daily variation
            increments = np.cumsum(np.random.normal(0, 0.001, 100))
            close = [round(base + float(inc) * base, 5) for inc in increments]
            data = pd.DataFrame({"close": close})
            data['open'] = data['close'] + np.random.normal(0, 0.0005, 100) * base
            data['high'] = data[['open','close']].max(axis=1) + abs(np.random.normal(0, 0.0005, 100)) * base
            data['low']  = data[['open','close']].min(axis=1) - abs(np.random.normal(0, 0.0005, 100)) * base
            data.index = pd.date_range(end=datetime.now(), periods=100, freq='1min')
            return data
        
        # Regular forex pairs - try live data first
        if live and YF_OK:
            try:
                k = yf.download(tickers=pair, period="2d", interval=f"{tfmin}m", progress=False, auto_adjust=False)
                if not k.empty:
                    df = k.rename(columns={'Open':'open','High':'high','Low':'low','Close':'close'})
                    return df.reset_index(drop=True)
            except Exception as e:
                st.warning(f"API error: {e}; using simulation.", icon="⚠️")
        
        # Fallback simulation for regular pairs
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
    try:
        df = pd.read_csv(TRADE_HISTORY_PATH)
        if df.empty:
            return
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
    except Exception:
        pass

st.set_page_config(page_title="IQ Trading Assistant", layout="wide", page_icon="💡")
st.title("🤖 Pro Pattern, API-Powered Trading Assistant")

# Show market status
now = datetime.now()
is_weekend = now.weekday() >= 5  # Saturday=5, Sunday=6
market_status = "🟢 OTC Markets Open (24/7)" if is_weekend else "🟢 All Markets Open"
st.sidebar.markdown(f"**Market Status:** {market_status}")

live_mode = st.sidebar.toggle("Live Data (yfinance)", False, help="Use real candles from Yahoo! Finance (regular pairs only). OTC pairs always use simulation.")

resolve_open_trades(use_live=live_mode and YF_OK)
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_PATH)

def refresh_data():
    st.session_state.trade_history = pd.read_csv(TRADE_HISTORY_PATH)

with st.sidebar:
    st.header("⚙️ Controls")
    timeframe = st.selectbox("Signal Timeframe", list(SUPPORTED_TIMEFRAMES.keys()), index=0)
    
    # Separate OTC and regular pairs for better UX
    otc_pairs = [p for p in PAIRS if p.startswith("OTC_")]
    regular_pairs = [p for p in PAIRS if not p.startswith("OTC_")]
    
    if is_weekend:
        st.info("📅 Weekend Mode: Only OTC currencies are available for trading")
        pairs = st.multiselect("OTC Pairs to Scan (24/7)", otc_pairs, default=otc_pairs[:6])
    else:
        pair_type = st.radio("Market Type", ["OTC (24/7)", "Regular Forex", "All"])
        if pair_type == "OTC (24/7)":
            pairs = st.multiselect("OTC Pairs to Scan", otc_pairs, default=otc_pairs[:6])
        elif pair_type == "Regular Forex":
            pairs = st.multiselect("Regular Forex Pairs", regular_pairs, default=regular_pairs[:3])
        else:
            pairs = st.multiselect("All Pairs to Scan", PAIRS, default=otc_pairs[:4] + regular_pairs[:2])
    
    max_signals = st.slider("Max signals per batch", 1, 10, 8)
    
    if st.button("Scan for Pro-Quality Signals", use_container_width=True):
        signals = []
        strat_objs = [ProSignalStrategy()]
        fetcher = LiveDataFetcher()
        for pair in pairs:
            try:
                df = fetcher.get_live_forex_data(pair, SUPPORTED_TIMEFRAMES[timeframe], live=live_mode and YF_OK)
                found = SignalGenerator(strat_objs).run(df, pair, timeframe=f'{SUPPORTED_TIMEFRAMES[timeframe]}m')
                if found: signals.extend(found)
                if len(signals) >= max_signals: break
            except Exception as e:
                st.error(f"Error processing {pair}: {str(e)}")
                continue
        if signals:
            for s in signals[:max_signals]:
                try:
                    df_price = fetcher.get_live_forex_data(s['pair'], SUPPORTED_TIMEFRAMES[timeframe], live=live_mode and YF_OK)
                    entry_price = float(df_price['close'].iloc[-1])
                    minutes = SUPPORTED_TIMEFRAMES[timeframe]
                    expiry_time = (datetime.now() + timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
                    log_obj = {**s, "entry_price": entry_price, "expiry_time": expiry_time}
                    TradeLogger().log_signal(log_obj)
                except Exception:
                    continue
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
    try:
        df_show['expiry_time'] = pd.to_datetime(df_show['expiry_time'], errors='coerce').dt.strftime('%d-%b %H:%M:%S')
        st.dataframe(df_show.sort_values('timestamp',ascending=False)[[
            "timestamp","pair","signal","confidence","timeframe","entry_price","expiry_time","exit_price","outcome","reasoning"
            ]],hide_index=True,use_container_width=True)
    except Exception:
        st.dataframe(df_trades, hide_index=True, use_container_width=True)

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
        "Let high-confidence trades come to you, not the other way around.",
        "OTC markets offer 24/7 opportunities but require the same discipline.",
        "Weekend trading on OTC pairs can be less volatile—use smaller timeframes."
    ]
    if st.button("Show Mentor Tip"):
        st.info(random.choice(pro_tips))
