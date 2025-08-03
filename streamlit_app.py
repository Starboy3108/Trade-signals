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
import json
import requests
from typing import Dict, List, Optional

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

# File paths
TRADE_HISTORY_PATH = '/tmp/trade_history.csv'
STRATEGY_WEIGHTS_PATH = '/tmp/strategy_weights.json'
PERFORMANCE_LOG_PATH = '/tmp/performance_analysis.json'
MARKET_CONDITIONS_PATH = '/tmp/market_conditions.json'

SUPPORTED_TIMEFRAMES = {"1 Min": 1, "3 Min": 3, "5 Min": 5, "15 Min": 15}

# Enhanced OTC pairs with volatility characteristics
PAIRS = [
    "OTC_EURUSD", "OTC_GBPUSD", "OTC_USDJPY", "OTC_AUDUSD", "OTC_USDCAD", 
    "OTC_EURJPY", "OTC_GBPJPY", "OTC_EURGBP", "OTC_AUDCAD", "OTC_NZDUSD",
    "OTC_XAUUSD", "OTC_XAGUSD", "OTC_BTCUSD", "OTC_ETHUSD", "OTC_USDCHF"
]

def ensure_file_exists(path, default_content=None):
    if not os.path.exists(path):
        if path.endswith('.csv'):
            pd.DataFrame(default_content if default_content else [{
                "trade_id":"", "timestamp":"", "pair":"", "signal":"", "confidence":"", 
                "reasoning":"", "outcome":"", "rating":"", "user_comment":"",
                "timeframe":"", "entry_price":"", "expiry_time":"", "exit_price":"",
                "market_condition":"", "volatility":"", "trend_strength":"", "news_impact":""
            }]).to_csv(path, index=False)
        elif path.endswith('.json'):
            with open(path, 'w') as f:
                json.dump(default_content if default_content else {}, f)

# Initialize files
ensure_file_exists(TRADE_HISTORY_PATH)
ensure_file_exists(STRATEGY_WEIGHTS_PATH, {
    "multi_timeframe_confirmation": 1.2,
    "volume_price_analysis": 1.1,
    "market_structure": 1.3,
    "volatility_breakout": 1.0,
    "news_sentiment": 0.9,
    "fibonacci_levels": 1.1
})
ensure_file_exists(PERFORMANCE_LOG_PATH, {"daily_performance": {}, "strategy_performance": {}})
ensure_file_exists(MARKET_CONDITIONS_PATH, {"current_trend": "neutral", "volatility_regime": "normal"})

@dataclass
class SignalFeedback:
    trade_id: str
    outcome: str
    rating: str
    user_comment: str = ""

class MarketDataProvider:
    """Enhanced market data with realistic price movements"""
    
    def __init__(self):
        self.pair_characteristics = {
            "OTC_EURUSD": {"base": 1.0850, "volatility": 0.001, "trend_bias": 0.0002},
            "OTC_GBPUSD": {"base": 1.2650, "volatility": 0.0015, "trend_bias": -0.0001},
            "OTC_USDJPY": {"base": 148.50, "volatility": 0.002, "trend_bias": 0.0003},
            "OTC_AUDUSD": {"base": 0.6720, "volatility": 0.0012, "trend_bias": 0.0001},
            "OTC_USDCAD": {"base": 1.3580, "volatility": 0.001, "trend_bias": -0.0002},
            "OTC_EURJPY": {"base": 161.20, "volatility": 0.0018, "trend_bias": 0.0002},
            "OTC_GBPJPY": {"base": 187.80, "volatility": 0.002, "trend_bias": 0.0001},
            "OTC_EURGBP": {"base": 0.8580, "volatility": 0.0008, "trend_bias": 0.0001},
            "OTC_AUDCAD": {"base": 0.9120, "volatility": 0.0013, "trend_bias": 0.0},
            "OTC_NZDUSD": {"base": 0.6180, "volatility": 0.0014, "trend_bias": 0.0001},
            "OTC_XAUUSD": {"base": 2650.00, "volatility": 0.003, "trend_bias": 0.0005},
            "OTC_XAGUSD": {"base": 31.50, "volatility": 0.004, "trend_bias": 0.0003},
            "OTC_BTCUSD": {"base": 42000.00, "volatility": 0.008, "trend_bias": 0.001},
            "OTC_ETHUSD": {"base": 2450.00, "volatility": 0.006, "trend_bias": 0.0008},
            "OTC_USDCHF": {"base": 0.8850, "volatility": 0.0009, "trend_bias": -0.0001}
        }
    
    def generate_realistic_data(self, pair: str, periods: int = 200) -> pd.DataFrame:
        """Generate realistic market data with trends, support/resistance, and volatility clusters"""
        char = self.pair_characteristics.get(pair, {"base": 1.2, "volatility": 0.001, "trend_bias": 0})
        
        # Create realistic price series
        base_price = char["base"]
        volatility = char["volatility"]
        trend_bias = char["trend_bias"]
        
        # Add market regime changes
        regime_changes = np.random.choice([0, 1], periods, p=[0.95, 0.05])  # 5% chance of regime change
        
        prices = [base_price]
        current_trend = 0
        
        for i in range(1, periods):
            if regime_changes[i]:
                current_trend = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            
            # Trend component
            trend_component = current_trend * trend_bias * base_price
            
            # Volatility clustering
            if i > 20:
                recent_volatility = np.std(np.diff(prices[-20:]))
                vol_multiplier = 1 + 0.5 * (recent_volatility / volatility - 1)
            else:
                vol_multiplier = 1
            
            # Random walk with trend and volatility clustering
            change = trend_component + np.random.normal(0, volatility * base_price * vol_multiplier)
            new_price = prices[-1] + change
            prices.append(max(new_price, base_price * 0.8))  # Prevent extreme moves
        
        df = pd.DataFrame()
        df['close'] = prices
        df['open'] = df['close'].shift(1).fillna(df['close'].iloc[0])
        
        # Realistic high/low based on intrabar movement
        intrabar_range = np.random.uniform(0.3, 1.2, periods) * volatility * base_price
        df['high'] = df[['open', 'close']].max(axis=1) + intrabar_range * 0.6
        df['low'] = df[['open', 'close']].min(axis=1) - intrabar_range * 0.4
        
        # Add volume proxy
        df['volume'] = np.random.lognormal(10, 0.5, periods)
        
        df.index = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        return df

class AdvancedSignalStrategy:
    """Multi-timeframe, self-learning trading strategy"""
    
    def __init__(self, strategy_weights: Dict[str, float]):
        self.name = "Advanced Multi-Confirmation Strategy"
        self.weights = strategy_weights
        self.min_confidence_threshold = 0.75  # Higher threshold for quality
    
    def calculate_market_structure(self, data: pd.DataFrame) -> pd.Series:
        """Identify market structure: trending, ranging, breakout"""
        highs = data['high'].rolling(20).max()
        lows = data['low'].rolling(20).min()
        
        hh = data['high'] > highs.shift(5)  # Higher high
        ll = data['low'] < lows.shift(5)   # Lower low
        hl = (data['low'] > lows.shift(5)) & (data['high'] < highs.shift(5))  # Higher low
        lh = (data['high'] < highs.shift(5)) & (data['low'] > lows.shift(5))  # Lower high
        
        # Trend strength
        uptrend = (hh | hl).rolling(10).sum() > 6
        downtrend = (ll | lh).rolling(10).sum() > 6
        
        trend_strength = pd.Series(0.0, index=data.index)
        trend_strength[uptrend] = 1.0
        trend_strength[downtrend] = -1.0
        
        return trend_strength
    
    def fibonacci_retracement_levels(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate dynamic Fibonacci levels"""
        period = 50
        swing_high = data['high'].rolling(period).max()
        swing_low = data['low'].rolling(period).min()
        
        diff = swing_high - swing_low
        
        levels = {
            'fib_236': swing_high - 0.236 * diff,
            'fib_382': swing_high - 0.382 * diff,
            'fib_500': swing_high - 0.500 * diff,
            'fib_618': swing_high - 0.618 * diff,
        }
        
        return levels
    
    def volume_price_analysis(self, data: pd.DataFrame) -> pd.Series:
        """Volume-price relationship analysis"""
        if 'volume' not in data.columns:
            return pd.Series(0, index=data.index)
        
        # Price-volume correlation
        price_change = data['close'].pct_change()
        volume_ma = data['volume'].rolling(20).mean()
        volume_ratio = data['volume'] / volume_ma
        
        # Strong moves with high volume = continuation
        # Strong moves with low volume = reversal likely
        vpa_signal = np.where(
            (abs(price_change) > 0.005) & (volume_ratio > 1.5), 1.0,  # Strong move + high volume
            np.where((abs(price_change) > 0.005) & (volume_ratio < 0.7), -0.5, 0)  # Strong move + low volume
        )
        
        return pd.Series(vpa_signal, index=data.index)
    
    def multi_timeframe_confirmation(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze multiple timeframes for confirmation"""
        try:
            # 5-min timeframe
            data_5m = data.resample('5min').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
            }).dropna()
            
            if len(data_5m) < 50:
                return {"trend_alignment": 0, "momentum_alignment": 0}
            
            # Calculate trend on higher timeframe
            ema_20 = data_5m['close'].ewm(span=20).mean()
            ema_50 = data_5m['close'].ewm(span=50).mean()
            
            higher_tf_trend = 1 if ema_20.iloc[-1] > ema_50.iloc[-1] else -1
            
            # Current timeframe trend
            current_ema_20 = data['close'].ewm(span=20).mean()
            current_ema_50 = data['close'].ewm(span=50).mean()
            current_trend = 1 if current_ema_20.iloc[-1] > current_ema_50.iloc[-1] else -1
            
            trend_alignment = 1.0 if higher_tf_trend == current_trend else 0.0
            
            return {"trend_alignment": trend_alignment, "momentum_alignment": 0.8}
            
        except Exception:
            return {"trend_alignment": 0.5, "momentum_alignment": 0.5}
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate high-quality signals with multiple confirmations"""
        data = data.copy()
        
        if len(data) < 100:
            data['signal'] = 'hold'
            data['confidence'] = 0
            return data
        
        # Core indicators
        data['ema_9'] = data['close'].ewm(span=9).mean()
        data['ema_21'] = data['close'].ewm(span=21).mean()
        data['ema_50'] = data['close'].ewm(span=50).mean()
        
        # RSI with multiple periods
        data['rsi_14'] = self.calculate_rsi(data['close'], 14)
        data['rsi_21'] = self.calculate_rsi(data['close'], 21)
        
        # MACD
        data['macd'], data['macd_signal'] = self.calculate_macd(data['close'])
        
        # Bollinger Bands
        data['bb_upper'], data['bb_middle'], data['bb_lower'] = self.calculate_bollinger_bands(data['close'])
        
        # Advanced components
        market_structure = self.calculate_market_structure(data)
        fib_levels = self.fibonacci_retracement_levels(data)
        vpa_signal = self.volume_price_analysis(data)
        mtf_analysis = self.multi_timeframe_confirmation(data)
        
        # ATR for volatility
        data['atr'] = self.calculate_atr(data)
        
        # Initialize signals
        data['signal'] = 'hold'
        data['confidence'] = 0.0
        
        # Multi-confirmation logic
        for i in range(50, len(data)):
            try:
                current_price = data['close'].iloc[i]
                
                # Trend conditions
                trend_up = (data['ema_9'].iloc[i] > data['ema_21'].iloc[i] > data['ema_50'].iloc[i])
                trend_down = (data['ema_9'].iloc[i] < data['ema_21'].iloc[i] < data['ema_50'].iloc[i])
                
                # Momentum conditions
                rsi_oversold = data['rsi_14'].iloc[i] < 35 and data['rsi_21'].iloc[i] < 40
                rsi_overbought = data['rsi_14'].iloc[i] > 65 and data['rsi_21'].iloc[i] > 60
                
                macd_bullish = data['macd'].iloc[i] > data['macd_signal'].iloc[i]
                macd_bearish = data['macd'].iloc[i] < data['macd_signal'].iloc[i]
                
                # Support/Resistance from Fibonacci
                near_support = any(abs(current_price - fib_levels[level].iloc[i]) < data['atr'].iloc[i] * 0.5 
                                 for level in fib_levels if not pd.isna(fib_levels[level].iloc[i]))
                
                # Volatility condition
                high_volatility = data['atr'].iloc[i] > data['atr'].iloc[:i].rolling(30).mean().iloc[-1] * 1.2
                
                # Volume confirmation
                volume_confirmation = vpa_signal.iloc[i] > 0
                
                # Market structure
                strong_trend = abs(market_structure.iloc[i]) > 0.7
                
                # BUY CONDITIONS
                buy_conditions = [
                    trend_up,  # Trend alignment
                    rsi_oversold,  # Oversold but not extreme
                    macd_bullish,  # Momentum confirmation
                    near_support,  # Price at key level
                    mtf_analysis["trend_alignment"] > 0.7,  # Higher timeframe confirmation
                    high_volatility,  # Volatility for good moves
                ]
                
                buy_score = sum(buy_conditions) / len(buy_conditions)
                
                # SELL CONDITIONS
                sell_conditions = [
                    trend_down,
                    rsi_overbought,
                    macd_bearish,
                    near_support,  # Could be resistance in downtrend
                    mtf_analysis["trend_alignment"] > 0.7,
                    high_volatility,
                ]
                
                sell_score = sum(sell_conditions) / len(sell_conditions)
                
                # Apply strategy weights
                weighted_buy_score = buy_score * self.weights.get("multi_timeframe_confirmation", 1.0)
                weighted_sell_score = sell_score * self.weights.get("multi_timeframe_confirmation", 1.0)
                
                # Final signal generation
                if weighted_buy_score >= self.min_confidence_threshold:
                    data.iloc[i, data.columns.get_loc('signal')] = 'buy'
                    data.iloc[i, data.columns.get_loc('confidence')] = min(weighted_buy_score, 0.95)
                elif weighted_sell_score >= self.min_confidence_threshold:
                    data.iloc[i, data.columns.get_loc('signal')] = 'sell'
                    data.iloc[i, data.columns.get_loc('confidence')] = min(weighted_sell_score, 0.95)
                    
            except Exception:
                continue
        
        return data
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, prices: pd.Series, period=20, std_dev=2):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    def calculate_atr(self, data: pd.DataFrame, period=14):
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

class SelfLearningEngine:
    """Adaptive learning system that improves strategy weights based on performance"""
    
    def __init__(self):
        self.performance_history = self.load_performance_history()
        self.strategy_weights = self.load_strategy_weights()
    
    def load_performance_history(self) -> Dict:
        try:
            with open(PERFORMANCE_LOG_PATH, 'r') as f:
                return json.load(f)
        except:
            return {"daily_performance": {}, "strategy_performance": {}}
    
    def load_strategy_weights(self) -> Dict[str, float]:
        try:
            with open(STRATEGY_WEIGHTS_PATH, 'r') as f:
                return json.load(f)
        except:
            return {
                "multi_timeframe_confirmation": 1.2,
                "volume_price_analysis": 1.1,
                "market_structure": 1.3,
                "volatility_breakout": 1.0,
                "fibonacci_levels": 1.1
            }
    
    def analyze_recent_performance(self) -> Dict[str, float]:
        """Analyze recent trade performance and identify patterns"""
        try:
            df = pd.read_csv(TRADE_HISTORY_PATH)
            if df.empty:
                return {}
            
            # Focus on last 50 trades
            recent_trades = df.tail(50)
            completed_trades = recent_trades[recent_trades['outcome'].isin(['win', 'loss'])]
            
            if completed_trades.empty:
                return {}
            
            analysis = {}
            
            # Overall win rate
            win_rate = len(completed_trades[completed_trades['outcome'] == 'win']) / len(completed_trades)
            analysis['recent_win_rate'] = win_rate
            
            # Performance by confidence level
            high_conf_trades = completed_trades[completed_trades['confidence'].astype(float) > 0.8]
            if not high_conf_trades.empty:
                high_conf_win_rate = len(high_conf_trades[high_conf_trades['outcome'] == 'win']) / len(high_conf_trades)
                analysis['high_confidence_win_rate'] = high_conf_win_rate
            
            # Performance by timeframe
            for tf in SUPPORTED_TIMEFRAMES.keys():
                tf_trades = completed_trades[completed_trades['timeframe'].str.contains(str(SUPPORTED_TIMEFRAMES[tf]))]
                if not tf_trades.empty:
                    tf_win_rate = len(tf_trades[tf_trades['outcome'] == 'win']) / len(tf_trades)
                    analysis[f'{tf}_win_rate'] = tf_win_rate
            
            # Performance by pair
            pair_performance = {}
            for pair in PAIRS:
                pair_trades = completed_trades[completed_trades['pair'] == pair]
                if len(pair_trades) >= 3:  # Minimum trades for statistical significance
                    pair_win_rate = len(pair_trades[pair_trades['outcome'] == 'win']) / len(pair_trades)
                    pair_performance[pair] = pair_win_rate
            
            analysis['pair_performance'] = pair_performance
            
            return analysis
            
        except Exception as e:
            st.error(f"Performance analysis error: {e}")
            return {}
    
    def adapt_strategy_weights(self, performance_analysis: Dict) -> Dict[str, float]:
        """Adapt strategy weights based on recent performance"""
        current_weights = self.strategy_weights.copy()
        
        try:
            recent_win_rate = performance_analysis.get('recent_win_rate', 0.5)
            
            # If performance is poor, be more conservative
            if recent_win_rate < 0.6:
                current_weights['multi_timeframe_confirmation'] *= 1.1
                current_weights['volatility_breakout'] *= 0.9
                st.info("🧠 AI Learning: Increasing multi-timeframe confirmation weight due to re