# streamlit_app.py - ULTIMATE POCKET OPTION AI TRADER
import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
import json
from datetime import datetime, timezone, timedelta
import random
import math
import os

# Ultimate Configuration
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
MIN_CONFIDENCE = 0.82
MAX_TRADES_PER_HOUR = 3
DB_PATH = 'ultimate_trading_ai.db'

# Realistic base prices
BASE_PRICES = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2750, 
    "USD/JPY": 150.25
}

# Global data storage
price_history = {pair: [] for pair in PAIRS}
signal_log = []
strategy_weights = {
    'rsi_weight': 1.0, 
    'momentum_weight': 1.0, 
    'volatility_weight': 1.0,
    'learning_boost': 0.0
}
performance_data = {'total_trades': 0, 'wins': 0, 'win_rate': 0.0}
system_start_time = datetime.now(timezone.utc)

class UltimateTradingAI:
    """Complete self-learning trading system"""
    
    def __init__(self):
        self.init_database()
        self.load_performance_data()
        self.trades_this_hour = 0
        self.last_hour_check = datetime.now(timezone.utc).hour
        self.test_start_time = datetime.now(timezone.utc)
        self.daily_trades = 0
        self.last_day_check = datetime.now(timezone.utc).day
    
    def init_database(self):
        """Initialize comprehensive trading database"""
        conn = sqlite3.connect(DB_PATH)
        
        # Complete signals table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pair TEXT,
                direction TEXT,
                confidence REAL,
                entry_price REAL,
                expiry_time TEXT,
                reasoning TEXT,
                rsi REAL,
                momentum REAL,
                velocity REAL,
                signal_strength TEXT,
                outcome TEXT DEFAULT 'pending',
                win_loss TEXT,
                actual_result REAL,
                profit_loss REAL,
                created_at TEXT,
                hour_of_day INTEGER,
                day_of_week INTEGER,
                market_condition TEXT
            )
        ''')
        
        # Performance tracking
        conn.execute('''
            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                total_signals INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                avg_confidence REAL,
                best_pair TEXT,
                best_hour INTEGER,
                strategy_weights TEXT,
                profit_factor REAL
            )
        ''')
        
        # Learning data
        conn.execute('''
            CREATE TABLE IF NOT EXISTS learning_data (
                feature_name TEXT PRIMARY KEY,
                importance_score REAL,
                accuracy_contribution REAL,
                recent_performance REAL,
                weight_adjustment REAL,
                last_updated TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_performance_data(self):
        """Load and calculate performance metrics"""
        global performance_data, strategy_weights
        
        conn = sqlite3.connect(DB_PATH)
        try:
            # Overall performance
            result = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(confidence) as avg_conf
                FROM signals 
                WHERE win_loss IS NOT NULL AND win_loss != ''
            ''').fetchone()
            
            if result and result[0] > 0:
                total, wins, avg_conf = result
                win_rate = (wins / total) * 100 if total > 0 else 0
                performance_data = {
                    'total_trades': total,
                    'wins': wins,
                    'win_rate': win_rate,
                    'avg_confidence': avg_conf or 0
                }
                
                # Load adaptive weights
                weights_result = conn.execute('''
                    SELECT strategy_weights FROM daily_performance 
                    ORDER BY date DESC LIMIT 1
                ''').fetchone()
                
                if weights_result:
                    try:
                        loaded_weights = json.loads(weights_result[0])
                        strategy_weights.update(loaded_weights)
                    except:
                        pass
        except Exception as e:
            print(f"Performance load error: {e}")
        
        conn.close()
    
    def should_generate_signal(self, pair, confidence):
        """Advanced trade frequency management"""
        current_hour = datetime.now(timezone.utc).hour
        current_day = datetime.now(timezone.utc).day
        
        # Reset counters
        if current_hour != self.last_hour_check:
            self.trades_this_hour = 0
            self.last_hour_check = current_hour
        
        if current_day != self.last_day_check:
            self.daily_trades = 0
            self.last_day_check = current_day
        
        # Check limits
        if self.trades_this_hour >= MAX_TRADES_PER_HOUR:
            return False, f"Hourly limit reached ({MAX_TRADES_PER_HOUR}/hour)"
        
        if self.daily_trades >= 72:  # Daily safety limit
            return False, "Daily limit reached (72 trades)"
        
        # Quality threshold (adaptive based on performance)
        min_threshold = MIN_CONFIDENCE
        if performance_data['total_trades'] > 20:
            if performance_data['win_rate'] < 60:
                min_threshold = 0.88  # Raise bar if poor performance
            elif performance_data['win_rate'] > 80:
                min_threshold = 0.80  # Lower slightly if great performance
        
        if confidence >= min_threshold:
            self.trades_this_hour += 1
            self.daily_trades += 1
            return True, f"Signal #{self.trades_this_hour}/3 this hour (Quality: {confidence:.1%})"
        
        return False, f"Confidence {confidence:.1%} below threshold {min_threshold:.1%}"
    
    def save_signal(self, signal_data):
        """Save signal with comprehensive data"""
        conn = sqlite3.connect(DB_PATH)
        try:
            current_time = datetime.now(timezone.utc)
            conn.execute('''
                INSERT INTO signals (
                    timestamp, pair, direction, confidence, entry_price, 
                    expiry_time, reasoning, rsi, momentum, velocity,
                    signal_strength, created_at, hour_of_day, day_of_week,
                    market_condition
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_data['entry_time'], signal_data['pair'], signal_data['direction'],
                signal_data['confidence'], signal_data['entry_price'], signal_data['expiry_time'],
                signal_data['reasoning'], signal_data['rsi'], signal_data['momentum'],
                signal_data['velocity'], signal_data.get('signal_strength', 'STANDARD'),
                current_time.isoformat(), current_time.hour, current_time.weekday(),
                self.detect_market_condition()
            ))
            conn.commit()
        except Exception as e:
            print(f"Save signal error: {e}")
        
        conn.close()
    
    def detect_market_condition(self):
        """Detect current market condition"""
        hour = datetime.now(timezone.utc).hour
        
        # Market session detection
        if 8 <= hour <= 16:
            return "LONDON_SESSION"
        elif 13 <= hour <= 21:
            return "NY_SESSION" 
        elif 22 <= hour <= 6:
            return "ASIAN_SESSION"
        else:
            return "OVERLAP"
    
    def update_learning_system(self):
        """Advanced learning system update"""
        global strategy_weights
        
        conn = sqlite3.connect(DB_PATH)
        try:
            # Get recent performance by feature
            recent_analysis = conn.execute('''
                SELECT 
                    rsi, momentum, velocity, confidence, win_loss, pair,
                    hour_of_day, market_condition
                FROM signals 
                WHERE win_loss IS NOT NULL 
                ORDER BY created_at DESC 
                LIMIT 100
            ''').fetchall()
            
            if len(recent_analysis) >= 20:
                # Analyze feature performance
                rsi_performance = []
                momentum_performance = []
                confidence_performance = []
                
                for trade in recent_analysis:
                    rsi, momentum, velocity, conf, result, pair, hour, market = trade
                    win = 1 if result == 'win' else 0
                    
                    # RSI effectiveness
                    if abs(rsi - 50) > 20:  # Strong RSI signal
                        rsi_performance.append(win)
                    
                    # Momentum effectiveness  
                    if abs(momentum) > 0.1:  # Strong momentum
                        momentum_performance.append(win)
                    
                    # High confidence effectiveness
                    if conf > 0.85:
                        confidence_performance.append(win)
                
                # Calculate performance rates
                rsi_win_rate = np.mean(rsi_performance) if rsi_performance else 0.5
                momentum_win_rate = np.mean(momentum_performance) if momentum_performance else 0.5
                
                # Adaptive weight adjustment
                strategy_weights['rsi_weight'] = max(0.5, min(2.0, 
                    strategy_weights['rsi_weight'] + (rsi_win_rate - 0.7) * 0.1))
                
                strategy_weights['momentum_weight'] = max(0.5, min(2.0,
                    strategy_weights['momentum_weight'] + (momentum_win_rate - 0.7) * 0.1))
                
                # Learning boost based on overall performance
                if performance_data['win_rate'] > 75:
                    strategy_weights['learning_boost'] = min(0.1, 
                        strategy_weights['learning_boost'] + 0.01)
                
                # Save updated weights
                today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
                conn.execute('''
                    INSERT OR REPLACE INTO daily_performance 
                    (date, total_signals, wins, losses, win_rate, strategy_weights)
                    SELECT 
                        ?, 
                        COUNT(*), 
                        SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END),
                        SUM(CASE WHEN win_loss = 'loss' THEN 1 ELSE 0 END),
                        AVG(CASE WHEN win_loss = 'win' THEN 1.0 ELSE 0.0 END) * 100,
                        ?
                    FROM signals 
                    WHERE DATE(created_at) = ?
                ''', (today, json.dumps(strategy_weights), today))
                
                conn.commit()
                
        except Exception as e:
            print(f"Learning update error: {e}")
        
        conn.close()
    
    def get_comprehensive_stats(self):
        """Get comprehensive trading statistics"""
        conn = sqlite3.connect(DB_PATH)
        
        try:
            # Today's performance
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            today_stats = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(confidence) as avg_conf
                FROM signals 
                WHERE DATE(created_at) = ? AND win_loss IS NOT NULL
            ''', (today,)).fetchone()
            
            # Best performing pair
            best_pair = conn.execute('''
                SELECT 
                    pair,
                    AVG(CASE WHEN win_loss = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                    COUNT(*) as count
                FROM signals 
                WHERE win_loss IS NOT NULL
                GROUP BY pair
                HAVING COUNT(*) >= 10
                ORDER BY win_rate DESC
                LIMIT 1
            ''').fetchone()
            
            # Best performing hour
            best_hour = conn.execute('''
                SELECT 
                    hour_of_day,
                    AVG(CASE WHEN win_loss = 'win' THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                    COUNT(*) as count
                FROM signals 
                WHERE win_loss IS NOT NULL
                GROUP BY hour_of_day
                HAVING COUNT(*) >= 5
                ORDER BY win_rate DESC
                LIMIT 1
            ''').fetchone()
            
            return {
                'today': today_stats or (0, 0, 0),
                'best_pair': best_pair,
                'best_hour': best_hour
            }
        
        except:
            return {'today': (0, 0, 0), 'best_pair': None, 'best_hour': None}
        
        finally:
            conn.close()

# Initialize Ultimate AI System
ultimate_ai = UltimateTradingAI()

def generate_ultimate_price(pair, previous_prices):
    """Generate highly realistic price movements"""
    base_price = BASE_PRICES[pair]
    
    if not previous_prices:
        return base_price
    
    last_price = previous_prices[-1]
    
    # Advanced volatility modeling
    base_volatility = {
        "EUR/USD": 0.0003,
        "GBP/USD": 0.0004,  
        "USD/JPY": 0.003
    }
    
    # Market session adjustments
    current_hour = datetime.now(timezone.utc).hour
    session_multiplier = 1.0
    
    if 8 <= current_hour <= 16:  # London session
        session_multiplier = 1.3
    elif 13 <= current_hour <= 21:  # NY session
        session_multiplier = 1.5
    elif 0 <= current_hour <= 6:  # Asian session
        session_multiplier = 0.8
    
    volatility = base_volatility[pair] * session_multiplier * strategy_weights.get('volatility_weight', 1.0)
    
    # Advanced trending with market cycles
    time_factor = len(previous_prices) % 30  # 30-period cycles
    trend_strength = math.sin(time_factor * 0.2) * 0.0003
    
    # News event simulation (random spikes)
    if random.random() < 0.02:  # 2% chance of news event
        news_impact = random.choice([-0.002, 0.002])  # ±20 pips
        trend_strength += news_impact
    
    # Generate price change
    random_change = random.gauss(trend_strength, volatility)
    new_price = last_price * (1 + random_change)
    
    # Realistic bounds
    max_deviation = base_price * 0.025  # ±2.5%
    new_price = max(base_price - max_deviation, 
                   min(base_price + max_deviation, new_price))
    
    return round(new_price, 5)

def update_price_data():
    """Update price data with market realism"""
    current_time = datetime.now(timezone.utc)
    
    for pair in PAIRS:
        current_prices = [p['price'] for p in price_history[pair]]
        new_price = generate_ultimate_price(pair, current_prices)
        
        price_history[pair].append({
            'timestamp': current_time,
            'price': new_price
        })
        
        # Keep optimal data window
        if len(price_history[pair]) > 120:  # 2 hours of data
            price_history[pair] = price_history[pair][-120:]

def calculate_advanced_rsi(prices, period=9):
    """Advanced RSI with smoothing"""
    if len(prices) < period + 5:
        return 50
        
    # Calculate RSI
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Smoothed averages
    avg_gain = pd.Series(gains).ewm(span=period).mean().iloc[-1]
    avg_loss = pd.Series(losses).ewm(span=period).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Apply learning weight
    weighted_rsi = rsi * strategy_weights.get('rsi_weight', 1.0)
    return min(100, max(0, weighted_rsi))

def ultimate_binary_prediction(pair):
    """Ultimate 5-minute binary prediction with full AI"""
    data = price_history[pair]
    
    if len(data) < 40:  # Need sufficient data
        return None
    
    prices = [d['price'] for d in data]
    current_price = prices[-1]
    
    # Advanced technical indicators
    rsi_value = calculate_advanced_rsi(prices)
    
    # Multi-timeframe momentum
    momentum_1m = (np.mean(prices[-2:]) - np.mean(prices[-4:])) / np.mean(prices[-4:]) * 100
    momentum_5m = (np.mean(prices[-5:]) - np.mean(prices[-15:])) / np.mean(prices[-15:]) * 100
    momentum_15m = (np.mean(prices[-15:]) - np.mean(prices[-30:])) / np.mean(prices[-30:]) * 100
    
    # Price velocity and acceleration
    velocity = (prices[-1] - prices[-5]) / prices[-5] * 100 if len(prices) >= 5 else 0
    acceleration = (momentum_1m - momentum_5m)
    
    # Market structure analysis
    recent_high = max(prices[-10:])
    recent_low = min(prices[-10:])
    price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
    
    # Apply learned weights
    rsi_weight = strategy_weights.get('rsi_weight', 1.0)
    momentum_weight = strategy_weights.get('momentum_weight', 1.0)
    learning_boost = strategy_weights.get('learning_boost', 0.0)
    
    # Ultimate signal generation
    score = 0
    conditions = 0
    reasoning = []
    direction = None
    signal_quality = "STANDARD"
    
    # 1. RSI Extreme Analysis (weighted by learning)
    if rsi_value < 20:  # Extreme oversold
        score += 0.40 * rsi_weight
        conditions += 1
        reasoning.append(f"Extreme oversold (RSI: {rsi_value:.1f})")
        direction = "CALL"
        signal_quality = "EXTREME"
    elif rsi_value > 80:  # Extreme overbought
        score += 0.40 * rsi_weight
        conditions += 1
        reasoning.append(f"Extreme overbought (RSI: {rsi_value:.1f})")
        direction = "PUT"
        signal_quality = "EXTREME"
    elif rsi_value < 30:  # Strong oversold
        score += 0.30 * rsi_weight
        conditions += 1
        reasoning.append(f"Strong oversold (RSI: {rsi_value:.1f})")
        direction = "CALL"
        signal_quality = "STRONG"
    elif rsi_value > 70:  # Strong overbought
        score += 0.30 * rsi_weight
        conditions += 1
        reasoning.append(f"Strong overbought (RSI: {rsi_value:.1f})")
        direction = "PUT"
        signal_quality = "STRONG"
    
    if not direction:
        return None
    
    # 2. Multi-timeframe momentum confirmation (weighted)
    momentum_aligned = False
    if direction == "CALL":
        if momentum_1m > 0 and momentum_5m > 0:
            score += 0.25 * momentum_weight
            conditions += 1
            reasoning.append("Multi-timeframe bullish momentum")
            momentum_aligned = True
    else:  # PUT
        if momentum_1m < 0 and momentum_5m < 0:
            score += 0.25 * momentum_weight
            conditions += 1
            reasoning.append("Multi-timeframe bearish momentum")
            momentum_aligned = True
    
    # 3. Velocity and acceleration boost
    if direction == "CALL" and velocity > 0.12 and acceleration > 0:
        score += 0.15
        conditions += 1
        reasoning.append("Strong upward acceleration")
    elif direction == "PUT" and velocity < -0.12 and acceleration < 0:
        score += 0.15
        conditions += 1
        reasoning.append("Strong downward acceleration")
    
    # 4. Market structure confirmation
    if direction == "CALL" and price_position < 0.3:  # Near recent lows
        score += 0.10
        condition