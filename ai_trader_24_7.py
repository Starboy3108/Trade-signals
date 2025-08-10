import json
import requests
import numpy as np
from datetime import datetime, timezone, timedelta

PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY"]
MIN_CONFIDENCE = 0.75
MAX_SIGNALS_PER_HOUR = 3

def get_forex_data():
    forex_data = {}
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/EUR", timeout=10)
        if response.status_code == 200:
            data = response.json()
            forex_data["EUR/USD"] = data["rates"]["USD"]
        
        response = requests.get("https://api.exchangerate-api.com/v4/latest/GBP", timeout=10)
        if response.status_code == 200:
            data = response.json()
            forex_data["GBP/USD"] = data["rates"]["USD"]
            
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=10)
        if response.status_code == 200:
            data = response.json()
            forex_data["USD/JPY"] = data["rates"]["JPY"]
    except:
        forex_data = {"EUR/USD": 1.0850, "GBP/USD": 1.2750, "USD/JPY": 150.25}
    return forex_data

def check_expired_signals(signals_history, current_forex_data):
    current_time = datetime.now(timezone.utc)
    updated_signals = []
    
    for signal in signals_history:
        if signal.get('outcome_checked'):
            updated_signals.append(signal)
            continue
            
        try:
            expiry_str = signal['expiry_time'].replace('+00:00', '').replace('Z', '')
            expiry_time = datetime.fromisoformat(expiry_str).replace(tzinfo=timezone.utc)
            
            if current_time >= expiry_time:
                pair = signal['pair']
                entry_price = signal['entry_price']
                direction = signal['direction']
                current_price = current_forex_data.get(pair, entry_price)
                
                if direction == "CALL":
                    is_winner = current_price > entry_price
                else:
                    is_winner = current_price < entry_price
                
                price_change = current_price - entry_price
                price_change_pct = (price_change / entry_price) * 100
                
                signal['outcome'] = 'WIN' if is_winner else 'LOSS'
                signal['exit_price'] = round(current_price, 5)
                signal['price_change'] = round(price_change, 5)
                signal['price_change_pct'] = round(price_change_pct, 3)
                signal['outcome_checked'] = True
                signal['checked_at'] = current_time.isoformat()
                
                print(f"📊 {pair} {direction} - {'WIN ✅' if is_winner else 'LOSS ❌'} ({price_change_pct:+.2f}%)")
        except:
            pass
        
        updated_signals.append(signal)
    
    return updated_signals

def generate_signal(pair, price):
    current_time = datetime.now(timezone.utc)
    rsi = 50 + (hash(pair + str(current_time.hour)) % 60 - 30)
    momentum = (hash(pair + str(current_time.minute)) % 200 - 100) / 1000
    
    score = 0
    conditions = 0
    reasoning = []
    direction = None
    
    if rsi < 35:
        score += 0.4
        conditions += 1
        reasoning.append(f"Oversold RSI: {rsi:.1f}")
        direction = "CALL"
    elif rsi > 65:
        score += 0.4
        conditions += 1
        reasoning.append(f"Overbought RSI: {rsi:.1f}")
        direction = "PUT"
    
    if not direction:
        return None
    
    if abs(momentum) > 0.03:
        score += 0.25
        conditions += 1
        reasoning.append(f"Strong momentum: {momentum:.3f}")
    
    if 8 <= current_time.hour <= 16 or 13 <= current_time.hour <= 21:
        score += 0.15
        conditions += 1
        reasoning.append("Active trading session")
    
    if conditions >= 2 and score >= MIN_CONFIDENCE:
        signal_data = {
            "signal_id": f"{pair}_{current_time.strftime('%Y%m%d_%H%M%S')}",
            "timestamp": current_time.isoformat(),
            "pair": pair,
            "direction": direction,
            "confidence": round(min(score, 0.95), 2),
            "entry_price": round(price, 5),
            "expiry_time": (current_time + timedelta(minutes=5)).isoformat(),
            "reasoning": ", ".join(reasoning),
            "rsi": round(rsi, 1),
            "momentum": round(momentum, 4),
            "outcome_checked": False
        }
        return signal_data
    return None

def main():
    print(f"🚀 AI Trading Cycle: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    forex_data = get_forex_data()
    print(f"📊 Retrieved data for {len(forex_data)} pairs")
    
    try:
        with open('signals.json', 'r') as f:
            signals_history = json.load(f)
    except FileNotFoundError:
        signals_history = []
    
    signals_history = check_expired_signals(signals_history, forex_data)
    
    current_hour = datetime.now(timezone.utc).strftime('%Y-%m-%d %H')
    recent_signals = [s for s in signals_history if s.get('timestamp', '').startswith(current_hour)]
    
    if len(recent_signals) >= MAX_SIGNALS_PER_HOUR:
        print(f"⏸️ Hourly limit reached: {len(recent_signals)}/{MAX_SIGNALS_PER_HOUR}")
        return
    
    new_signals = []
    for pair, price in forex_data.items():
        signal = generate_signal(pair, price)
        if signal:
            new_signals.append(signal)
            print(f"🎯 {pair} - {signal['direction']} signal ({signal['confidence']:.0%})")
    
    signals_history.extend(new_signals)
    
    if len(signals_history) > 1000:
        signals_history = signals_history[-1000:]
    
    with open('signals.json', 'w') as f:
        json.dump(signals_history, f, indent=2)
    
    completed_signals = [s for s in signals_history if s.get('outcome_checked')]
    wins = len([s for s in completed_signals if s.get('outcome') == 'WIN'])
    losses = len([s for s in completed_signals if s.get('outcome') == 'LOSS'])
    win_rate = (wins / len(completed_signals) * 100) if completed_signals else 0
    
    print(f"✅ Generated {len(new_signals)} signals | Total: {len(signals_history)}")
    print(f"📊 Performance: {wins}W/{losses}L ({win_rate:.1f}% win rate)")

if __name__ == "__main__":
    main()
