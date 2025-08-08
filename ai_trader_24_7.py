import json
from datetime import datetime, timezone, timedelta

def main():
    print("🚀 AI Trading System Started")
    
    current_time = datetime.now(timezone.utc)
    
    signal = {
        "timestamp": current_time.isoformat(),
        "pair": "EUR/USD",
        "direction": "CALL",
        "confidence": 0.85,
        "entry_price": 1.0850,
        "expiry_time": (current_time + timedelta(minutes=5)).isoformat(),
        "reasoning": "Test signal generated successfully"
    }
    
    signals = [signal]
    with open('signals.json', 'w') as f:
        json.dump(signals, f, indent=2)
    
    print(f"✅ Generated 1 test signal")
    print(f"Signal: {signal['pair']} {signal['direction']} ({signal['confidence']:.0%})")

if __name__ == "__main__":
    main()
    main()
    if minute % 10 == 0:
        score += 0.1
        conditions += 1
        reasoning.append("Pattern-based entry")
    
    if conditions >= 2 and score >= MIN_CONFIDENCE:
        return {
            "timestamp": current_time.isoformat(),
            "pair": pair,
            "direction": direction,
            "confidence": round(min(score, 0.95), 2),
            "entry_price": round(price, 5),
            "expiry_time": (current_time + timedelta(minutes=5)).isoformat(),
            "reasoning": ", ".join(reasoning),
            "rsi": round(rsi, 1),
            "momentum": round(momentum, 4),
            "volatility": round(volatility, 2),
            "conditions_met": f"{conditions}/6",
            "weekend_mode": weekday >= 5
        }
    
    return None

def main():
    current_time = datetime.now(timezone.utc)
    print(f"🚀 AI Trading Cycle: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    forex_data = get_forex_data()
    print(f"📊 Retrieved data for {len(forex_data)} pairs")
    for pair, price in forex_data.items():
        print(f"   {pair}: {price:.5f}")
    
    try:
        with open('signals.json', 'r') as f:
            signals_history = json.load(f)
    except FileNotFoundError:
        signals_history = []
    
    current_hour = current_time.strftime('%Y-%m-%d %H')
    recent_signals = [s for s in signals_history if s.get('timestamp', '').startswith(current_hour)]
    
    if len(recent_signals) >= MAX_SIGNALS_PER_HOUR:
        print(f"⏸️ Hourly limit reached: {len(recent_signals)}/{MAX_SIGNALS_PER_HOUR}")
        return
    
    new_signals = []
    for pair, price in forex_data.items():
        print(f"🔍 Analyzing {pair} at {price:.5f}...")
        signal = generate_signal(pair, price)
        if signal:
            new_signals.append(signal)
            print(f"🎯 SIGNAL GENERATED!")
            print(f"   {pair} - {signal['direction']} ({signal['confidence']:.0%})")
            print(f"   RSI: {signal['rsi']}, Conditions: {signal['conditions_met']}")
            print(f"   Reasoning: {signal['reasoning']}")
        else:
            print(f"❌ {pair} - No signal (conditions not met)")
    
    signals_history.extend(new_signals)
    
    if len(signals_history) > 1000:
        signals_history = signals_history[-1000:]
    
    with open('signals.json', 'w') as f:
        json.dump(signals_history, f, indent=2)
    
    print(f"✅ Generated {len(new_signals)} signals | Total: {len(signals_history)}")
    
    weekday = current_time.weekday()
    if weekday >= 5:
        print(f"📅 Weekend Mode Active - Enhanced signal generation enabled")
    
    if signals_history:
        recent_count = min(3, len(signals_history))
        print(f"📋 Last {recent_count} signals:")
        for i, sig in enumerate(signals_history[-recent_count:], 1):
            print(f"   {i}. {sig['pair']} {sig['direction']} ({sig['confidence']:.0%}) - {sig['timestamp'][11:19]} UTC")

if __name__ == "__main__":
    main()
        reasoning.append("London session active")
    elif 13 <= hour <= 21:  # New York session
        score += 0.2
        conditions += 1
        reasoning.append("New York session active")
    else:
        score += 0.1
        conditions += 1
        reasoning.append("Off-hours opportunity")
    
    # Time-based pattern bonus
    if minute % 10 == 0:  # Every 10 minutes
        score += 0.1
        conditions += 1
        reasoning.append("Pattern-based entry")
    
    # Generate signal with lower threshold
    if conditions >= 2 and score >= MIN_CONFIDENCE:
        return {
            "timestamp": current_time.isoformat(),
            "pair": pair,
            "direction": direction,
            "confidence": round(min(score, 0.95), 2),
            "entry_price": round(price, 5),
            "expiry_time": (current_time + timedelta(minutes=5)).isoformat(),
            "reasoning": ", ".join(reasoning),
            "rsi": round(rsi, 1),
            "momentum": round(momentum, 4),
            "volatility": round(volatility, 2),
            "conditions_met": f"{conditions}/6",
            "weekend_mode": weekday >= 5
        }
    
    return None

def main():
    current_time = datetime.now(timezone.utc)
    print(f"🚀 AI Trading Cycle: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    # Get forex data
    forex_data = get_forex_data()
    print(f"📊 Retrieved data for {len(forex_data)} pairs")
    for pair, price in forex_data.items():
        print(f"   {pair}: {price:.5f}")
    
    # Load existing signals
    try:
        with open('signals.json', 'r') as f:
            signals_history = json.load(f)
    except FileNotFoundError:
        signals_history = []
    
    # Check hourly limit
    current_hour = current_time.strftime('%Y-%m-%d %H')
    recent_signals = [s for s in signals_history if s.get('timestamp', '').startswith(current_hour)]
    
    if len(recent_signals) >= MAX_SIGNALS_PER_HOUR:
        print(f"⏸️ Hourly limit reached: {len(recent_signals)}/{MAX_SIGNALS_PER_HOUR}")
        return
    
    # Generate signals with detailed logging
    new_signals = []
    for pair, price in forex_data.items():
        print(f"🔍 Analyzing {pair} at {price:.5f}...")
        signal = generate_signal(pair, price)
        if signal:
            new_signals.append(signal)
            print(f"🎯 SIGNAL GENERATED!")
            print(f"   {pair} - {signal['direction']} ({signal['confidence']:.0%})")
            print(f"   RSI: {signal['rsi']}, Conditions: {signal['conditions_met']}")
            print(f"   Reasoning: {signal['reasoning']}")
        else:
            print(f"❌ {pair} - No signal (conditions not met)")
    
    # Update signals history
    signals_history.extend(new_signals)
    
    # Keep last 1000 signals
    if len(signals_history) > 1000:
        signals_history = signals_history[-1000:]
    
    # Save signals
    with open('signals.json', 'w') as f:
        json.dump(signals_history, f, indent=2)
    
    print(f"✅ Generated {len(new_signals)} signals | Total: {len(signals_history)}")
    
    # Show weekend status
    weekday = current_time.weekday()
    if weekday >= 5:
        print(f"📅 Weekend Mode Active - Enhanced signal generation enabled")
    
    # Display recent signals for debugging
    if signals_history:
        recent_count = min(3, len(signals_history))
        print(f"📋 Last {recent_count} signals:")
        for i, sig in enumerate(signals_history[-recent_count:], 1):
            print(f"   {i}. {sig['pair']} {sig['direction']} ({sig['confidence']:.0%}) - {sig['timestamp'][11:19]} UTC")

if __name__ == "__main__":
    main()
    
    new_signals = []
    for pair, price in forex_data.items():
        signal = generate_signal(pair, price)
        if signal:
            new_signals.append(signal)
            print(f"🎯 {pair} - {signal['direction']} signal ({signal['confidence']:.0%})")
        else:
            print(f"❌ {pair} - No signal generated")
    
    signals_history.extend(new_signals)
    
    if len(signals_history) > 1000:
        signals_history = signals_history[-1000:]
    
    with open('signals.json', 'w') as f:
        json.dump(signals_history, f, indent=2)
    
    print(f"✅ Generated {len(new_signals)} signals | Total: {len(signals_history)}")

if __name__ == "__main__":
    main()
