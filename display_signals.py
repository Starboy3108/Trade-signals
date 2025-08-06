# display_signals.py - View AI Trading Results
import json
import pandas as pd
from datetime import datetime, timezone

def display_latest_signals():
    """Display latest trading signals in a readable format"""
    try:
        with open('live_signals.json', 'r') as f:
            signals = json.load(f)
        
        if not signals:
            print("No signals generated yet.")
            return
        
        # Get latest signals (last 10)
        latest_signals = signals[-10:]
        
        print("🎯 LATEST AI TRADING SIGNALS FOR POCKET OPTION")
        print("=" * 60)
        
        for signal in latest_signals:
            confidence_emoji = "💎" if signal['confidence'] > 0.9 else "⭐" if signal['confidence'] > 0.85 else "💪"
            direction_emoji = "📈" if signal['direction'] == "CALL" else "📉"
            
            print(f"{confidence_emoji} {direction_emoji} {signal['pair']} - {signal['direction']}")
            print(f"   Confidence: {signal['confidence']:.1%}")
            print(f"   Entry Price: {signal['entry_price']:.5f}")
            print(f"   Expiry: 5 minutes ({signal['expiry_time'][11:19]})")
            print(f"   Reasoning: {signal['reasoning']}")
            print(f"   Time: {signal['timestamp'][11:19]} UTC")
            print("-" * 40)
        
    except FileNotFoundError:
        print("No signals file found yet. System will create it on first run.")

def display_performance():
    """Display AI performance statistics"""
    try:
        with open('performance_history.json', 'r') as f:
            performance = json.load(f)
        
        if performance:
            latest = performance[-1]
            print(f"🧠 AI PERFORMANCE SUMMARY")
            print(f"Win Rate: {latest['win_rate']:.1f}%")
            print(f"Total Trades Learned: {latest['total_trades']}")
            print(f"Model Version: {latest['model_version']}")
            print(f"Last Updated: {latest['timestamp'][11:19]} UTC")
        else:
            print("No performance data yet. AI is still learning...")
            
    except FileNotFoundError:
        print("No performance file found yet.")

if __name__ == "__main__":
    display_latest_signals()
    print()
    display_performance()
