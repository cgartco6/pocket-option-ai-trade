import time
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from config import settings
from utils.data_fetcher import PocketOptionAPI
from utils.telegram import send_message
from utils.trade_executor import execute_trade
from signals import generate_signals
from agents.signal_optimizer import SignalOptimizer
from agents.performance_analyzer import PerformanceAnalyzer
from agents.market_observer import MarketObserver

class TradingBot:
    def __init__(self):
        self.api = PocketOptionAPI(
            settings.EMAIL, 
            settings.PASSWORD,
            demo=True
        )
        self.signal_optimizer = SignalOptimizer()
        self.performance_analyzer = PerformanceAnalyzer()
        self.market_observer = MarketObserver()
        self.assets = []
        self.last_optimization = datetime.now()
        self.session_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'profit': 0.0
        }
    
    def run(self):
        send_message("🚀 OTC Trading Bot Activated")
        
        while True:
            try:
                # Refresh OTC assets every 6 hours
                if datetime.now().hour % 6 == 0 or not self.assets:
                    self.assets = self.api.get_otc_assets(
                        min_payout=settings.MIN_PAYOUT
                    )
                    send_message(f"🔄 Updated {len(self.assets)} OTC assets")
                
                # Check market conditions
                market_status = self.market_observer.analyze_market()
                if market_status != "NORMAL":
                    send_message(f"⚠️ Market alert: {market_status}")
                    time.sleep(settings.TIMEFRAME * 60)
                    continue
                
                # Process each asset
                for asset in self.assets:
                    try:
                        # Fetch data
                        df = self.api.get_historical_data(
                            asset, 
                            settings.TIMEFRAME,
                            count=500
                        )
                        if df is None or len(df) < 100:
                            continue
                        
                        # Generate signals
                        signals = generate_signals(df, asset)
                        
                        # Get the strongest signal
                        signal_type, signal_value = self.get_strongest_signal(signals)
                        if signal_value == 0:
                            continue
                        
                        # Prepare signal data for AI evaluation
                        signal_data = {
                            'asset': asset,
                            'signal_type': signal_type,
                            'signal_strength': signal_value,
                            'payout': self.api.get_payout(asset),
                            'indicators': {
                                'ema5': df['ema5'].iloc[-1],
                                'ema20': df['ema20'].iloc[-1],
                                'rsi': df['rsi'].iloc[-1],
                                'macd': df['macd'].iloc[-1],
                                'adx': df['adx'].iloc[-1],
                                'atr': df['atr'].iloc[-1],
                                'volatility': df['volatility'].iloc[-1]
                            }
                        }
                        
                        # Predict success probability
                        success_prob = self.signal_optimizer.predict_success_probability(signal_data)
                        
                        # Only trade if probability meets threshold
                        if success_prob >= settings.MIN_SUCCESS_PROB:
                            # Execute trade
                            trade_result = execute_trade(
                                self.api,
                                asset,
                                signal_type,
                                signal_value,
                                settings.INVESTMENT_AMOUNT,
                                settings.TRADE_DURATION
                            )
                            
                            if trade_result:
                                # Update session stats
                                self.session_stats['trades'] += 1
                                if trade_result['outcome'] == 'WIN':
                                    self.session_stats['wins'] += 1
                                    self.session_stats['profit'] += trade_result['profit']
                                else:
                                    self.session_stats['losses'] += 1
                                    self.session_stats['profit'] -= settings.INVESTMENT_AMOUNT
                                
                                # Add to AI training data
                                signal_data['outcome'] = trade_result['outcome']
                                signal_data['profit'] = trade_result['profit']
                                self.signal_optimizer.add_trade_record(signal_data)
                                
                                # Update performance analyzer
                                self.performance_analyzer.add_trade(trade_result)
                    except Exception as e:
                        send_message(f"❌ Error processing {asset}: {str(e)}")
                
                # Hourly performance report
                if datetime.now().minute == 0:
                    self.send_performance_report()
                
                # Daily model optimization
                if datetime.now().hour == 3 and datetime.now().minute < 5:
                    if self.last_optimization.date() < datetime.now().date():
                        self.signal_optimizer.optimize_model()
                        self.last_optimization = datetime.now()
                
                time.sleep(settings.TIMEFRAME * 60)
                
            except Exception as e:
                send_message(f"🔴 CRITICAL ERROR: {str(e)}")
                time.sleep(60)
    
    def get_strongest_signal(self, signals):
        """Select the strongest valid signal"""
        # Signal priority: Golden > Breakout > Quantum > Rian
        for signal_type in ['GOLDEN', 'BREAKOUT', 'QUANTUM', 'RIAN']:
            if signals.get(signal_type, 0) != 0:
                return signal_type, signals[signal_type]
        return None, 0
    
    def send_performance_report(self):
        """Send hourly performance update to Telegram"""
        win_rate = (self.session_stats['wins'] / self.session_stats['trades'] * 100 
                   if self.session_stats['trades'] > 0 else 0)
        
        message = (
            f"📊 <b>Hourly Performance Report</b>\n"
            f"Trades: {self.session_stats['trades']}\n"
            f"Wins: <font color='green'>{self.session_stats['wins']}</font> | "
            f"Losses: <font color='red'>{self.session_stats['losses']}</font>\n"
            f"Win Rate: {win_rate:.1f}%\n"
            f"Net Profit: <b>${self.session_stats['profit']:.2f}</b>"
        )
        send_message(message)
        
        # Reset session stats
        self.session_stats = {'trades':0, 'wins':0, 'losses':0, 'profit':0.0}

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
