import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ..config import settings
from ..utils.telegram import send_message
import matplotlib.pyplot as plt
import os
from pathlib import Path

class PerformanceAnalyzer:
    def __init__(self):
        self.trade_history = pd.DataFrame(columns=[
            'timestamp', 'asset', 'signal_type', 'direction', 
            'amount', 'payout', 'outcome', 'profit'
        ])
        self.performance_metrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'equity_curve': []
        }
        self.start_time = datetime.now()
        self.report_path = Path(settings.BASE_DIR) / 'reports'
        self.report_path.mkdir(parents=True, exist_ok=True)
    
    def add_trade(self, trade_result):
        """Add a new trade to history and update metrics"""
        # Add to history
        new_trade = {
            'timestamp': trade_result['timestamp'],
            'asset': trade_result['asset'],
            'signal_type': trade_result['signal_type'],
            'direction': trade_result['direction'],
            'amount': trade_result['amount'],
            'payout': trade_result['payout'],
            'outcome': trade_result['outcome'],
            'profit': trade_result['profit']
        }
        self.trade_history = pd.concat(
            [self.trade_history, pd.DataFrame([new_trade])], 
            ignore_index=True
        )
        
        # Update performance metrics
        self.update_metrics()
        
        # Save history to CSV
        self.save_history()
    
    def update_metrics(self):
        """Calculate key performance metrics"""
        if len(self.trade_history) == 0:
            return
            
        # Basic metrics
        wins = self.trade_history[self.trade_history['outcome'] == 'WIN']
        losses = self.trade_history[self.trade_history['outcome'] == 'LOSS']
        
        total_trades = len(self.trade_history)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit metrics
        total_profit = self.trade_history['profit'].sum()
        gross_profit = wins['profit'].sum()
        gross_loss = abs(losses['profit'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Equity curve
        cumulative_profit = self.trade_history['profit'].cumsum()
        peak = cumulative_profit.cummax()
        drawdown = (cumulative_profit - peak)
        max_drawdown = drawdown.min()
        
        # Risk-adjusted return (simplified Sharpe)
        avg_return = self.trade_history['profit'].mean()
        std_return = self.trade_history['profit'].std()
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Update metrics
        self.performance_metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_profit': total_profit,
            'equity_curve': cumulative_profit.tolist()
        }
    
    def save_history(self):
        """Save trade history to CSV"""
        history_file = self.report_path / 'trade_history.csv'
        self.trade_history.to_csv(history_file, index=False)
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        if len(self.trade_history) == 0:
            return "No trades yet"
            
        # Generate equity curve plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.trade_history['timestamp'], self.performance_metrics['equity_curve'])
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Profit')
        plt.grid(True)
        equity_curve_path = self.report_path / 'equity_curve.png'
        plt.savefig(equity_curve_path)
        plt.close()
        
        # Generate signal performance report
        signal_performance = self.trade_history.groupby('signal_type').agg(
            trades=('signal_type', 'count'),
            wins=('outcome', lambda x: (x == 'WIN').sum()),
            win_rate=('outcome', lambda x: (x == 'WIN').mean()),
            avg_profit=('profit', 'mean'),
            total_profit=('profit', 'sum')
        ).reset_index()
        
        # Format report
        report = f"📊 Performance Report\n"
        report += f"Runtime: {datetime.now() - self.start_time}\n"
        report += f"Total Trades: {self.performance_metrics['total_trades']}\n"
        report += f"Win Rate: {self.performance_metrics['win_rate']:.2%}\n"
        report += f"Total Profit: ${self.performance_metrics['total_profit']:.2f}\n"
        report += f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}\n"
        report += f"Max Drawdown: ${self.performance_metrics['max_drawdown']:.2f}\n"
        report += f"Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}\n\n"
        report += "Signal Performance:\n"
        
        for _, row in signal_performance.iterrows():
            report += (f"- {row['signal_type']}: {row['trades']} trades, "
                      f"{row['win_rate']:.2%} win rate, "
                      f"${row['total_profit']:.2f} profit\n")
        
        # Send report via Telegram
        send_message(report)
        send_message(photo=equity_curve_path)
        
        return report
    
    def detect_anomalies(self):
        """Detect performance anomalies"""
        if len(self.trade_history) < 20:
            return []
            
        # Detect losing streaks
        losses = (self.trade_history['outcome'] == 'LOSS').astype(int)
        loss_streak = losses * (losses.groupby((losses != losses.shift()).cumsum()).cumsum())
        current_streak = loss_streak.iloc[-1]
        
        anomalies = []
        if current_streak >= 5:
            anomalies.append({
                'type': 'LOSING_STREAK',
                'severity': 'HIGH' if current_streak > 7 else 'MEDIUM',
                'description': f"{current_streak} consecutive losses detected",
                'suggestion': "Review recent signals and market conditions"
            })
        
        # Detect drawdown
        current_drawdown = self.performance_metrics['equity_curve'][-1] - max(self.performance_metrics['equity_curve'])
        if current_drawdown < -0.3 * abs(self.performance_metrics['total_profit']):
            anomalies.append({
                'type': 'DRAWDOWN',
                'severity': 'HIGH',
                'description': f"Significant drawdown: ${current_drawdown:.2f}",
                'suggestion': "Reduce position size or pause trading"
            })
        
        return anomalies
