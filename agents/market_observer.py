import pandas as pd
import numpy as np
import talib
from datetime import datetime, timedelta
from ..config import settings
from ..utils.telegram import send_message
import requests
import json
from pathlib import Path

class MarketObserver:
    def __init__(self):
        self.market_status = "NORMAL"
        self.last_alert = datetime.min
        self.indicators = {}
        self.news_data = []
        self.news_path = Path(settings.BASE_DIR) / 'data' / 'market_news.json'
        self.news_path.parent.mkdir(parents=True, exist_ok=True)
        self.load_news_data()
    
    def load_news_data(self):
        """Load market news data from file"""
        if self.news_path.exists():
            try:
                with open(self.news_path, 'r') as f:
                    self.news_data = json.load(f)
            except:
                self.news_data = []
    
    def save_news_data(self):
        """Save market news data to file"""
        with open(self.news_path, 'w') as f:
            json.dump(self.news_data, f, indent=2)
    
    def fetch_market_news(self):
        """Fetch financial news from API (simplified)"""
        try:
            # In production, replace with actual news API
            # Example: newsapi.org, financialmodelingprep.com, etc.
            # For demo, we'll use mock data
            important_news = [
                {"title": "FED Rate Decision", "impact": "HIGH", "date": datetime.now().isoformat()},
                {"title": "CPI Data Release", "impact": "HIGH", "date": (datetime.now() - timedelta(days=1)).isoformat()}
            ]
            
            # Filter new news
            existing_titles = {n['title'] for n in self.news_data}
            new_news = [n for n in important_news if n['title'] not in existing_titles]
            
            if new_news:
                self.news_data.extend(new_news)
                self.save_news_data()
                return new_news
        except Exception as e:
            print(f"News fetch failed: {str(e)}")
        
        return []
    
    def analyze_market(self, df=None):
        """Analyze current market conditions"""
        current_time = datetime.now()
        
        # Check for scheduled events in next 30 mins
        upcoming_events = self.check_upcoming_events()
        if upcoming_events:
            self.market_status = "HIGH_VOLATILITY"
            if (current_time - self.last_alert) > timedelta(minutes=30):
                self.send_market_alert("Upcoming high-impact events detected")
                self.last_alert = current_time
            return self.market_status
        
        # If we have market data, analyze volatility
        if df is not None and len(df) > 50:
            # Calculate volatility indicators
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            current_atr = df['atr'].iloc[-1]
            avg_atr = df['atr'].mean()
            
            # Calculate ADX for trend strength
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            
            # Update indicators
            self.indicators = {
                'atr': current_atr,
                'avg_atr': avg_atr,
                'adx': adx,
                'volatility_ratio': current_atr / avg_atr if avg_atr > 0 else 1.0
            }
            
            # Check for abnormal volatility
            if self.indicators['volatility_ratio'] > 2.0:
                self.market_status = "HIGH_VOLATILITY"
                if (current_time - self.last_alert) > timedelta(minutes=30):
                    self.send_market_alert("Abnormal volatility detected")
                    self.last_alert = current_time
                return self.market_status
        
        # Reset to normal if no issues detected
        self.market_status = "NORMAL"
        return self.market_status
    
    def check_upcoming_events(self):
        """Check for upcoming high-impact events in next 30 mins"""
        now = datetime.now()
        next_30min = now + timedelta(minutes=30)
        
        # Fetch latest news
        new_news = self.fetch_market_news()
        high_impact_events = []
        
        # Check all news items
        for event in self.news_data:
            try:
                event_time = datetime.fromisoformat(event['date'])
                if now <= event_time <= next_30min and event.get('impact') == 'HIGH':
                    high_impact_events.append(event)
            except:
                continue
        
        return high_impact_events
    
    def send_market_alert(self, reason):
        """Send market condition alert via Telegram"""
        message = f"⚠️ MARKET ALERT: {reason}\n"
        message += f"Status: {self.market_status}\n"
        
        if self.indicators:
            message += "Current Indicators:\n"
            for k, v in self.indicators.items():
                message += f"- {k}: {v:.4f}\n"
        
        send_message(message)
        
        # Include upcoming events if any
        events = self.check_upcoming_events()
        if events:
            message = "📅 Upcoming Events:\n"
            for event in events:
                event_time = datetime.fromisoformat(event['date']).strftime("%H:%M")
                message += f"- {event_time} {event['title']} ({event['impact']})\n"
            send_message(message)
