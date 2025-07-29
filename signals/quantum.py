import talib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from pathlib import Path

# Load model on import (will be cached)
try:
    model_path = Path(__file__).parent.parent / 'models' / 'quantum_model.pkl'
    QUANTUM_MODEL = joblib.load(model_path)
except:
    QUANTUM_MODEL = None
    print("Quantum model not found. Using fallback logic.")

def calculate_quantum_signal(df):
    """
    Calculate Quantum signal using ML predictions
    Args:
        df: DataFrame with OHLCV data and datetime index
    Returns:
        int: Signal value (-1, 0, 1)
        dict: Feature values for AI analysis
    """
    df = df.copy()
    
    # Feature engineering
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    df['volume_change'] = df['volume'].pct_change()
    df['rsi6'] = talib.RSI(df['close'], timeperiod=6)
    df['macd'], _, _ = talib.MACD(df['close'], fastperiod=8, slowperiod=18, signalperiod=6)
    
    # Create feature vector for last candle
    last = df.iloc[-1]
    features = pd.DataFrame([{
        'returns': last['returns'],
        'volatility': last['volatility'],
        'volume_change': last['volume_change'],
        'rsi': last['rsi6'],
        'macd': last['macd']
    }])
    
    # ML prediction if model available
    signal = 0
    if QUANTUM_MODEL is not None:
        try:
            prediction = QUANTUM_MODEL.predict(features)[0]
            signal = 1 if prediction == 1 else -1
        except Exception as e:
            print(f"Quantum prediction failed: {e}")
            signal = 0
    
    # Fallback heuristic if no model
    if signal == 0:
        if last['macd'] > 0 and last['rsi6'] > 50:
            signal = 1
        elif last['macd'] < 0 and last['rsi6'] < 50:
            signal = -1
    
    return signal, features.iloc[0].to_dict()
