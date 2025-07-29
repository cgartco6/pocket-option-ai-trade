import talib
import pandas as pd
import numpy as np

def generate_signals(df, asset):
    # Calculate indicators
    df['ema5'] = talib.EMA(df['close'], timeperiod=5)
    df['ema20'] = talib.EMA(df['close'], timeperiod=20)
    df['rsi'] = talib.RSI(df['close'], timeperiod=6)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=8, slowperiod=18, signalperiod=6)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=20)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    
    # Calculate signals
    signals = {
        'RIAN': rian_signal(df),
        'QUANTUM': quantum_signal(df),
        'GOLDEN': golden_signal(df),
        'BREAKOUT': breakout_signal(df)
    }
    
    return signals

def rian_signal(df):
    # Rian Signal Conditions
    last = df.iloc[-1]
    if (last['ema5'] > last['ema20'] and 
        last['rsi'] < 70 and 
        last['macd'] > last['macd_signal']):
        return 1  # Buy
    elif (last['ema5'] < last['ema20'] and 
          last['rsi'] > 30 and 
          last['macd'] < last['macd_signal']):
        return -1  # Sell
    return 0

def quantum_signal(df):
    # Quantum Signal (ML-based)
    # This would be replaced with actual ML model in production
    # For now, a simple heuristic
    last = df.iloc[-1]
    if last['macd'] > 0 and last['rsi'] > 50:
        return 1
    elif last['macd'] < 0 and last['rsi'] < 50:
        return -1
    return 0

def golden_signal(df):
    # Golden Signal (Confirmation)
    last = df.iloc[-1]
    engulfing = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']).iloc[-1]
    harami = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']).iloc[-1]
    
    buy_condition = (
        (last['adx'] > 25) and
        (last['atr'] > df['atr'].rolling(50).mean().iloc[-1]) and
        (engulfing > 0 or harami > 0)
    )
    
    sell_condition = (
        (last['adx'] > 25) and
        (last['atr'] > df['atr'].rolling(50).mean().iloc[-1]) and
        (engulfing < 0 or harami < 0)
    )
    
    if buy_condition:
        return 1
    elif sell_condition:
        return -1
    return 0

def breakout_signal(df):
    # Breakout Signal
    last = df.iloc[-1]
    upper_band = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2.0)[0].iloc[-1]
    lower_band = talib.BBANDS(df['close'], timeperiod=20, nbdevdn=2.0)[2].iloc[-1]
    vol_ma = df['volume'].rolling(10).mean().iloc[-1]
    
    if last['close'] > upper_band and last['volume'] > vol_ma * 1.5:
        return 1
    elif last['close'] < lower_band and last['volume'] > vol_ma * 1.5:
        return -1
    return 0
