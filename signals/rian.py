import talib
import numpy as np

def calculate_rian_signal(df):
    """
    Calculate Rian signal for OTC markets
    Args:
        df: DataFrame with OHLCV data and datetime index
    Returns:
        int: Signal value (-1, 0, 1)
        dict: Indicator values for AI analysis
    """
    # Calculate indicators
    df = df.copy()
    df['ema5'] = talib.EMA(df['close'], timeperiod=5)
    df['ema20'] = talib.EMA(df['close'], timeperiod=20)
    df['rsi6'] = talib.RSI(df['close'], timeperiod=6)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=8, slowperiod=18, signalperiod=6)
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    
    # Get the last candle
    last = df.iloc[-1]
    
    # Signal conditions
    buy_signal = (
        last['ema5'] > last['ema20'] and 
        last['rsi6'] < 70 and 
        last['macd'] > last['macd_signal']
    )
    
    sell_signal = (
        last['ema5'] < last['ema20'] and 
        last['rsi6'] > 30 and 
        last['macd'] < last['macd_signal']
    )
    
    # Determine signal value
    signal = 0
    if buy_signal:
        signal = 1
    elif sell_signal:
        signal = -1
        
    # Prepare indicator values for AI
    indicators = {
        'ema5': last['ema5'],
        'ema20': last['ema20'],
        'rsi': last['rsi6'],
        'macd': last['macd'] - last['macd_signal'],  # Histogram value
        'macd_raw': last['macd'],
        'macd_signal': last['macd_signal']
    }
    
    return signal, indicators
