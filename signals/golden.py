import talib
import numpy as np

def calculate_golden_signal(df):
    """
    Calculate Golden signal (confirmation signal)
    Args:
        df: DataFrame with OHLCV data and datetime index
    Returns:
        int: Signal value (-1, 0, 1)
        dict: Indicator values for AI analysis
    """
    df = df.copy()
    
    # Calculate indicators
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=20)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
    df['atr_ma'] = df['atr'].rolling(50).mean()
    
    # Candlestick patterns
    df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
    df['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
    
    # Get the last candle
    last = df.iloc[-1]
    
    # Signal conditions
    buy_signal = (
        last['adx'] > 25 and
        last['atr'] > last['atr_ma'] and
        (last['engulfing'] > 0 or last['harami'] > 0)
    )
    
    sell_signal = (
        last['adx'] > 25 and
        last['atr'] > last['atr_ma'] and
        (last['engulfing'] < 0 or last['harami'] < 0)
    )
    
    # Determine signal value
    signal = 0
    if buy_signal:
        signal = 1
    elif sell_signal:
        signal = -1
        
    # Prepare indicator values for AI
    indicators = {
        'adx': last['adx'],
        'atr': last['atr'],
        'atr_ma': last['atr_ma'],
        'engulfing': last['engulfing'],
        'harami': last['harami'],
        'candle_pattern': 'engulfing' if last['engulfing'] != 0 else 'harami' if last['harami'] != 0 else 'none'
    }
    
    return signal, indicators
