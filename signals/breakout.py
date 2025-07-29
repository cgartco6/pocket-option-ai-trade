import talib
import numpy as np

def calculate_breakout_signal(df):
    """
    Calculate Breakout signal for OTC markets
    Args:
        df: DataFrame with OHLCV data and datetime index
    Returns:
        int: Signal value (-1, 0, 1)
        dict: Indicator values for AI analysis
    """
    df = df.copy()
    
    # Calculate indicators
    df['upper_band'] = talib.BBANDS(
        df['close'], 
        timeperiod=20, 
        nbdevup=2.0,  # Wider bands for OTC
        nbdevdn=2.0
    )[0]
    
    df['lower_band'] = talib.BBANDS(
        df['close'], 
        timeperiod=20, 
        nbdevup=2.0,
        nbdevdn=2.0
    )[2]
    
    df['vol_ma'] = df['volume'].rolling(10).mean()
    
    # Get the last candle
    last = df.iloc[-1]
    
    # Signal conditions
    buy_signal = (
        last['close'] > last['upper_band'] and 
        last['volume'] > last['vol_ma'] * 1.5
    )
    
    sell_signal = (
        last['close'] < last['lower_band'] and 
        last['volume'] > last['vol_ma'] * 1.5
    )
    
    # Determine signal value
    signal = 0
    if buy_signal:
        signal = 1
    elif sell_signal:
        signal = -1
        
    # Prepare indicator values for AI
    indicators = {
        'price_close': last['close'],
        'upper_band': last['upper_band'],
        'lower_band': last['lower_band'],
        'volume': last['volume'],
        'vol_ma': last['vol_ma'],
        'vol_ratio': last['volume'] / last['vol_ma'] if last['vol_ma'] > 0 else 1.0
    }
    
    return signal, indicators
