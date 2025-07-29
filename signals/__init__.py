from .rian import calculate_rian_signal
from .quantum import calculate_quantum_signal
from .golden import calculate_golden_signal
from .breakout import calculate_breakout_signal

def generate_all_signals(df):
    """
    Calculate all signals for the given DataFrame
    Returns:
        dict: Signals with values and indicators
    """
    signals = {}
    
    # Rian Signal
    rian_value, rian_indicators = calculate_rian_signal(df)
    signals['RIAN'] = {
        'value': rian_value,
        'indicators': rian_indicators
    }
    
    # Quantum Signal
    quantum_value, quantum_indicators = calculate_quantum_signal(df)
    signals['QUANTUM'] = {
        'value': quantum_value,
        'indicators': quantum_indicators
    }
    
    # Golden Signal
    golden_value, golden_indicators = calculate_golden_signal(df)
    signals['GOLDEN'] = {
        'value': golden_value,
        'indicators': golden_indicators
    }
    
    # Breakout Signal
    breakout_value, breakout_indicators = calculate_breakout_signal(df)
    signals['BREAKOUT'] = {
        'value': breakout_value,
        'indicators': breakout_indicators
    }
    
    return signals

def get_strongest_signal(signals):
    """
    Get the strongest valid signal based on priority
    Priority: Golden > Breakout > Quantum > Rian
    """
    # Signal priority
    priority_order = ['GOLDEN', 'BREAKOUT', 'QUANTUM', 'RIAN']
    
    for signal_type in priority_order:
        signal_data = signals.get(signal_type, {})
        if signal_data.get('value', 0) != 0:
            return signal_type, signal_data
    
    return None, {}
