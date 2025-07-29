import os

class Config:
    # Trading parameters
    TIMEFRAME = 5  # Minutes
    TRADE_DURATION = TIMEFRAME
    INVESTMENT_AMOUNT = 10  # USD
    MIN_PAYOUT = 0.92  # 92%
    MIN_SUCCESS_PROB = 0.65  # 65% confidence
    
    # AI model settings
    MODEL_PATH = "models/current_model.pkl"
    DATA_PATH = "data/trade_data.csv"
    MODEL_UPDATE_FREQUENCY = 50  # Update after every 50 trades
    MIN_TRAINING_SAMPLES = 100
    MIN_PREDICTION_SAMPLES = 20
    
    # System settings
    LOG_LEVEL = "INFO"
    
    # Path setup
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', MODEL_PATH)
    DATA_PATH = os.path.join(BASE_DIR, '..', DATA_PATH)

settings = Config()
