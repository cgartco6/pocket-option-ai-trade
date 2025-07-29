import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp, Trials
import joblib
import os
from datetime import datetime
from ..config import settings

class SignalOptimizer:
    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.data_path = settings.DATA_PATH
        self.model = None
        self.trade_data = pd.DataFrame()
        self.load_model()
        self.load_data()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                print(f"Loaded model from {self.model_path}")
            except:
                print("Model loading failed. Initializing new model.")
                self.initialize_model()
        else:
            self.initialize_model()
    
    def initialize_model(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        print("Initialized new model")
    
    def load_data(self):
        if os.path.exists(self.data_path):
            try:
                self.trade_data = pd.read_csv(self.data_path)
                print(f"Loaded {len(self.trade_data)} trade records")
            except:
                print("Trade data loading failed. Starting fresh.")
                self.trade_data = pd.DataFrame(columns=[
                    'timestamp', 'asset', 'signal_type', 'signal_strength',
                    'ema5', 'ema20', 'rsi', 'macd', 'adx', 'atr', 'volatility',
                    'payout', 'outcome', 'profit'
                ])
    
    def add_trade_record(self, trade_info):
        """Add a new trade to the dataset"""
        new_record = {
            'timestamp': datetime.now().isoformat(),
            'asset': trade_info['asset'],
            'signal_type': trade_info['signal_type'],
            'signal_strength': trade_info['signal_strength'],
            'ema5': trade_info['indicators']['ema5'],
            'ema20': trade_info['indicators']['ema20'],
            'rsi': trade_info['indicators']['rsi'],
            'macd': trade_info['indicators']['macd'],
            'adx': trade_info['indicators']['adx'],
            'atr': trade_info['indicators']['atr'],
            'volatility': trade_info['indicators']['volatility'],
            'payout': trade_info['payout'],
            'outcome': 1 if trade_info['outcome'] == 'WIN' else 0,
            'profit': trade_info['profit']
        }
        
        self.trade_data = self.trade_data.append(new_record, ignore_index=True)
        self.trade_data.to_csv(self.data_path, index=False)
        print(f"Added new trade record for {trade_info['asset']}")
        
        # Trigger model update if we have enough new data
        if len(self.trade_data) % settings.MODEL_UPDATE_FREQUENCY == 0:
            self.optimize_model()
    
    def optimize_model(self):
        """Retrain and optimize the model using all available data"""
        if len(self.trade_data) < settings.MIN_TRAINING_SAMPLES:
            print("Not enough data for optimization")
            return
        
        print(f"Starting model optimization with {len(self.trade_data)} samples")
        
        # Prepare data
        X = self.trade_data[['signal_strength', 'ema5', 'ema20', 'rsi', 
                             'macd', 'adx', 'atr', 'volatility', 'payout']]
        y = self.trade_data['outcome']
        
        # Hyperparameter optimization
        def objective(params):
            model = GradientBoostingClassifier(
                n_estimators=int(params['n_estimators']),
                learning_rate=params['learning_rate'],
                max_depth=int(params['max_depth']),
                subsample=params['subsample'],
                random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            return -accuracy_score(y_val, y_pred)  # Minimize negative accuracy
        
        space = {
            'n_estimators': hp.quniform('n_estimators', 50, 300, 25),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
            'max_depth': hp.quniform('max_depth', 3, 10, 1),
            'subsample': hp.uniform('subsample', 0.6, 1.0)
        }
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=50,
            trials=trials
        )
        
        # Train final model with best params
        best_model = GradientBoostingClassifier(
            n_estimators=int(best['n_estimators']),
            learning_rate=best['learning_rate'],
            max_depth=int(best['max_depth']),
            subsample=best['subsample'],
            random_state=42
        )
        best_model.fit(X, y)
        
        # Evaluate
        train_pred = best_model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        print(f"New model accuracy: {train_acc:.4f}")
        
        # Save model
        self.model = best_model
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Send notification
        from ..utils.telegram import send_message
        send_message(f"🤖 Model updated! Accuracy: {train_acc:.2%} "
                    f"| Samples: {len(self.trade_data)}")
    
    def predict_success_probability(self, signal_data):
        """Predict success probability for a new signal"""
        if len(self.trade_data) < settings.MIN_PREDICTION_SAMPLES:
            return 0.5  # Default value
        
        features = pd.DataFrame([{
            'signal_strength': signal_data['signal_strength'],
            'ema5': signal_data['indicators']['ema5'],
            'ema20': signal_data['indicators']['ema20'],
            'rsi': signal_data['indicators']['rsi'],
            'macd': signal_data['indicators']['macd'],
            'adx': signal_data['indicators']['adx'],
            'atr': signal_data['indicators']['atr'],
            'volatility': signal_data['indicators']['volatility'],
            'payout': signal_data['payout']
        }])
        
        return self.model.predict_proba(features)[0][1]
