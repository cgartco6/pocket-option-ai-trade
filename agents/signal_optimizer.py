import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import joblib
import os
import json
from datetime import datetime
from pathlib import Path
from ..config import settings
from ..utils.telegram import send_message

class SignalOptimizer:
    def __init__(self):
        self.model_path = Path(settings.MODEL_PATH)
        self.data_path = Path(settings.DATA_PATH)
        self.model = None
        self.trade_data = pd.DataFrame()
        self.optimization_history = []
        self.load_model()
        self.load_data()
        
    def load_model(self):
        if self.model_path.exists():
            try:
                self.model = joblib.load(self.model_path)
                print(f"Loaded model from {self.model_path}")
            except Exception as e:
                print(f"Model loading failed: {str(e)}. Initializing new model.")
                self.initialize_model()
        else:
            self.initialize_model()
            print("Initialized new model")
    
    def initialize_model(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def load_data(self):
        if self.data_path.exists():
            try:
                self.trade_data = pd.read_csv(self.data_path)
                print(f"Loaded {len(self.trade_data)} trade records")
            except Exception as e:
                print(f"Trade data loading failed: {str(e)}. Starting fresh.")
                self.trade_data = pd.DataFrame(columns=[
                    'timestamp', 'asset', 'signal_type', 'signal_strength',
                    'ema5', 'ema20', 'rsi', 'macd', 'adx', 'atr', 'volatility',
                    'payout', 'outcome', 'profit', 'model_version'
                ])
        else:
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.trade_data = pd.DataFrame(columns=[
                'timestamp', 'asset', 'signal_type', 'signal_strength',
                'ema5', 'ema20', 'rsi', 'macd', 'adx', 'atr', 'volatility',
                'payout', 'outcome', 'profit', 'model_version'
            ])
    
    def add_trade_record(self, trade_info):
        """Add a new trade to the dataset"""
        if 'model_version' not in trade_info:
            trade_info['model_version'] = self.get_model_version()
            
        new_record = {
            'timestamp': datetime.now().isoformat(),
            'asset': trade_info['asset'],
            'signal_type': trade_info['signal_type'],
            'signal_strength': trade_info['signal_strength'],
            'ema5': trade_info['indicators'].get('ema5', 0),
            'ema20': trade_info['indicators'].get('ema20', 0),
            'rsi': trade_info['indicators'].get('rsi', 0),
            'macd': trade_info['indicators'].get('macd', 0),
            'adx': trade_info['indicators'].get('adx', 0),
            'atr': trade_info['indicators'].get('atr', 0),
            'volatility': trade_info['indicators'].get('volatility', 0),
            'payout': trade_info['payout'],
            'outcome': 1 if trade_info['outcome'] == 'WIN' else 0,
            'profit': trade_info['profit'],
            'model_version': trade_info['model_version']
        }
        
        self.trade_data = pd.concat([self.trade_data, pd.DataFrame([new_record])], 
                                    ignore_index=True)
        self.trade_data.to_csv(self.data_path, index=False)
        print(f"Added new trade record for {trade_info['asset']}")
        
        # Trigger model update if we have enough new data
        if len(self.trade_data) % settings.MODEL_UPDATE_FREQUENCY == 0:
            self.optimize_model()
    
    def get_model_version(self):
        """Get current model version based on file timestamp"""
        if self.model_path.exists():
            return datetime.fromtimestamp(self.model_path.stat().st_mtime).isoformat()
        return "initial"
    
    def optimize_model(self):
        """Retrain and optimize the model using all available data"""
        if len(self.trade_data) < settings.MIN_TRAINING_SAMPLES:
            print(f"Not enough data for optimization ({len(self.trade_data)} < {settings.MIN_TRAINING_SAMPLES})")
            return
        
        print(f"Starting model optimization with {len(self.trade_data)} samples")
        
        # Prepare data
        feature_cols = ['signal_strength', 'ema5', 'ema20', 'rsi', 
                        'macd', 'adx', 'atr', 'volatility', 'payout']
        X = self.trade_data[feature_cols]
        y = self.trade_data['outcome']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Hyperparameter optimization with Hyperopt
        def objective(params):
            model = GradientBoostingClassifier(
                n_estimators=int(params['n_estimators']),
                learning_rate=params['learning_rate'],
                max_depth=int(params['max_depth']),
                subsample=params['subsample'],
                random_state=42
            )
            
            # Use time-based split (oldest 80% for training, newest 20% for validation)
            split_index = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_index], X.iloc[split_index:]
            y_train, y_val = y.iloc[:split_index], y.iloc[split_index:]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            # Use combined metric (70% accuracy, 30% F1)
            loss = - (0.7 * acc + 0.3 * f1)
            
            return {
                'loss': loss,
                'status': STATUS_OK,
                'accuracy': acc,
                'f1': f1
            }
        
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
            max_evals=settings.HYPEROPT_EVALS,
            trials=trials,
            rstate=np.random.default_rng(42)
        
        # Get best trial results
        best_trial = trials.best_trial
        best_accuracy = -best_trial['result']['loss']  # Convert back to positive
        best_f1 = best_trial['result']['f1']
        
        # Train final model with best params on all data
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
        train_f1 = f1_score(y, train_pred)
        
        # Save optimization record
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(self.trade_data),
            'best_params': best,
            'validation_accuracy': best_accuracy,
            'validation_f1': best_f1,
            'training_accuracy': train_acc,
            'training_f1': train_f1
        }
        self.optimization_history.append(optimization_record)
        
        # Save history to file
        history_path = self.model_path.parent / 'optimization_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.optimization_history, f, indent=2)
        
        # Save model
        self.model = best_model
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Send notification
        message = (f"🤖 Model updated! Samples: {len(self.trade_data)}\n"
                  f"Val Acc: {best_accuracy:.2%} | Val F1: {best_f1:.2%}\n"
                  f"Train Acc: {train_acc:.2%} | Train F1: {train_f1:.2%}")
        send_message(message)
        
        return best_model
    
    def predict_success_probability(self, signal_data):
        """Predict success probability for a new signal"""
        if self.model is None or len(self.trade_data) < settings.MIN_PREDICTION_SAMPLES:
            return 0.5  # Default value
        
        try:
            # Prepare feature vector
            features = pd.DataFrame([{
                'signal_strength': signal_data['signal_strength'],
                'ema5': signal_data['indicators'].get('ema5', 0),
                'ema20': signal_data['indicators'].get('ema20', 0),
                'rsi': signal_data['indicators'].get('rsi', 0),
                'macd': signal_data['indicators'].get('macd', 0),
                'adx': signal_data['indicators'].get('adx', 0),
                'atr': signal_data['indicators'].get('atr', 0),
                'volatility': signal_data['indicators'].get('volatility', 0),
                'payout': signal_data['payout']
            }])
            
            # Fill missing values
            features = features.fillna(0)
            
            return self.model.predict_proba(features)[0][1]
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return 0.5
