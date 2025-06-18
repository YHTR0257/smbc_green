import os
import dotenv
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import root_mean_squared_error, r2_score

import xgboost as xgb
from pathlib import Path
import warnings
import logging

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available for XGBoost."""
    try:
        # Try to create a simple DMatrix and train with GPU
        import numpy as np
        test_data = np.random.random((10, 5))
        test_labels = np.random.random(10)
        dtrain = xgb.DMatrix(test_data, label=test_labels)
        
        # Try GPU training
        params = {
            'tree_method': 'gpu_hist',
            'predictor': 'gpu_predictor',
            'objective': 'reg:squarederror'
        }
        
        xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
        logger.info("GPU is available for XGBoost")
        return True
    except Exception as e:
        logger.info(f"GPU not available for XGBoost: {e}")
        return False

class XGBoost():
    """XGBoost model with GPU/CPU support and time-based validation."""

    def __init__(self, train_config: dict):
        self.random_state = train_config.get('random_seed', [42])[0]
        self.config = train_config
        self.model = None
        self.validation_scores = []
        self.best_iteration = None
        self.use_gpu = self._determine_gpu_usage()
        
        logger.info(f"XGBoost initialized with {'GPU' if self.use_gpu else 'CPU'} mode")
    
    def _determine_gpu_usage(self):
        """Determine whether to use GPU based on config and availability."""
        gpu_setting = self.config.get('use_gpu', 'auto')
        
        if gpu_setting == 'false' or gpu_setting is False:
            logger.info("GPU disabled by configuration")
            return False
        elif gpu_setting == 'true' or gpu_setting is True:
            if check_gpu_availability():
                return True
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                return False
        else:  # auto
            return check_gpu_availability()


    def train(self, X_train, y_train, X_val, y_val):
        """Fit the model to the training data with optional validation."""
        
        # Development mode sampling
        if os.getenv("APP_ENV") == "development":
            sample_size = min(1000, len(X_train))
            X_train = X_train[:sample_size]
            y_train = y_train[:sample_size]
            val_sample_size = min(200, len(X_val))
            X_val = X_val[:val_sample_size]
            y_val = y_val[:val_sample_size]
        
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Prepare parameters with GPU/CPU configuration
        if self.use_gpu:
            tree_method = self.config.get('tree_method_gpu', 'gpu_hist')
            predictor = self.config.get('predictor_gpu', 'gpu_predictor')
            logger.info(f"Using GPU: tree_method={tree_method}, predictor={predictor}")
        else:
            tree_method = self.config.get('tree_method_cpu', 'hist')
            predictor = self.config.get('predictor_cpu', 'auto')
            logger.info(f"Using CPU: tree_method={tree_method}, predictor={predictor}")
        
        params = {
            'objective': self.config.get('objective', 'reg:squarederror'),
            'tree_method': tree_method,
            'predictor': predictor,
            'max_depth': self.config.get('max_depth', 6),
            'eta': self.config.get('eta', 0.1),
            'eval_metric': self.config.get('eval_metric', 'rmse'),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'random_state': self.random_state
        }
        
        # Set up validation if provided
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))
        
        # Training parameters
        num_boost_round = self.config.get('num_boost_round', 1000)
        early_stopping_rounds = self.config.get('early_stopping_rounds', 50) if X_val is not None else None
        verbose_eval = self.config.get('verbose_eval', 100)
        
        # Train the model with error handling for GPU fallback
        try:
            self.model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval
            )
            logger.info(f"Training completed successfully with {'GPU' if self.use_gpu else 'CPU'}")
        except Exception as e:
            if self.use_gpu:
                logger.warning(f"GPU training failed: {e}. Falling back to CPU...")
                # Fallback to CPU
                params['tree_method'] = self.config.get('tree_method_cpu', 'hist')
                params['predictor'] = self.config.get('predictor_cpu', 'auto')
                self.use_gpu = False
                
                self.model = xgb.train(
                    params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_round,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=verbose_eval
                )
                logger.info("Training completed successfully with CPU fallback")
            else:
                raise e
        
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
            
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)  # Use the trained model for predictions

    def score(self, X, y):
        """Return evaluation metrics on the given test data and labels."""
        predictions = self.predict(X)
        rmse = root_mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        mae = np.mean(np.abs(y - predictions))
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'best_iteration': self.best_iteration
        }
    
    def get_feature_importance(self, importance_type='weight'):
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.get_score(importance_type=importance_type)
    
    def save_model(self, filepath):
        """Save the trained model to file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """Load a trained model from file."""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        return self