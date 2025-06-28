import os
import dotenv
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import root_mean_squared_error, r2_score

import lightgbm as lgb
from pathlib import Path
import warnings
import logging


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
PYTORCH_AVAILABLE = True

import optuna
from optuna.integration.lightgbm import LightGBMPruningCallback
OPTUNA_AVAILABLE = True

dotenv.load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check if GPU is available for LightGBM."""
    try:
        # Try to create a simple Dataset and train with GPU
        import numpy as np
        test_data = np.random.random((10, 5))
        test_labels = np.random.random(10)
        dtrain = lgb.Dataset(test_data, label=test_labels)
        
        # Try GPU training
        params = {
            'device': 'gpu',
            'objective': 'regression',
            'verbosity': -1
        }
        
        lgb.train(params, dtrain, num_boost_round=1, valid_sets=[dtrain], callbacks=[lgb.log_evaluation(0)])
        logger.info("GPU is available for LightGBM")
        return True
    except Exception as e:
        logger.info(f"GPU not available for LightGBM: {e}")
        return False

def check_pytorch_gpu():
    """Check if GPU is available for PyTorch."""
    if not PYTORCH_AVAILABLE:
        logger.info("PyTorch not available")
        return False
    
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory if device_count > 0 else 0
            gpu_memory_gb = gpu_memory / (1024**3)
            
            logger.info(f"GPU available for PyTorch: {device_name} ({device_count} devices, {gpu_memory_gb:.1f}GB)")
            return True
        else:
            logger.info("GPU not available for PyTorch - using CPU")
            return False
    except Exception as e:
        logger.info(f"Error checking PyTorch GPU: {e}")
        return False

class LightGBM():
    """LightGBM model with GPU/CPU support and time-based validation."""

    def __init__(self, train_config: dict):
        self.random_state = train_config.get('random_seed', [42])[0]
        self.config = train_config
        self.validation_scores = []
        self.best_iteration = None
        self.use_gpu = self._determine_gpu_usage()
        self.best_params = None
        self.optimization_history = []
        
        logger.info(f"LightGBM initialized with {'GPU' if self.use_gpu else 'CPU'} mode")
        if OPTUNA_AVAILABLE:
            logger.info("Optuna available for hyperparameter optimization")
        else:
            logger.warning("Optuna not available - hyperparameter optimization disabled")
    
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
        
        # Create Dataset for training
        dtrain = lgb.Dataset(X_train, label=y_train)
        
        # Prepare parameters with GPU/CPU configuration
        if self.use_gpu:
            device = 'gpu'
            logger.info(f"Using GPU device")
        else:
            device = 'cpu'
            logger.info(f"Using CPU device")
        
        params = {
            'objective': self.config.get('objective', 'regression'),
            'device': device,
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'metric': self.config.get('metric', 'rmse'),
            'bagging_fraction': self.config.get('bagging_fraction', 0.8),
            'feature_fraction': self.config.get('feature_fraction', 0.8),
            'random_state': self.random_state,
            'verbosity': -1
        }
        
        # Set up validation if provided
        valid_sets = [dtrain]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append('validation')
        
        # Training parameters
        num_boost_round = self.config.get('num_boost_round', 1000)
        early_stopping_rounds = self.config.get('early_stopping_rounds', 50) if X_val is not None else None
        verbose_eval = self.config.get('verbose_eval', 100)
        
        # Train the model with error handling for GPU fallback
        try:
            callbacks = []
            if verbose_eval > 0:
                callbacks.append(lgb.log_evaluation(verbose_eval))
            if early_stopping_rounds is not None:
                callbacks.append(lgb.early_stopping(early_stopping_rounds))
            
            self.model: lgb.Booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=num_boost_round,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
            logger.info(f"Training completed successfully with {'GPU' if self.use_gpu else 'CPU'}")
        except Exception as e:
            if self.use_gpu:
                logger.warning(f"GPU training failed: {e}. Falling back to CPU...")
                # Fallback to CPU
                params['device'] = 'cpu'
                self.use_gpu = False
                
                self.model = lgb.train(
                    params=params,
                    train_set=dtrain,
                    num_boost_round=num_boost_round,
                    valid_sets=valid_sets,
                    valid_names=valid_names,
                    callbacks=callbacks
                )
                logger.info("Training completed successfully with CPU fallback")
            else:
                raise e
        
        if hasattr(self.model, 'best_iteration'):
            self.best_iteration = self.model.best_iteration
            
        return self

    def predict(self, X):
        """Predict using the fitted model."""
        return self.model.predict(X, num_iteration=self.model.best_iteration)

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
    
    def get_feature_importance(self, importance_type='split'):
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return dict(zip(self.model.feature_name(), self.model.feature_importance(importance_type=importance_type)))
    
    def save_model(self, filepath):
        """Save the trained model to file."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        self.model.save_model(filepath)
    
    def load_model(self, filepath):
        """Load a trained model from file."""
        self.model = lgb.Booster(model_file=filepath)
        return self
    
    def _create_optuna_study(self, optuna_config):
        """Create Optuna study for hyperparameter optimization."""
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Please install optuna to use hyperparameter optimization.")
        
        # Set up pruner
        pruner_name = optuna_config.get('pruner', 'median')
        if pruner_name == 'median':
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_name == 'successive_halving':
            pruner = optuna.pruners.SuccessiveHalvingPruner()
        elif pruner_name == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner()
        else:
            pruner = optuna.pruners.MedianPruner()
        
        # Create study
        study_name = optuna_config.get('study_name', 'lightgbm_optimization')
        direction = optuna_config.get('direction', 'minimize')
        storage = optuna_config.get('storage', None)
        
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            pruner=pruner,
            load_if_exists=True
        )
        
        return study
    
    def _create_objective(self, X_train, y_train, X_val, y_val, search_space):
        """Create objective function for Optuna optimization."""
        def objective(trial):
            # Suggest hyperparameters based on search space
            params = {
                'objective': self.config.get('objective', 'regression'),
                'metric': self.config.get('metric', 'rmse'),
                'random_state': self.random_state,
                'verbosity': -1
            }
            
            # Add GPU/CPU configuration
            if self.use_gpu:
                params['device'] = 'gpu'
            else:
                params['device'] = 'cpu'
            
            # Suggest hyperparameters
            for param, param_range in search_space.items():
                if param in ['max_depth', 'min_child_weight', 'num_leaves']:
                    params[param] = trial.suggest_int(param, param_range[0], param_range[1])
                else:
                    params[param] = trial.suggest_float(param, param_range[0], param_range[1])
            
            # Create Datasets
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Add pruning callback
            pruning_callback = LightGBMPruningCallback(trial, 'rmse', 'validation')
            
            # Train model
            model = lgb.train(
                params,
                train_set=dtrain,
                num_boost_round=self.config.get('num_boost_round', 1000),
                valid_sets=[dtrain, dval],
                valid_names=['train', 'validation'],
                callbacks=[lgb.early_stopping(self.config.get('early_stopping_rounds', 50)), pruning_callback]
            )
            
            # Get validation score
            predictions = model.predict(X_val, num_iteration=model.best_iteration)
            rmse = root_mean_squared_error(y_val, predictions)
            
            return rmse
        
        return objective
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Optimize hyperparameters using Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available - skipping hyperparameter optimization")
            return
        
        optuna_config = self.config.get('optuna', {})
        if not optuna_config.get('enabled', False):
            logger.info("Optuna optimization disabled in config")
            return
        
        logger.info("Starting hyperparameter optimization with Optuna...")
        
        # Development mode - reduce trials for faster testing
        if os.getenv("APP_ENV") == "development":
            n_trials = min(10, optuna_config.get('n_trials', 100))
            timeout = min(300, optuna_config.get('timeout', 3600))  # 5 minutes max
            sample_size = min(1000, len(X_train))
            X_train = X_train[:sample_size]
            y_train = y_train[:sample_size]
            val_sample_size = min(200, len(X_val))
            X_val = X_val[:val_sample_size]
            y_val = y_val[:val_sample_size]
            logger.info(f"Development mode: reduced to {n_trials} trials, {timeout}s timeout")
        else:
            n_trials = optuna_config.get('n_trials', 100)
            timeout = optuna_config.get('timeout', 3600)
        
        # Create study
        study = self._create_optuna_study(optuna_config)
        
        # Get search space
        search_space = optuna_config.get('search_space', {
            'max_depth': [3, 10],
            'learning_rate': [0.01, 0.3],
            'bagging_fraction': [0.6, 1.0],
            'feature_fraction': [0.6, 1.0],
            'num_leaves': [10, 300]
        })
        
        # Create objective function
        objective = self._create_objective(X_train, y_train, X_val, y_val, search_space)
        
        # Optimize
        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
            
            # Store results
            self.best_params = study.best_params
            self.optimization_history = [
                {'trial': trial.number, 'value': trial.value, 'params': trial.params}
                for trial in study.trials
            ]
            
            logger.info(f"Optimization completed. Best RMSE: {study.best_value:.4f}")
            logger.info(f"Best parameters: {study.best_params}")
            
            # Update config with best parameters
            for param, value in study.best_params.items():
                self.config[param] = value
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def get_optimization_results(self):
        """Get optimization results and history."""
        return {
            'best_params': self.best_params,
            'optimization_history': self.optimization_history
        }


class LSTMModel(nn.Module):
    """LSTM model for time series prediction with GPU support."""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.1, bidirectional=False):
        super(LSTMModel, self).__init__()
        
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use LSTM models.")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Fully connected layers
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.fc2 = nn.Linear(lstm_output_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output from LSTM
        if self.bidirectional:
            # For bidirectional LSTM, take the last output
            output = lstm_out[:, -1, :]
        else:
            output = lstm_out[:, -1, :]
        
        # Apply dropout and fully connected layers
        output = self.dropout(output)
        output = self.relu(self.fc1(output))
        output = self.dropout(output)
        output = self.fc2(output)
        
        return output


class LSTMTrainer:
    """LSTM trainer with GPU/CPU support and time-based validation."""
    
    def __init__(self, train_config: dict):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use LSTM models.")
        
        self.config = train_config
        self.device = self._determine_device()
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.training_history = []
        
        logger.info(f"LSTM Trainer initialized with device: {self.device}")
    
    def _determine_device(self):
        """Determine whether to use GPU or CPU."""
        gpu_setting = self.config.get('use_gpu', 'auto')
        
        if gpu_setting == 'false' or gpu_setting is False:
            logger.info("GPU disabled by configuration")
            return torch.device('cpu')
        elif gpu_setting == 'true' or gpu_setting is True:
            if check_pytorch_gpu():
                return torch.device('cuda')
            else:
                logger.warning("GPU requested but not available, falling back to CPU")
                return torch.device('cpu')
        else:  # auto
            if check_pytorch_gpu():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
    
    def prepare_sequences(self, X, y, sequence_length):
        """Prepare sequences for LSTM training."""
        sequences = []
        targets = []
        
        for i in range(len(X) - sequence_length + 1):
            seq = X.iloc[i:i + sequence_length].values
            target = y.iloc[i + sequence_length - 1] if isinstance(y, pd.Series) else y[i + sequence_length - 1]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def create_model(self, input_size, sequence_length):
        """Create LSTM model based on configuration."""
        hidden_size = self.config.get('hidden_size', 64)
        num_layers = self.config.get('num_layers', 2)
        output_size = self.config.get('output_size', 1)
        dropout = self.config.get('dropout', 0.1)
        bidirectional = self.config.get('bidirectional', False)
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=output_size,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        return model.to(self.device)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the LSTM model."""
        sequence_length = self.config.get('sequence_length', 24)
        batch_size = self.config.get('batch_size', 32)
        epochs = self.config.get('epochs', 100)
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Development mode sampling
        if os.getenv("APP_ENV") == "development":
            sample_size = min(1000, len(X_train))
            X_train = X_train[:sample_size]
            y_train = y_train[:sample_size]
            if X_val is not None and y_val is not None:
                val_sample_size = min(200, len(X_val))
                X_val = X_val[:val_sample_size]
                y_val = y_val[:val_sample_size]
            epochs = min(10, epochs)
        
        # Prepare sequences
        X_seq, y_seq = self.prepare_sequences(X_train, y_train, sequence_length)
        
        # Create model
        input_size = X_seq.shape[2]
        self.model = self.create_model(input_size, sequence_length)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_seq).to(self.device),
            torch.FloatTensor(y_seq).to(self.device)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Validation data preparation
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val, sequence_length)
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_seq).to(self.device),
                torch.FloatTensor(y_val_seq).to(self.device)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_loss = 0.0
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        output = self.model(data)
                        val_loss += self.criterion(output.squeeze(), target).item()
                val_loss /= len(val_loader)
                self.model.train()
            
            # Store history
            epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss if val_loader else None
            }
            self.training_history.append(epoch_info)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                if val_loader:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        logger.info("LSTM training completed successfully")
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        sequence_length = self.config.get('sequence_length', 24)
        
        # Prepare sequences
        X_seq, _ = self.prepare_sequences(X, pd.Series(range(len(X))), sequence_length)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            raw_predictions = self.model(X_tensor)
            raw_predictions = raw_predictions.cpu().numpy().squeeze()
        
        # Pad predictions to match input length (for ensemble compatibility)
        # LSTM can only predict for samples where we have enough history
        padded_predictions = np.full(len(X), np.nan)
        padded_predictions[sequence_length-1:sequence_length-1+len(raw_predictions)] = raw_predictions
        
        return padded_predictions
    
    def score(self, X, y):
        """Return evaluation metrics on the given test data and labels."""
        predictions = self.predict(X)
        
        # Find valid prediction range (where predictions are not NaN)
        valid_mask = ~np.isnan(predictions)
        
        if not valid_mask.any():
            return {
                'rmse': np.nan,
                'r2': np.nan,
                'mae': np.nan
            }
        
        # Use only valid predictions
        predictions_valid = predictions[valid_mask]
        if isinstance(y, pd.Series):
            y_valid = y.iloc[valid_mask]
        else:
            y_valid = y[valid_mask]
        
        rmse = root_mean_squared_error(y_valid, predictions_valid)
        r2 = r2_score(y_valid, predictions_valid)
        mae = np.mean(np.abs(y_valid - predictions_valid))
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae
        }
    
    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)
        
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Recreate model architecture
        # This requires knowing the input size, which should be stored in config
        input_size = checkpoint['config'].get('input_size')
        sequence_length = checkpoint['config'].get('sequence_length', 24)
        
        if input_size is None:
            raise ValueError("Input size not found in saved model config")
        
        self.model = self.create_model(input_size, sequence_length)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.training_history = checkpoint.get('training_history', [])
        
        logger.info(f"LSTM model loaded from {filepath}")
        return self


class EnsembleModel:
    """Ensemble model for combining multiple base models (XGBoost, LSTM, etc.)."""
    
    def __init__(self, ensemble_config: dict):
        self.config = ensemble_config
        self.models = []
        self.model_weights = []
        self.model_names = []
        self.ensemble_method = ensemble_config.get('method', 'average')
        self.validation_scores = {}
        
        logger.info(f"Ensemble model initialized with method: {self.ensemble_method}")
    
    def add_model(self, model, weight=1.0, name=None):
        """Add a trained model to the ensemble."""
        self.models.append(model)
        self.model_weights.append(weight)
        self.model_names.append(name or f"model_{len(self.models)}")
        
        logger.info(f"Added model '{self.model_names[-1]}' to ensemble with weight {weight}")
    
    def train_ensemble(self, X_train, y_train, X_val, y_val, model_configs):
        """Train multiple models and add them to the ensemble."""
        models_config = self.config.get('models', [])
        
        for i, model_config in enumerate(models_config):
            model_type = model_config.get('type', 'xgboost')
            model_name = model_config.get('name', f'{model_type}_{i}')
            seeds = model_config.get('seeds', [42])
            
            logger.info(f"Training {model_type} models with seeds: {seeds}")
            
            for seed_idx, seed in enumerate(seeds):
                # Update random seed in config
                if model_type == 'lightgbm':
                    config = model_configs['lightgbm'].copy()
                    config['random_state'] = seed
                    model = LightGBM(config)
                    model.train(X_train, y_train, X_val, y_val)
                    
                elif model_type == 'lstm':
                    if not PYTORCH_AVAILABLE:
                        logger.warning(f"PyTorch not available, skipping LSTM model")
                        continue
                    
                    config = model_configs['lstm'].copy()
                    # Set random seed for PyTorch
                    if PYTORCH_AVAILABLE:
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed(seed)
                    
                    model = LSTMTrainer(config)
                    model.train(X_train, y_train, X_val, y_val)
                
                elif model_type == 'xgboost':
                    logger.warning(f"XGBoost model type is deprecated, use 'lightgbm' instead")
                    continue
                else:
                    logger.warning(f"Unsupported model type: {model_type}")
                    continue
                
                # Add model to ensemble
                model_full_name = f"{model_name}_seed_{seed}"
                self.add_model(model, weight=1.0, name=model_full_name)
                
                # Evaluate individual model
                val_scores = model.score(X_val, y_val)
                self.validation_scores[model_full_name] = val_scores
                logger.info(f"Model {model_full_name} - Validation RMSE: {val_scores['rmse']:.4f}")
    
    def optimize_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation data."""
        from scipy.optimize import minimize
        
        if len(self.models) < 2:
            logger.warning("Need at least 2 models for weight optimization")
            return
        
        # Get predictions from all models
        predictions = []
        
        for model in self.models:
            pred = model.predict(X_val)
            predictions.append(pred)
        
        # Find valid prediction range (where all models can predict)
        # Remove NaN values and find common valid indices
        predictions_array = np.array(predictions)  # Shape: (n_models, n_samples)
        valid_mask = ~np.isnan(predictions_array).any(axis=0)  # True where all models have valid predictions
        
        if not valid_mask.any():
            logger.warning("No valid prediction range found for weight optimization")
            return
        
        # Use only valid predictions
        predictions_valid = predictions_array[:, valid_mask].T  # Shape: (n_valid_samples, n_models)
        
        # Align y_val to valid prediction range
        if isinstance(y_val, pd.Series):
            y_val_aligned = y_val.iloc[valid_mask]
        else:
            y_val_aligned = y_val[valid_mask]
        
        logger.info(f"Using {len(y_val_aligned)} valid samples for weight optimization")
        
        # Objective function to minimize (RMSE)
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize weights
            ensemble_pred = np.average(predictions_valid, axis=1, weights=weights)
            return root_mean_squared_error(y_val_aligned, ensemble_pred)
        
        # Initial weights (equal for all models)
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        # Optimize weights
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            self.model_weights = result.x.tolist()
            logger.info("Weight optimization completed successfully")
            for i, (name, weight) in enumerate(zip(self.model_names, self.model_weights)):
                logger.info(f"  {name}: {weight:.4f}")
        else:
            logger.warning("Weight optimization failed, using equal weights")
    
    def predict(self, X):
        """Make ensemble predictions."""
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = []
        valid_weights = []
        valid_names = []
        
        for i, model in enumerate(self.models):
            try:
                pred = model.predict(X)
                predictions.append(pred)
                valid_weights.append(self.model_weights[i])
                valid_names.append(self.model_names[i])
            except Exception as e:
                logger.warning(f"Model {self.model_names[i]} prediction failed: {e}")
                continue
        
        if not predictions:
            raise ValueError("All models failed to make predictions")
        
        # Handle NaN values - use only valid predictions for each sample
        predictions_array = np.array(predictions)  # Shape: (n_models, n_samples)
        valid_weights_array = np.array(valid_weights)
        
        # Prepare output array
        ensemble_predictions = np.full(predictions_array.shape[1], np.nan)
        
        for i in range(predictions_array.shape[1]):
            # Find models with valid predictions for this sample
            valid_mask = ~np.isnan(predictions_array[:, i])
            
            if valid_mask.any():
                valid_preds = predictions_array[valid_mask, i]
                valid_weights_sample = valid_weights_array[valid_mask]
                
                # Normalize weights for valid models only
                valid_weights_sample = valid_weights_sample / np.sum(valid_weights_sample)
                
                if self.ensemble_method == 'weighted_average':
                    ensemble_predictions[i] = np.average(valid_preds, weights=valid_weights_sample)
                elif self.ensemble_method == 'average':
                    ensemble_predictions[i] = np.mean(valid_preds)
                elif self.ensemble_method == 'median':
                    ensemble_predictions[i] = np.median(valid_preds)
                else:
                    ensemble_predictions[i] = np.mean(valid_preds)
        
        return ensemble_predictions
    
    def score(self, X, y):
        """Evaluate ensemble performance."""
        predictions = self.predict(X)
        
        # Find valid prediction range (where predictions are not NaN)
        valid_mask = ~np.isnan(predictions)
        
        if not valid_mask.any():
            logger.warning("No valid predictions found for evaluation")
            return {
                'rmse': np.nan,
                'r2': np.nan,
                'mae': np.nan,
                'n_models': len(self.models),
                'ensemble_method': self.ensemble_method
            }
        
        # Use only valid predictions
        predictions_valid = predictions[valid_mask]
        if isinstance(y, pd.Series):
            y_valid = y.iloc[valid_mask]
        else:
            y_valid = y[valid_mask]
        
        rmse = root_mean_squared_error(y_valid, predictions_valid)
        r2 = r2_score(y_valid, predictions_valid)
        mae = np.mean(np.abs(y_valid - predictions_valid))
        
        return {
            'rmse': rmse,
            'r2': r2,
            'mae': mae,
            'n_models': len(self.models),
            'ensemble_method': self.ensemble_method,
            'n_valid_samples': len(predictions_valid)
        }
    
    def save_ensemble(self, filepath):
        """Save ensemble configuration and model paths."""
        ensemble_data = {
            'config': self.config,
            'model_weights': self.model_weights,
            'model_names': self.model_names,
            'ensemble_method': self.ensemble_method,
            'validation_scores': self.validation_scores,
            'n_models': len(self.models)
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(ensemble_data, f, indent=2)
        
        logger.info(f"Ensemble configuration saved to {filepath}")
    
    def get_model_contributions(self, X):
        """Get individual model predictions and their contributions."""
        contributions = {}
        
        for i, (model, name, weight) in enumerate(zip(self.models, self.model_names, self.model_weights)):
            try:
                pred = model.predict(X)
                contributions[name] = {
                    'predictions': pred,
                    'weight': weight,
                    'contribution': pred * weight
                }
            except Exception as e:
                logger.warning(f"Failed to get contribution from {name}: {e}")
        
        return contributions