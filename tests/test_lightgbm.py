import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import LightGBM, EnsembleModel, check_gpu_availability


class TestLightGBM:
    """Test LightGBM model functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
        
        # Split into train/validation
        split_idx = int(0.8 * n_samples)
        X_train = pd.DataFrame(X[:split_idx], columns=[f'feature_{i}' for i in range(n_features)])
        y_train = pd.Series(y[:split_idx], name='target')
        X_val = pd.DataFrame(X[split_idx:], columns=[f'feature_{i}' for i in range(n_features)])
        y_val = pd.Series(y[split_idx:], name='target')
        
        return X_train, y_train, X_val, y_val
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for LightGBM."""
        return {
            'objective': 'regression',
            'use_gpu': False,  # Use CPU for tests
            'max_depth': 6,
            'learning_rate': 0.1,
            'metric': 'rmse',
            'num_boost_round': 100,
            'bagging_fraction': 0.8,
            'feature_fraction': 0.8,
            'early_stopping_rounds': 10,
            'verbose_eval': 0,
            'random_state': 42
        }
    
    def test_lightgbm_initialization(self, basic_config):
        """Test LightGBM model initialization."""
        model = LightGBM(basic_config)
        
        assert model.config == basic_config
        assert model.random_state == 42
        assert model.validation_scores == []
        assert model.best_iteration is None
        assert model.best_params is None
        assert model.optimization_history == []
    
    def test_lightgbm_training(self, sample_data, basic_config):
        """Test basic LightGBM training."""
        X_train, y_train, X_val, y_val = sample_data
        model = LightGBM(basic_config)
        
        # Train the model
        model.train(X_train, y_train, X_val, y_val)
        
        # Check that model was trained
        assert model.model is not None
        assert hasattr(model.model, 'predict')
    
    def test_lightgbm_prediction(self, sample_data, basic_config):
        """Test LightGBM prediction."""
        X_train, y_train, X_val, y_val = sample_data
        model = LightGBM(basic_config)
        
        # Train the model
        model.train(X_train, y_train, X_val, y_val)
        
        # Make predictions
        predictions = model.predict(X_val)
        
        # Check predictions
        assert len(predictions) == len(X_val)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()
    
    def test_lightgbm_scoring(self, sample_data, basic_config):
        """Test LightGBM model scoring."""
        X_train, y_train, X_val, y_val = sample_data
        model = LightGBM(basic_config)
        
        # Train the model
        model.train(X_train, y_train, X_val, y_val)
        
        # Score the model
        scores = model.score(X_val, y_val)
        
        # Check scores
        assert 'rmse' in scores
        assert 'r2' in scores
        assert 'mae' in scores
        assert isinstance(scores['rmse'], float)
        assert isinstance(scores['r2'], float)
        assert isinstance(scores['mae'], float)
        assert scores['rmse'] > 0
        assert -1 <= scores['r2'] <= 1
        assert scores['mae'] > 0
    
    def test_feature_importance(self, sample_data, basic_config):
        """Test feature importance extraction."""
        X_train, y_train, X_val, y_val = sample_data
        model = LightGBM(basic_config)
        
        # Train the model
        model.train(X_train, y_train, X_val, y_val)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        # Check importance
        assert isinstance(importance, dict)
        assert len(importance) == X_train.shape[1]
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_model_save_load(self, sample_data, basic_config):
        """Test model saving and loading."""
        X_train, y_train, X_val, y_val = sample_data
        model = LightGBM(basic_config)
        
        # Train the model
        model.train(X_train, y_train, X_val, y_val)
        
        # Get predictions before saving
        predictions_before = model.predict(X_val)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            model_path = f.name
        
        try:
            model.save_model(model_path)
            
            # Load model
            new_model = LightGBM(basic_config)
            new_model.load_model(model_path)
            
            # Get predictions after loading
            predictions_after = new_model.predict(X_val)
            
            # Check that predictions are the same
            np.testing.assert_array_almost_equal(predictions_before, predictions_after, decimal=5)
            
        finally:
            # Clean up
            if os.path.exists(model_path):
                os.remove(model_path)
    
    def test_gpu_availability_check(self):
        """Test GPU availability checking."""
        # This test will pass regardless of GPU availability
        # Just checking that the function runs without error
        gpu_available = check_gpu_availability()
        assert isinstance(gpu_available, bool)
    
    def test_development_mode_sampling(self, sample_data, basic_config):
        """Test development mode data sampling."""
        X_train, y_train, X_val, y_val = sample_data
        model = LightGBM(basic_config)
        
        # Set development mode
        os.environ["APP_ENV"] = "development"
        
        try:
            # Train the model
            model.train(X_train, y_train, X_val, y_val)
            
            # Check that model was trained (should work with sampled data)
            assert model.model is not None
            
        finally:
            # Clean up environment variable
            if "APP_ENV" in os.environ:
                del os.environ["APP_ENV"]
    
    def test_cpu_fallback(self, sample_data):
        """Test CPU fallback when GPU fails."""
        X_train, y_train, X_val, y_val = sample_data
        
        # Force GPU usage that might fail
        config = {
            'objective': 'regression',
            'use_gpu': True,
            'max_depth': 6,
            'learning_rate': 0.1,
            'metric': 'rmse',
            'num_boost_round': 10,
            'early_stopping_rounds': 5,
            'verbose_eval': 0,
            'random_state': 42
        }
        
        model = LightGBM(config)
        
        # Train should succeed either with GPU or CPU fallback
        model.train(X_train, y_train, X_val, y_val)
        assert model.model is not None
    
    def test_optuna_configuration(self, basic_config):
        """Test Optuna configuration creation."""
        # Add Optuna config
        config = basic_config.copy()
        config['optuna'] = {
            'enabled': False,  # Disabled for testing
            'n_trials': 10,
            'timeout': 60,
            'direction': 'minimize',
            'pruner': 'median',
            'study_name': 'test_lightgbm',
            'search_space': {
                'max_depth': [3, 10],
                'learning_rate': [0.01, 0.3]
            }
        }
        
        model = LightGBM(config)
        assert model.config['optuna']['enabled'] == False


class TestEnsembleWithLightGBM:
    """Test ensemble functionality with LightGBM."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 500
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.1
        
        # Split into train/validation
        split_idx = int(0.8 * n_samples)
        X_train = pd.DataFrame(X[:split_idx], columns=[f'feature_{i}' for i in range(n_features)])
        y_train = pd.Series(y[:split_idx], name='target')
        X_val = pd.DataFrame(X[split_idx:], columns=[f'feature_{i}' for i in range(n_features)])
        y_val = pd.Series(y[split_idx:], name='target')
        
        return X_train, y_train, X_val, y_val
    
    @pytest.fixture
    def ensemble_config(self):
        """Configuration for ensemble with LightGBM."""
        return {
            'method': 'weighted_average',
            'optimize_weights': False,  # Disabled for testing
            'models': [
                {
                    'type': 'lightgbm',
                    'name': 'lgb_test',
                    'seeds': [42, 123]
                }
            ]
        }
    
    @pytest.fixture
    def model_configs(self):
        """Model configurations for ensemble."""
        return {
            'lightgbm': {
                'objective': 'regression',
                'use_gpu': False,
                'max_depth': 3,
                'learning_rate': 0.1,
                'metric': 'rmse',
                'num_boost_round': 50,
                'early_stopping_rounds': 10,
                'verbose_eval': 0
            }
        }
    
    def test_ensemble_initialization(self, ensemble_config):
        """Test ensemble initialization."""
        ensemble = EnsembleModel(ensemble_config)
        
        assert ensemble.config == ensemble_config
        assert ensemble.models == []
        assert ensemble.model_weights == []
        assert ensemble.model_names == []
        assert ensemble.ensemble_method == 'weighted_average'
    
    def test_ensemble_add_model(self, ensemble_config, sample_data, model_configs):
        """Test adding LightGBM model to ensemble."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble = EnsembleModel(ensemble_config)
        
        # Train a LightGBM model
        lightgbm_model = LightGBM(model_configs['lightgbm'])
        lightgbm_model.train(X_train, y_train, X_val, y_val)
        
        # Add to ensemble
        ensemble.add_model(lightgbm_model, weight=1.0, name='test_lightgbm')
        
        assert len(ensemble.models) == 1
        assert len(ensemble.model_weights) == 1
        assert len(ensemble.model_names) == 1
        assert ensemble.model_names[0] == 'test_lightgbm'
        assert ensemble.model_weights[0] == 1.0
    
    def test_ensemble_prediction(self, ensemble_config, sample_data, model_configs):
        """Test ensemble prediction with LightGBM models."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble = EnsembleModel(ensemble_config)
        
        # Train and add multiple LightGBM models
        for seed in [42, 123]:
            config = model_configs['lightgbm'].copy()
            config['random_state'] = seed
            
            model = LightGBM(config)
            model.train(X_train, y_train, X_val, y_val)
            ensemble.add_model(model, weight=1.0, name=f'lightgbm_{seed}')
        
        # Make ensemble predictions
        predictions = ensemble.predict(X_val)
        
        assert len(predictions) == len(X_val)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()
    
    def test_ensemble_scoring(self, ensemble_config, sample_data, model_configs):
        """Test ensemble scoring."""
        X_train, y_train, X_val, y_val = sample_data
        ensemble = EnsembleModel(ensemble_config)
        
        # Train and add LightGBM model
        model = LightGBM(model_configs['lightgbm'])
        model.train(X_train, y_train, X_val, y_val)
        ensemble.add_model(model, weight=1.0, name='lightgbm_test')
        
        # Score ensemble
        scores = ensemble.score(X_val, y_val)
        
        assert 'rmse' in scores
        assert 'r2' in scores
        assert 'mae' in scores
        assert 'n_models' in scores
        assert 'ensemble_method' in scores
        assert scores['n_models'] == 1
        assert scores['ensemble_method'] == 'weighted_average'


if __name__ == "__main__":
    pytest.main([__file__])