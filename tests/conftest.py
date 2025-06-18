import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture(scope="session")
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    if config_path.exists():
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        # Return minimal config for testing
        return {
            'train_config': {
                'general': {
                    'target': 'price_actual',
                    'drop_columns': ['valencia_snow_3h', 'madrid_snow_3h'],
                    'test_size': 0.2,
                    'random_seed': [42]
                }
            },
            'data_path': {
                'raw_data': 'data/raw/',
                'processed_data': 'data/processed/'
            }
        }


@pytest.fixture
def sample_weather_data():
    """Create sample weather data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'time': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        'generation_biomass': np.random.uniform(400, 500, n_samples),
        'generation_fossil_gas': np.random.uniform(4000, 6000, n_samples),
        'total_load_actual': np.random.uniform(20000, 30000, n_samples),
        'valencia_temp': np.random.uniform(15, 35, n_samples),
        'valencia_humidity': np.random.uniform(40, 90, n_samples),
        'valencia_temp_max': np.random.uniform(18, 38, n_samples),
        'valencia_rain': np.random.uniform(0, 5, n_samples),
        'madrid_temp': np.random.uniform(10, 30, n_samples),
        'madrid_humidity': np.random.uniform(30, 80, n_samples),
        'madrid_temp_max': np.random.uniform(13, 33, n_samples),
        'madrid_rain': np.random.uniform(0, 3, n_samples),
        'price_actual': np.random.uniform(40, 80, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def small_train_test_data():
    """Create small train/test datasets for testing."""
    np.random.seed(123)
    n_train, n_test = 50, 20
    
    base_features = {
        'feature1': np.random.normal(10, 2, n_train + n_test),
        'feature2': np.random.normal(5, 1, n_train + n_test),
        'temperature': np.random.uniform(15, 35, n_train + n_test),
        'humidity': np.random.uniform(40, 90, n_train + n_test),
    }
    
    train_data = pd.DataFrame({
        **{k: v[:n_train] for k, v in base_features.items()},
        'target': np.random.uniform(40, 80, n_train)
    })
    
    test_data = pd.DataFrame({
        k: v[n_train:] for k, v in base_features.items()
    })
    
    return train_data, test_data


@pytest.fixture(scope="session")
def test_data_paths():
    """Provide paths to test data files."""
    base_path = Path(__file__).parent.parent
    return {
        'train_csv': base_path / 'data' / 'raw' / 'train.csv',
        'test_csv': base_path / 'data' / 'raw' / 'test.csv',
        'config_yaml': base_path / 'config' / 'config.yaml'
    }