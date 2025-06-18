import pytest
import sys
import os
import pandas as pd
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_processing import load_data, preprocess_data, cal_discomfort, DataProcessor


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    data = {
        'temperature': [20.0, 25.0, 30.0, 15.0],
        'humidity': [60, 70, 80, 50],
        'valencia_temp': [22.0, 27.0, 32.0, 17.0],
        'valencia_humidity': [65, 75, 85, 55],
        'valencia_rain': [0.0, 0.1, 0.2, 0.0],
        'valencia_temp_max': [24.0, 29.0, 34.0, 19.0],
        'target_value': [100, 110, 120, 90]
    }
    return pd.DataFrame(data)


def test_load_data():
    """Test loading of actual data files."""
    train_data = load_data('data/raw/train.csv')
    assert train_data is not None
    assert len(train_data) > 0
    assert 'price_actual' in train_data.columns
    
    test_data = load_data('data/raw/test.csv')
    assert test_data is not None
    assert len(test_data) > 0


def test_cal_discomfort(sample_data):
    """Test discomfort index calculation."""
    result = cal_discomfort(sample_data.copy())
    
    # Check that discomfort columns are added
    assert 'discomfort' in result.columns
    assert 'valencia_discomfort1' in result.columns
    assert 'valencia_discomfort2' in result.columns
    
    # Check that discomfort values are calculated
    assert not result['discomfort'].isna().all()
    assert not result['valencia_discomfort1'].isna().all()


def test_preprocess_data_without_train_ref(sample_data):
    """Test preprocessing using same data for both test and train."""
    train_data = sample_data.copy()
    test_data = sample_data.copy()
    
    processed_test, processed_train = preprocess_data(test_data, train_data)
    
    assert processed_test is not None
    assert processed_train is not None
    assert processed_test.shape[0] == len(sample_data)
    assert processed_train.shape[0] == len(sample_data)
    
    # Check that scaled columns are added
    test_scaled_cols = [col for col in processed_test.columns if col.endswith('_scaled')]
    train_scaled_cols = [col for col in processed_train.columns if col.endswith('_scaled')]
    assert len(test_scaled_cols) > 0
    assert len(train_scaled_cols) > 0
    
    # Check that discomfort columns are added
    assert 'discomfort' in processed_test.columns
    assert 'discomfort' in processed_train.columns


def test_preprocess_data_with_train_ref(sample_data):
    """Test preprocessing with different test data normalized using train statistics."""
    train_data = sample_data.copy()
    test_data = sample_data.copy()
    
    # Modify test data to have different values
    test_data['temperature'] = test_data['temperature'] + 10
    
    # Process with train reference
    processed_test, processed_train = preprocess_data(test_data, train_data)
    
    assert processed_test is not None
    assert processed_train is not None
    
    # Check that the same scaled columns exist
    train_scaled_cols = [col for col in processed_train.columns if col.endswith('_scaled')]
    test_scaled_cols = [col for col in processed_test.columns if col.endswith('_scaled')]
    assert set(train_scaled_cols) == set(test_scaled_cols)
    
    # Check that test data was normalized using train statistics
    # The mean and std columns should be the same for both datasets
    for col in ['temperature', 'humidity']:
        if f'{col}_mean_to_t' in processed_test.columns:
            assert processed_test[f'{col}_mean_to_t'].iloc[0] == processed_train[f'{col}_mean_to_t'].iloc[0]
            assert processed_test[f'{col}_std_to_t'].iloc[0] == processed_train[f'{col}_std_to_t'].iloc[0]


def test_data_processor_integration():
    """Test DataProcessor class integration."""
    config = {
        'train_config': {
            'general': {
                'target': 'price_actual',
                'drop_columns': ['valencia_snow_3h', 'madrid_snow_3h']
            }
        },
        'test_config': {
            'general': {
                'drop_columns': ['valencia_snow_3h', 'madrid_snow_3h']
            }
        }
    }
    
    processor = DataProcessor(
        train_file_path=Path('data/raw/train.csv'),
        test_file_path=Path('data/raw/test.csv'),
        config=config
    )
    
    train_processed, test_processed = processor.process()
    
    assert train_processed is not None
    assert test_processed is not None
    assert len(train_processed) > 0
    assert len(test_processed) > 0
    
    # Check that target column is preserved in training data
    assert 'price_actual' in train_processed.columns
    # Check that target column is not in test data
    assert 'price_actual' not in test_processed.columns


@pytest.mark.unit
def test_preprocess_data_edge_cases():
    """Test edge cases in preprocessing."""
    # Test with empty dataframe
    empty_df = pd.DataFrame()
    test_result, train_result = preprocess_data(empty_df.copy(), empty_df.copy())
    assert test_result is not None
    assert train_result is not None
    
    # Test with single row
    single_row = pd.DataFrame({'temp': [25.0], 'humidity': [60]})
    test_result, train_result = preprocess_data(single_row.copy(), single_row.copy())
    assert len(test_result) == 1
    assert len(train_result) == 1
    
    # Test with constant values (std = 0)
    constant_data = pd.DataFrame({
        'constant_col': [10.0, 10.0, 10.0],
        'varying_col': [1.0, 2.0, 3.0]
    })
    test_result, train_result = preprocess_data(constant_data.copy(), constant_data.copy())
    # Should handle division by zero gracefully
    assert test_result is not None
    assert train_result is not None
    assert 'constant_col_scaled' in test_result.columns
    assert 'constant_col_scaled' in train_result.columns