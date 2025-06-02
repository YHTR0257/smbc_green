import pytest
from src.data_processing import load_data, preprocess_data

def test_load_data():
    # Test loading of data
    data = load_data('data/raw/sample_data.csv')
    assert data is not None
    assert len(data) > 0

def test_preprocess_data():
    # Test preprocessing of data
    raw_data = load_data('data/raw/sample_data.csv')
    processed_data = preprocess_data(raw_data)
    assert processed_data is not None
    assert 'processed_column' in processed_data.columns
    assert processed_data.shape[0] == len(raw_data)  # Ensure no rows are lost during processing