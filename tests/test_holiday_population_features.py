"""
休日・人口特徴量のテスト
Tests for holiday and population feature functionality
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing import DataProcessor


@pytest.fixture
def sample_holiday_data():
    """Create sample data with holiday periods."""
    # Create data spanning holiday periods
    dates = pd.date_range('2015-01-01', '2015-12-31', freq='D')
    
    data = {
        'time': dates,
        'price_actual': np.random.uniform(40, 80, len(dates)),
        'madrid_temp': np.random.uniform(10, 30, len(dates)),
        'valencia_temp': np.random.uniform(15, 35, len(dates))
    }
    
    return pd.DataFrame(data)


@pytest.fixture 
def temp_config_for_holidays():
    """Create temporary config files for holiday testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # City property config
        city_config = {
            'cities': {
                'coordinates': {
                    'Madrid': [40.4168, -3.7033],
                    'Barcelona': [41.3851, 2.1686],
                    'Valencia': [39.4699, -0.3756],
                    'Sevilla': [37.3891, -5.9845],
                    'Bilbao': [43.2627, -2.9253]
                }
            },
            'population_weights': {
                'admin_population': {
                    'Madrid': 3200000,
                    'Barcelona': 1600000,
                    'Valencia': 790000,
                    'Sevilla': 690000,
                    'Bilbao': 352000
                },
                'metro_population': {
                    'Madrid': 6500000,
                    'Barcelona': 5200000,
                    'Valencia': 1700000,
                    'Sevilla': 1400000,
                    'Bilbao': 950000
                }
            },
            'feature_engineering': {
                'interactions': {
                    'holiday_amplification_factor': 1.5
                }
            }
        }
        
        # Holiday config with test holidays
        holiday_config = {
            'national_holidays': {
                '2015': {
                    '2015-01-01': 'New Year\'s Day',
                    '2015-01-06': 'Epiphany',
                    '2015-04-03': 'Good Friday',
                    '2015-05-01': 'Labor Day',
                    '2015-12-25': 'Christmas Day'
                },
                '2016': {
                    '2016-01-01': 'New Year\'s Day',
                    '2016-01-06': 'Epiphany',
                    '2016-03-25': 'Good Friday',
                    '2016-05-01': 'Labor Day',
                    '2016-12-25': 'Christmas Day'
                }
            },
            'festivals': {}
        }
        
        # Write config files
        with open(config_dir / 'city_property.yml', 'w') as f:
            yaml.dump(city_config, f)
            
        with open(config_dir / 'holiday_festrival_config.yml', 'w') as f:
            yaml.dump(holiday_config, f)
            
        yield config_dir


@pytest.fixture
def holiday_data_processor(temp_config_for_holidays, sample_holiday_data):
    """Create DataProcessor for holiday testing."""
    # Save sample data to temporary files
    train_file = temp_config_for_holidays / 'train.csv'
    test_file = temp_config_for_holidays / 'test.csv'
    
    sample_holiday_data.to_csv(train_file, index=False)
    sample_holiday_data.iloc[:100].to_csv(test_file, index=False)
    
    config = {
        'population_type': 'admin',
        'train_config': {
            'general': {
                'target': 'price_actual',
                'drop_columns': []
            }
        }
    }
    
    processor = DataProcessor(
        train_file_path=train_file,
        test_file_path=test_file,
        config=config
    )
    
    # Update config_loader to use temp directory
    from config.config_loader import ConfigLoader
    processor.config_loader = ConfigLoader(config_dir=str(temp_config_for_holidays))
    processor.cities_config = processor.config_loader.get_cities_config()
    processor.feature_config = processor.config_loader.get_feature_engineering_config()
    processor.population_admin = processor.config_loader.get_population_config("admin")  
    processor.population_metro = processor.config_loader.get_population_config("metro")
    processor.population_weights = processor.population_admin
    
    return processor


class TestHolidayFeatures:
    """Test cases for holiday feature generation."""
    
    def test_add_holiday_features_basic(self, holiday_data_processor):
        """Test basic holiday feature addition."""
        processor = holiday_data_processor
        
        # Add holiday features
        processor.add_holiday_features()
        
        # Check that holiday columns were added
        assert 'is_national_holiday' in processor.train_data.columns
        assert 'holiday_name' in processor.train_data.columns
        assert 'is_national_holiday' in processor.test_data.columns
        assert 'holiday_name' in processor.test_data.columns
        
    def test_holiday_detection_accuracy(self, holiday_data_processor):
        """Test that holidays are correctly detected."""
        processor = holiday_data_processor
        processor.add_holiday_features()
        
        # Check specific known holidays
        train_data = processor.train_data
        
        # New Year's Day (2015-01-01)
        new_year_mask = train_data['time'].dt.date == pd.to_datetime('2015-01-01').date()
        if new_year_mask.any():
            assert train_data.loc[new_year_mask, 'is_national_holiday'].iloc[0] == 1
            assert 'New Year' in train_data.loc[new_year_mask, 'holiday_name'].iloc[0]
            
        # Christmas Day (2015-12-25)
        christmas_mask = train_data['time'].dt.date == pd.to_datetime('2015-12-25').date()
        if christmas_mask.any():
            assert train_data.loc[christmas_mask, 'is_national_holiday'].iloc[0] == 1
            assert 'Christmas' in train_data.loc[christmas_mask, 'holiday_name'].iloc[0]
            
    def test_non_holiday_detection(self, holiday_data_processor):
        """Test that non-holidays are correctly identified."""
        processor = holiday_data_processor
        processor.add_holiday_features()
        
        train_data = processor.train_data
        
        # Random date that should not be a holiday (2015-02-15)
        random_date_mask = train_data['time'].dt.date == pd.to_datetime('2015-02-15').date()
        if random_date_mask.any():
            assert train_data.loc[random_date_mask, 'is_national_holiday'].iloc[0] == 0
            assert train_data.loc[random_date_mask, 'holiday_name'].iloc[0] == ''
            
    def test_holiday_count_reasonable(self, holiday_data_processor):
        """Test that total holiday count is reasonable."""
        processor = holiday_data_processor
        processor.add_holiday_features()
        
        train_data = processor.train_data
        total_holidays = train_data['is_national_holiday'].sum()
        
        # Should have approximately 5 holidays for 2015 (as defined in test config)
        # Allow some tolerance for edge cases
        assert 3 <= total_holidays <= 10
        
    def test_holiday_features_without_time_column(self, holiday_data_processor):
        """Test holiday feature addition when time column is missing."""
        processor = holiday_data_processor
        
        # Remove time column
        processor.train_data = processor.train_data.drop('time', axis=1)
        processor.test_data = processor.test_data.drop('time', axis=1)
        
        # Should handle gracefully without crashing
        processor.add_holiday_features()
        
        # Should not add holiday columns if time is missing
        assert 'is_national_holiday' not in processor.train_data.columns


class TestPopulationFeatures:
    """Test cases for population feature generation."""
    
    def test_add_population_features_admin_type(self, holiday_data_processor):
        """Test population feature addition with admin population type."""
        processor = holiday_data_processor
        processor.population_type = 'admin'
        processor.population_weights = processor.population_admin
        
        processor.add_population_features()
        
        # Check that population columns were added
        expected_cities = ['madrid', 'barcelona', 'valencia', 'seville', 'bilbao']
        
        for city in expected_cities:
            weight_col = f'{city}_population_weight'
            raw_col = f'{city}_population_raw'
            
            assert weight_col in processor.train_data.columns
            assert raw_col in processor.train_data.columns
            assert weight_col in processor.test_data.columns
            assert raw_col in processor.test_data.columns
            
    def test_add_population_features_metro_type(self, holiday_data_processor):
        """Test population feature addition with metro population type."""
        processor = holiday_data_processor
        processor.population_type = 'metro'
        processor.population_weights = processor.population_metro
        
        processor.add_population_features()
        
        # Check that metro populations are used
        madrid_raw = processor.train_data['madrid_population_raw'].iloc[0]
        expected_madrid_metro = 6500000
        
        assert madrid_raw == expected_madrid_metro
        
    def test_population_weight_normalization(self, holiday_data_processor):
        """Test that population weights are correctly normalized."""
        processor = holiday_data_processor
        processor.add_population_features()
        
        # Get all population weight columns
        weight_cols = [col for col in processor.train_data.columns if col.endswith('_population_weight')]
        
        # Sum of all weights should be approximately 1.0
        total_weight = processor.train_data[weight_cols].iloc[0].sum()
        assert abs(total_weight - 1.0) < 0.001
        
    def test_population_weight_consistency(self, holiday_data_processor):
        """Test that population weights are consistent across all rows."""
        processor = holiday_data_processor
        processor.add_population_features()
        
        # Population weights should be the same for all rows
        weight_cols = [col for col in processor.train_data.columns if col.endswith('_population_weight')]
        
        for col in weight_cols:
            unique_values = processor.train_data[col].nunique()
            assert unique_values == 1  # Should have only one unique value
            
    def test_population_metadata_columns(self, holiday_data_processor):
        """Test that population metadata columns are added."""
        processor = holiday_data_processor
        processor.add_population_features()
        
        assert 'population_type' in processor.train_data.columns
        assert 'total_population' in processor.train_data.columns
        
        # Check values
        assert processor.train_data['population_type'].iloc[0] == 'admin'
        total_pop = processor.train_data['total_population'].iloc[0]
        expected_total = sum(processor.population_admin.values())
        assert total_pop == expected_total
        
    def test_population_admin_vs_metro_difference(self, holiday_data_processor):
        """Test that admin and metro populations are different."""
        processor = holiday_data_processor
        
        # Test admin population
        processor.population_type = 'admin'
        processor.population_weights = processor.population_admin
        processor.add_population_features()
        admin_madrid = processor.train_data['madrid_population_raw'].iloc[0]
        
        # Clear data and test metro population
        processor.train_data = processor.train_data.drop([col for col in processor.train_data.columns if 'population' in col], axis=1)
        processor.test_data = processor.test_data.drop([col for col in processor.test_data.columns if 'population' in col], axis=1)
        
        processor.population_type = 'metro'
        processor.population_weights = processor.population_metro
        processor.add_population_features()
        metro_madrid = processor.train_data['madrid_population_raw'].iloc[0]
        
        # Metro should be larger than admin
        assert metro_madrid > admin_madrid


class TestHolidayPopulationIntegration:
    """Test integration between holiday and population features."""
    
    def test_combined_feature_generation(self, holiday_data_processor):
        """Test generating both holiday and population features together."""
        processor = holiday_data_processor
        
        # Add both types of features
        processor.add_population_features()
        processor.add_holiday_features()
        
        # Check that both types of columns exist
        assert 'is_national_holiday' in processor.train_data.columns
        assert 'madrid_population_weight' in processor.train_data.columns
        
        # Check that we can identify holidays on populated data
        holiday_count = processor.train_data['is_national_holiday'].sum()
        assert holiday_count > 0
        
        # Check that population weights are still normalized
        weight_cols = [col for col in processor.train_data.columns if col.endswith('_population_weight')]
        total_weight = processor.train_data[weight_cols].iloc[0].sum()
        assert abs(total_weight - 1.0) < 0.001
        
    def test_feature_interaction_preparation(self, holiday_data_processor):
        """Test that features are ready for interaction calculations."""
        processor = holiday_data_processor
        
        processor.add_population_features()
        processor.add_holiday_features()
        
        # Verify we have the necessary columns for interactions
        train_data = processor.train_data
        
        # Should have holiday indicator
        assert 'is_national_holiday' in train_data.columns
        
        # Should have population weights for major cities
        for city in ['madrid', 'barcelona', 'valencia']:
            assert f'{city}_population_weight' in train_data.columns
            
        # Population weights should be numeric and in reasonable range
        for city in ['madrid', 'barcelona', 'valencia']:
            weight_col = f'{city}_population_weight'
            weights = train_data[weight_col]
            assert weights.dtype in [np.float64, np.float32]
            assert (weights >= 0).all()
            assert (weights <= 1).all()


class TestEdgeCases:
    """Test edge cases for holiday and population features."""
    
    def test_empty_dataframe_handling(self, holiday_data_processor):
        """Test handling of empty dataframes."""
        processor = holiday_data_processor
        
        # Create empty dataframes
        processor.train_data = pd.DataFrame()
        processor.test_data = pd.DataFrame()
        
        # Should not crash
        processor.add_population_features()
        processor.add_holiday_features()
        
    def test_missing_population_config(self):
        """Test handling when population config is missing."""
        config = {
            'population_type': 'admin',
            'train_config': {'general': {'target': 'price_actual'}}
        }
        
        # Create processor with missing config (should be handled in __init__)
        # This tests the error handling in the initialization
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
            pd.DataFrame({'time': [pd.Timestamp.now()], 'price_actual': [50]}).to_csv(tmp.name, index=False)
            
            try:
                processor = DataProcessor(
                    train_file_path=Path(tmp.name),
                    test_file_path=Path(tmp.name),
                    config=config
                )
                # Should handle missing config gracefully
                assert hasattr(processor, 'population_weights')
            finally:
                Path(tmp.name).unlink()
                
    def test_invalid_date_formats(self, holiday_data_processor):
        """Test handling of invalid date formats."""
        processor = holiday_data_processor
        
        # Create data with invalid time column
        invalid_data = pd.DataFrame({
            'time': ['invalid_date', 'another_invalid', '2015-01-01'],
            'price_actual': [50, 60, 70]
        })
        
        processor.train_data = invalid_data
        processor.test_data = invalid_data.copy()
        
        # Should handle gracefully without crashing
        try:
            processor.add_holiday_features()
        except Exception as e:
            # If it fails, it should be a known parsing error, not a crash
            assert 'time' in str(e).lower() or 'date' in str(e).lower()


@pytest.mark.unit
def test_population_normalization_math():
    """Test population normalization mathematics."""
    test_populations = {'A': 1000, 'B': 2000, 'C': 3000}
    total = sum(test_populations.values())  # 6000
    
    normalized = {city: pop/total for city, pop in test_populations.items()}
    
    assert abs(normalized['A'] - 1000/6000) < 0.001
    assert abs(normalized['B'] - 2000/6000) < 0.001  
    assert abs(normalized['C'] - 3000/6000) < 0.001
    assert abs(sum(normalized.values()) - 1.0) < 0.001


@pytest.mark.unit
def test_holiday_date_validation():
    """Test holiday date validation."""
    valid_dates = ['2015-01-01', '2015-12-25', '2016-01-01']
    
    for date_str in valid_dates:
        parsed = pd.to_datetime(date_str)
        assert parsed is not None
        assert parsed.year in [2015, 2016]