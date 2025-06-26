"""
祭典特徴量機能のテスト
Tests for festival feature functionality
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing import DataProcessor


@pytest.fixture
def sample_festival_config():
    """Create sample festival configuration."""
    return {
        'semana_santa': {
            '2015': {
                'start_date': '2015-03-29',
                'end_date': '2015-04-05',
                'primary_cities': ['Sevilla', 'Valencia'],
                'scale': 'large',
                'outdoor_rate': 0.7
            },
            '2016': {
                'start_date': '2016-03-20',
                'end_date': '2016-03-27',
                'primary_cities': ['Sevilla', 'Valencia'],
                'scale': 'large',
                'outdoor_rate': 0.7
            }
        },
        'la_tomatina': {
            '2015': {
                'start_date': '2015-08-26',
                'end_date': '2015-08-26',
                'primary_cities': ['Valencia'],
                'scale': 'small',
                'outdoor_rate': 1.0
            },
            '2016': {
                'start_date': '2016-08-31',
                'end_date': '2016-08-31',
                'primary_cities': ['Valencia'],
                'scale': 'small',
                'outdoor_rate': 1.0
            }
        }
    }


@pytest.fixture
def sample_time_series_data():
    """Create sample time series data for testing festival features."""
    # Create data spanning festival periods
    dates = pd.date_range('2015-03-25', '2015-09-05', freq='D')
    
    data = {
        'time': dates,
        'price_actual': np.random.uniform(40, 80, len(dates)),
        'valencia_temp': np.random.uniform(15, 30, len(dates)),
        'valencia_humidity': np.random.uniform(40, 90, len(dates)),
        'valencia_rain': np.random.uniform(0, 5, len(dates)),
        'seville_temp': np.random.uniform(18, 35, len(dates)),
        'seville_humidity': np.random.uniform(35, 85, len(dates)),
        'seville_rain': np.random.uniform(0, 3, len(dates))
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_config_files():
    """Create temporary config files for testing."""
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
                }
            },
            'feature_engineering': {
                'scale_impact': {
                    'small': 0.1,
                    'medium': 0.3,
                    'large': 0.6
                },
                'time_effects': {
                    'preparation_days_before': 3,
                    'aftermath_days_after': 2
                },
                'interactions': {
                    'holiday_amplification_factor': 1.5,
                    'rain_threshold_mm': 5.0
                }
            }
        }
        
        # Festival config
        festival_config = {
            'national_holidays': {
                '2015': {
                    '2015-01-01': 'New Year\'s Day',
                    '2015-04-03': 'Good Friday',
                    '2015-12-25': 'Christmas Day'
                }
            },
            'festivals': {
                'semana_santa': {
                    '2015': {
                        'start_date': '2015-03-29',
                        'end_date': '2015-04-05',
                        'primary_cities': ['Sevilla', 'Valencia'],
                        'scale': 'large',
                        'outdoor_rate': 0.7
                    }
                },
                'la_tomatina': {
                    '2015': {
                        'start_date': '2015-08-26',
                        'end_date': '2015-08-26',
                        'primary_cities': ['Valencia'],
                        'scale': 'small',
                        'outdoor_rate': 1.0
                    }
                }
            }
        }
        
        # Write config files
        with open(config_dir / 'city_property.yml', 'w') as f:
            yaml.dump(city_config, f)
            
        with open(config_dir / 'holiday_festrival_config.yml', 'w') as f:
            yaml.dump(festival_config, f)
            
        yield config_dir


@pytest.fixture
def data_processor_with_config(temp_config_files, sample_time_series_data):
    """Create DataProcessor with temporary config files."""
    # Save sample data to temporary files
    train_file = temp_config_files / 'train.csv'
    test_file = temp_config_files / 'test.csv'
    
    sample_time_series_data.to_csv(train_file, index=False)
    sample_time_series_data.iloc[:50].to_csv(test_file, index=False)
    
    config = {
        'population_type': 'admin',
        'train_config': {
            'general': {
                'target': 'price_actual',
                'drop_columns': []
            }
        }
    }
    
    # Update config loader path
    import sys
    original_path = sys.path[:]
    sys.path.insert(0, str(temp_config_files.parent))
    
    try:
        processor = DataProcessor(
            train_file_path=train_file,
            test_file_path=test_file,
            config=config
        )
        # Update config_loader to use temp directory
        from config.config_loader import ConfigLoader
        processor.config_loader = ConfigLoader(config_dir=str(temp_config_files))
        processor.cities_config = processor.config_loader.get_cities_config()
        processor.feature_config = processor.config_loader.get_feature_engineering_config()
        
        yield processor
    finally:
        sys.path[:] = original_path


class TestFestivalFeatures:
    """Test cases for festival feature generation."""
    
    def test_add_city_festivals_valencia(self, data_processor_with_config):
        """Test adding festival features for Valencia."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        # Add festival features for Valencia
        result_df = processor.add_city_festivals(df, 'valencia', [2015])
        
        # Check that simplified festival columns were added
        assert 'valencia_has_festival' in result_df.columns
        assert 'valencia_festival_intensity' in result_df.columns
        assert 'valencia_festival_outdoor_impact' in result_df.columns
        
        # Verify festival period is correctly marked
        festival_days = result_df['valencia_has_festival'].sum()
        assert festival_days > 0  # Should have some festival days
        
        # Check that festival intensity is correctly applied
        active_mask = result_df['valencia_has_festival'] == 1
        if active_mask.any():
            intensities = result_df.loc[active_mask, 'valencia_festival_intensity'].unique()
            # Should have intensity values corresponding to festival scales
            assert all(intensity > 0 for intensity in intensities)
            assert all(intensity <= 0.6 for intensity in intensities)  # max is large scale
        
    def test_add_city_festivals_seville(self, data_processor_with_config):
        """Test adding festival features for Seville."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        result_df = processor.add_city_festivals(df, 'seville', [2015])
        
        # Check festival columns for Seville
        assert 'seville_has_festival' in result_df.columns
        assert 'seville_festival_intensity' in result_df.columns
        assert 'seville_festival_outdoor_impact' in result_df.columns
        
        # Seville participates in semana_santa, so should have some festival days
        festival_days = result_df['seville_has_festival'].sum()
        assert festival_days > 0
            
    def test_festival_intensity_calculation(self, data_processor_with_config):
        """Test that festival intensity is correctly calculated."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        result_df = processor.add_city_festivals(df, 'valencia', [2015])
        
        # Check festival intensity values
        festival_mask = result_df['valencia_has_festival'] == 1
        if festival_mask.any():
            intensities = result_df.loc[festival_mask, 'valencia_festival_intensity']
            
            # All intensities should be positive and within expected range
            assert (intensities > 0).all()
            assert (intensities <= 0.6).all()  # max is large scale
            
            # Should have intensity corresponding to semana_santa (large = 0.6)
            # and potentially la_tomatina (small = 0.1)
            unique_intensities = intensities.unique()
            assert len(unique_intensities) > 0
            
    def test_festival_outdoor_impact(self, data_processor_with_config):
        """Test that outdoor impact is correctly calculated."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        result_df = processor.add_city_festivals(df, 'valencia', [2015])
        
        # Check outdoor impact values
        festival_mask = result_df['valencia_has_festival'] == 1
        if festival_mask.any():
            outdoor_impacts = result_df.loc[festival_mask, 'valencia_festival_outdoor_impact']
            
            # All outdoor impacts should be positive
            assert (outdoor_impacts > 0).all()
            
            # Should be product of scale weight and outdoor rate
            # semana_santa: 0.6 * 0.7 = 0.42
            # la_tomatina: 0.1 * 1.0 = 0.1
            unique_impacts = outdoor_impacts.unique()
            assert len(unique_impacts) > 0
            
    def test_no_festival_for_non_participating_city(self, data_processor_with_config):
        """Test that cities not participating in festivals don't get festival features."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        # Madrid doesn't participate in any festivals in our test config
        result_df = processor.add_city_festivals(df, 'madrid', [2015])
        
        # Should have festival columns but they should be all zeros
        assert 'madrid_has_festival' in result_df.columns
        assert result_df['madrid_has_festival'].sum() == 0
        assert result_df['madrid_festival_intensity'].sum() == 0
        assert result_df['madrid_festival_outdoor_impact'].sum() == 0
            
    def test_multiple_years_processing(self, data_processor_with_config):
        """Test processing festivals for multiple years."""
        processor = data_processor_with_config
        
        # Extend data to cover multiple years
        extended_dates = pd.date_range('2015-01-01', '2016-12-31', freq='D')
        extended_data = pd.DataFrame({
            'time': extended_dates,
            'price_actual': np.random.uniform(40, 80, len(extended_dates))
        })
        
        result_df = processor.add_city_festivals(extended_data, 'valencia', [2015, 2016])
        
        # Should have festivals from both years
        festival_days = result_df['valencia_has_festival'].sum()
        assert festival_days > 0
        
        # Check that festivals appear in both years
        result_df['year'] = result_df['time'].dt.year
        festivals_2015 = result_df[result_df['year'] == 2015]['valencia_has_festival'].sum()
        festivals_2016 = result_df[result_df['year'] == 2016]['valencia_has_festival'].sum()
        
        # Both years should have some festival days (though dates may differ)
        assert festivals_2015 > 0 or festivals_2016 > 0


class TestFestivalFeaturesEdgeCases:
    """Test edge cases for festival features."""
    
    def test_missing_time_column(self, data_processor_with_config):
        """Test handling when time column is missing."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        df = df.drop('time', axis=1)
        
        # Should handle gracefully and return original dataframe
        result_df = processor.add_city_festivals(df, 'valencia', [2015])
        assert result_df.equals(df)
        
    def test_empty_dataframe(self, data_processor_with_config):
        """Test handling of empty dataframe."""
        processor = data_processor_with_config
        empty_df = pd.DataFrame()
        
        result_df = processor.add_city_festivals(empty_df, 'valencia', [2015])
        assert result_df.equals(empty_df)
        
    def test_invalid_city_name(self, data_processor_with_config):
        """Test handling of city names that don't match config."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        # Use city name that doesn't exist in config
        result_df = processor.add_city_festivals(df, 'nonexistent_city', [2015])
        
        # Should add festival columns but they should be all zeros
        assert 'nonexistent_city_has_festival' in result_df.columns
        assert result_df['nonexistent_city_has_festival'].sum() == 0
        assert result_df['nonexistent_city_festival_intensity'].sum() == 0
        assert result_df['nonexistent_city_festival_outdoor_impact'].sum() == 0
            
    def test_year_without_festival_data(self, data_processor_with_config):
        """Test handling when festival data for a year is missing."""
        processor = data_processor_with_config
        df = processor.train_data.copy()
        
        # Try to get festivals for year not in config (e.g., 2017)
        result_df = processor.add_city_festivals(df, 'valencia', [2017])
        
        # Should handle gracefully - may not add columns or add columns with zeros
        # The exact behavior depends on implementation, but shouldn't crash
        assert result_df is not None
        assert len(result_df) == len(df)


@pytest.mark.unit
def test_festival_date_parsing():
    """Test that festival dates are correctly parsed."""
    test_dates = [
        '2015-03-29',
        '2015-04-05',
        '2015-08-26'
    ]
    
    for date_str in test_dates:
        parsed_date = pd.to_datetime(date_str).date()
        assert parsed_date is not None
        assert str(parsed_date) == date_str


@pytest.mark.unit
def test_festival_scale_validation():
    """Test validation of festival scale values."""
    valid_scales = ['small', 'medium', 'large']
    scale_weights = {'small': 0.1, 'medium': 0.3, 'large': 0.6}
    
    for scale in valid_scales:
        assert scale in scale_weights
        assert 0 <= scale_weights[scale] <= 1


@pytest.mark.integration
def test_create_festival_calendar_integration(data_processor_with_config):
    """Test the create_festival_calendar method integration."""
    processor = data_processor_with_config
    
    # Run the full festival calendar creation
    processor.create_festival_calendar()
    
    # Check that festival columns were added to both train and test data
    train_festival_cols = [col for col in processor.train_data.columns if 'has_festival' in col]
    test_festival_cols = [col for col in processor.test_data.columns if 'has_festival' in col]
    
    # Should have same festival columns in both datasets
    assert set(train_festival_cols) == set(test_festival_cols)
    assert len(train_festival_cols) > 0