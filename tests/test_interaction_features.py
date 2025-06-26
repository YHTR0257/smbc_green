"""
相互作用特徴量のテスト
Tests for interaction feature functionality
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from src.data_processing import DataProcessor


@pytest.fixture
def sample_interaction_data():
    """Create sample data for testing interactions."""
    np.random.seed(42)
    
    dates = pd.date_range('2015-01-01', '2015-12-31', freq='H')
    n_samples = len(dates)
    
    data = {
        'time': dates,
        'price_actual': np.random.uniform(40, 80, n_samples),
        # Weather data for interactions
        'madrid_temp': np.random.uniform(10, 30, n_samples),
        'madrid_humidity': np.random.uniform(30, 80, n_samples),
        'madrid_rain': np.random.exponential(1, n_samples),  # Some rain events
        'valencia_temp': np.random.uniform(15, 35, n_samples),
        'valencia_humidity': np.random.uniform(40, 90, n_samples),
        'valencia_rain': np.random.exponential(0.5, n_samples),
        'seville_temp': np.random.uniform(18, 38, n_samples),
        'seville_humidity': np.random.uniform(35, 85, n_samples),
        'seville_rain': np.random.exponential(0.3, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_config_for_interactions():
    """Create temporary config files for interaction testing."""
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
                    'rain_threshold_mm': 5.0,
                    'indoor_displacement_rate': 0.3,
                    'weekday_afternoon_reduction': -0.2,
                    'weekday_evening_increase': 0.3
                }
            }
        }
        
        # Holiday and festival config
        holiday_config = {
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
            yaml.dump(holiday_config, f)
            
        yield config_dir


@pytest.fixture
def interaction_data_processor(temp_config_for_interactions, sample_interaction_data):
    """Create DataProcessor for interaction testing."""
    # Save sample data to temporary files
    train_file = temp_config_for_interactions / 'train.csv'
    test_file = temp_config_for_interactions / 'test.csv'
    
    sample_interaction_data.to_csv(train_file, index=False)
    sample_interaction_data.iloc[:1000].to_csv(test_file, index=False)
    
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
    processor.config_loader = ConfigLoader(config_dir=str(temp_config_for_interactions))
    processor.cities_config = processor.config_loader.get_cities_config()
    processor.feature_config = processor.config_loader.get_feature_engineering_config()
    processor.population_admin = processor.config_loader.get_population_config("admin")
    processor.population_weights = processor.population_admin
    
    return processor


class TestBasicInteractions:
    """Test basic interaction feature generation."""
    
    def test_add_basic_interactions_time_features(self, interaction_data_processor):
        """Test that basic time-related interactions are added."""
        processor = interaction_data_processor
        
        # First add prerequisite features
        processor.time_handling()
        
        # Add basic interactions
        processor.add_basic_interactions()
        
        # Check time-related columns
        assert 'is_weekend' in processor.train_data.columns
        assert 'is_weekday' in processor.train_data.columns
        assert 'is_evening' in processor.train_data.columns
        
        # Validate weekend/weekday logic
        train_data = processor.train_data
        weekend_mask = train_data['day_of_week'] >= 5
        assert (train_data.loc[weekend_mask, 'is_weekend'] == 1).all()
        assert (train_data.loc[weekend_mask, 'is_weekday'] == 0).all()
        
        weekday_mask = train_data['day_of_week'] < 5
        assert (train_data.loc[weekday_mask, 'is_weekend'] == 0).all()
        assert (train_data.loc[weekday_mask, 'is_weekday'] == 1).all()
        
    def test_evening_time_detection(self, interaction_data_processor):
        """Test evening time detection (19-23 hours)."""
        processor = interaction_data_processor
        processor.time_handling()
        processor.add_basic_interactions()
        
        train_data = processor.train_data
        
        # Check evening hours (19-23)
        evening_mask = (train_data['hour'] >= 19) & (train_data['hour'] <= 23)
        assert (train_data.loc[evening_mask, 'is_evening'] == 1).all()
        
        # Check non-evening hours
        non_evening_mask = (train_data['hour'] < 19) | (train_data['hour'] > 23)
        assert (train_data.loc[non_evening_mask, 'is_evening'] == 0).all()


class TestHolidayFestivalInteractions:
    """Test holiday and festival interaction features."""
    
    def test_holiday_festival_amplification(self, interaction_data_processor):
        """Test holiday-festival amplification effect."""
        processor = interaction_data_processor
        
        # Add prerequisite features
        processor.time_handling()
        processor.add_population_features()
        processor.add_holiday_features()
        processor.create_festival_calendar()
        
        # Add interaction features
        processor.add_basic_interactions()
        
        # Check for amplified festival columns
        amplified_cols = [col for col in processor.train_data.columns if 'holiday_amplified' in col]
        
        # Should have some amplified columns if festivals exist
        if any('has_festival' in col for col in processor.train_data.columns):
            assert len(amplified_cols) > 0
            
            # Check amplification logic
            for col in amplified_cols:
                base_col = col.replace('_festival_holiday_amplified', '_has_festival')
                if base_col in processor.train_data.columns:
                    train_data = processor.train_data
                    
                    # Where both festival and holiday are active, amplification should be 1.5x
                    both_active = (train_data[base_col] == 1) & (train_data['is_national_holiday'] == 1)
                    if both_active.any():
                        expected = train_data.loc[both_active, base_col] * 1.5
                        actual = train_data.loc[both_active, col]
                        assert np.allclose(expected, actual)
                        
    def test_population_weighted_festivals(self, interaction_data_processor):
        """Test population-weighted festival effects."""
        processor = interaction_data_processor
        
        # Add prerequisite features
        processor.time_handling()
        processor.add_population_features()
        processor.create_festival_calendar()
        processor.add_basic_interactions()
        
        # Check for population-weighted columns
        pop_weighted_cols = [col for col in processor.train_data.columns if 'pop_weighted' in col]
        
        if len(pop_weighted_cols) > 0:
            for col in pop_weighted_cols:
                # Extract city from column name
                parts = col.split('_')
                city = parts[0]
                base_col = f'{city}_has_festival'
                pop_weight_col = f'{city}_population_weight'
                
                if base_col in processor.train_data.columns and pop_weight_col in processor.train_data.columns:
                    train_data = processor.train_data
                    
                    # Check that population weighting is correctly applied
                    expected = train_data[base_col] * train_data[pop_weight_col]
                    actual = train_data[col]
                    assert np.allclose(expected, actual, rtol=1e-10)


class TestWeatherFestivalInteractions:
    """Test weather-festival interaction features."""
    
    def test_rain_festival_reduction(self, interaction_data_processor):
        """Test rain-induced festival reduction effects."""
        processor = interaction_data_processor
        
        # Add prerequisite features
        processor.time_handling()
        processor.create_festival_calendar()
        processor.add_basic_interactions()
        
        # Check for rain reduction columns
        rain_reduction_cols = [col for col in processor.train_data.columns if 'rain_reduction' in col]
        
        if len(rain_reduction_cols) > 0:
            for col in rain_reduction_cols:
                # Extract city from column name
                parts = col.split('_')
                city = parts[0]
                rain_col = f'{city}_rain'
                
                if rain_col in processor.train_data.columns:
                    train_data = processor.train_data
                    
                    # Rain reduction should be between 0 and outdoor festival activity
                    assert (train_data[col] >= 0).all()
                    
                    # When rain is 0, reduction should be 0
                    no_rain_mask = train_data[rain_col] == 0
                    if no_rain_mask.any():
                        assert (train_data.loc[no_rain_mask, col] == 0).all()
                        
    def test_indoor_displacement_effect(self, interaction_data_processor):
        """Test indoor displacement effects from rain."""
        processor = interaction_data_processor
        
        # Add prerequisite features
        processor.time_handling()
        processor.create_festival_calendar()
        processor.add_basic_interactions()
        
        # Check for indoor displacement columns
        indoor_cols = [col for col in processor.train_data.columns if 'indoor_displacement' in col]
        
        if len(indoor_cols) > 0:
            for col in indoor_cols:
                rain_reduction_col = col.replace('_indoor_displacement', '_rain_reduction')
                
                if rain_reduction_col in processor.train_data.columns:
                    train_data = processor.train_data
                    
                    # Indoor displacement should be 30% of rain reduction
                    expected = train_data[rain_reduction_col] * 0.3
                    actual = train_data[col]
                    assert np.allclose(expected, actual, rtol=1e-10)


class TestTimeOfDayInteractions:
    """Test time-of-day interaction features."""
    
    def test_evening_festival_effects(self, interaction_data_processor):
        """Test evening festival effect calculations."""
        processor = interaction_data_processor
        
        # Add prerequisite features
        processor.time_handling()
        processor.create_festival_calendar()
        processor.add_basic_interactions()
        
        # Check for evening effect columns
        evening_effect_cols = [col for col in processor.train_data.columns if 'evening_effect' in col]
        
        if len(evening_effect_cols) > 0:
            for col in evening_effect_cols:
                base_col = col.replace('_festival_evening_effect', '_has_festival')
                
                if base_col in processor.train_data.columns:
                    train_data = processor.train_data
                    
                    # Evening effect should be base_col * is_evening
                    expected = train_data[base_col] * train_data['is_evening']
                    actual = train_data[col]
                    assert np.allclose(expected, actual)
                    
                    # During non-evening hours, effect should be 0
                    non_evening_mask = train_data['is_evening'] == 0
                    if non_evening_mask.any():
                        assert (train_data.loc[non_evening_mask, col] == 0).all()


class TestInteractionConfiguration:
    """Test that interaction parameters are correctly loaded and applied."""
    
    def test_interaction_config_loading(self, interaction_data_processor):
        """Test that interaction configuration is correctly loaded."""
        processor = interaction_data_processor
        
        interactions_config = processor.feature_config.get('interactions', {})
        
        # Check key parameters
        assert interactions_config.get('holiday_amplification_factor') == 1.5
        assert interactions_config.get('rain_threshold_mm') == 5.0
        assert interactions_config.get('indoor_displacement_rate') == 0.3
        
    def test_rain_threshold_application(self, interaction_data_processor):
        """Test that rain threshold is correctly applied."""
        processor = interaction_data_processor
        
        processor.time_handling()
        processor.create_festival_calendar()
        processor.add_basic_interactions()
        
        # Find rain reduction columns and check threshold application
        rain_reduction_cols = [col for col in processor.train_data.columns if 'rain_reduction' in col]
        
        if len(rain_reduction_cols) > 0:
            train_data = processor.train_data
            
            for col in rain_reduction_cols:
                parts = col.split('_')
                city = parts[0]
                rain_col = f'{city}_rain'
                
                if rain_col in train_data.columns:
                    # Check that rain values are clipped at threshold (5.0mm)
                    rain_factor = np.clip(train_data[rain_col] / 5.0, 0, 1)
                    
                    # The rain reduction should incorporate this clipped factor
                    # (exact calculation depends on other factors, but should be <= rain_factor)
                    assert (train_data[col] <= rain_factor * 2).all()  # allowing for other multipliers


class TestInteractionEdgeCases:
    """Test edge cases for interaction features."""
    
    def test_missing_prerequisite_columns(self, interaction_data_processor):
        """Test handling when prerequisite columns are missing."""
        processor = interaction_data_processor
        
        # Try adding interactions without prerequisite features
        processor.add_basic_interactions()
        
        # Should handle gracefully without crashing
        assert processor.train_data is not None
        assert processor.test_data is not None
        
    def test_empty_festival_data(self, interaction_data_processor):
        """Test interactions when no festivals are active."""
        processor = interaction_data_processor
        
        # Add basic features but skip festival creation
        processor.time_handling()
        processor.add_population_features()
        processor.add_holiday_features()
        
        # Add interactions without festivals
        processor.add_basic_interactions()
        
        # Should still add basic time interactions
        assert 'is_weekend' in processor.train_data.columns
        assert 'is_evening' in processor.train_data.columns
        
    def test_extreme_weather_values(self, interaction_data_processor):
        """Test interactions with extreme weather values."""
        processor = interaction_data_processor
        
        # Modify data to have extreme rain values
        processor.train_data['valencia_rain'] = 100.0  # Very high rain
        processor.test_data['valencia_rain'] = 100.0
        
        processor.time_handling()
        processor.create_festival_calendar()
        processor.add_basic_interactions()
        
        # Should handle extreme values gracefully (clipping should prevent issues)
        rain_reduction_cols = [col for col in processor.train_data.columns if 'rain_reduction' in col]
        
        for col in rain_reduction_cols:
            # Values should be reasonable (not infinite or extremely large)
            assert (processor.train_data[col] <= 10).all()  # reasonable upper bound
            assert (processor.train_data[col] >= 0).all()   # non-negative


@pytest.mark.unit
def test_clipping_function():
    """Test numpy clipping behavior used in interactions."""
    test_values = np.array([0, 2.5, 5.0, 7.5, 10.0])
    threshold = 5.0
    
    clipped = np.clip(test_values / threshold, 0, 1)
    expected = np.array([0.0, 0.5, 1.0, 1.0, 1.0])
    
    assert np.allclose(clipped, expected)


@pytest.mark.unit
def test_amplification_calculation():
    """Test amplification calculation logic."""
    festival_active = np.array([0, 1, 0, 1])
    holiday_active = np.array([0, 0, 1, 1])
    amplification_factor = 1.5
    
    result = festival_active * holiday_active * amplification_factor
    expected = np.array([0.0, 0.0, 0.0, 1.5])
    
    assert np.allclose(result, expected)


@pytest.mark.integration
def test_full_interaction_pipeline(interaction_data_processor):
    """Test complete interaction feature pipeline."""
    processor = interaction_data_processor
    
    # Run full pipeline
    processor.time_handling()
    processor.add_population_features()
    processor.add_holiday_features()
    processor.create_festival_calendar()
    processor.add_basic_interactions()
    
    # Verify we have a reasonable number of new columns
    original_cols = len(['time', 'price_actual', 'madrid_temp', 'madrid_humidity', 'madrid_rain',
                        'valencia_temp', 'valencia_humidity', 'valencia_rain',
                        'seville_temp', 'seville_humidity', 'seville_rain'])
    
    final_cols = len(processor.train_data.columns)
    
    # Should have significantly more columns after all feature additions
    assert final_cols > original_cols * 2
    
    # Check that both datasets have same columns
    assert set(processor.train_data.columns) == set(processor.test_data.columns)