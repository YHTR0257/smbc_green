"""
統合テスト
Integration tests for the complete festival feature pipeline
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
from utils.validation_utils import generate_validation_report


@pytest.fixture
def comprehensive_test_data():
    """Create comprehensive test data covering multiple years and scenarios."""
    # Create data spanning multiple years with holidays and festival periods
    dates = pd.date_range('2015-01-01', '2018-12-31', freq='D')
    n_samples = len(dates)
    
    np.random.seed(42)
    
    data = {
        'time': dates,
        'price_actual': np.random.uniform(30, 90, n_samples),
        
        # Weather data for all cities
        'madrid_temp': np.random.normal(15, 8, n_samples),
        'madrid_humidity': np.random.uniform(30, 80, n_samples),
        'madrid_rain': np.random.exponential(0.8, n_samples),
        'madrid_temp_max': np.random.normal(20, 8, n_samples),
        'madrid_temp_min': np.random.normal(10, 8, n_samples),
        
        'barcelona_temp': np.random.normal(18, 6, n_samples),
        'barcelona_humidity': np.random.uniform(40, 85, n_samples),
        'barcelona_rain': np.random.exponential(0.6, n_samples),
        'barcelona_temp_max': np.random.normal(23, 6, n_samples),
        'barcelona_temp_min': np.random.normal(13, 6, n_samples),
        
        'valencia_temp': np.random.normal(20, 7, n_samples),
        'valencia_humidity': np.random.uniform(45, 90, n_samples),
        'valencia_rain': np.random.exponential(0.5, n_samples),
        'valencia_temp_max': np.random.normal(25, 7, n_samples),
        'valencia_temp_min': np.random.normal(15, 7, n_samples),
        
        'seville_temp': np.random.normal(22, 9, n_samples),
        'seville_humidity': np.random.uniform(35, 85, n_samples),
        'seville_rain': np.random.exponential(0.4, n_samples),
        'seville_temp_max': np.random.normal(28, 9, n_samples),
        'seville_temp_min': np.random.normal(16, 9, n_samples),
        
        'bilbao_temp': np.random.normal(16, 6, n_samples),
        'bilbao_humidity': np.random.uniform(50, 90, n_samples),
        'bilbao_rain': np.random.exponential(1.2, n_samples),
        'bilbao_temp_max': np.random.normal(21, 6, n_samples),
        'bilbao_temp_min': np.random.normal(11, 6, n_samples),
        
        # Energy generation data
        'generation_biomass': np.random.uniform(400, 600, n_samples),
        'generation_fossil_gas': np.random.uniform(4000, 8000, n_samples),
        'generation_nuclear': np.random.uniform(5000, 7000, n_samples),
        'generation_solar': np.random.uniform(0, 3000, n_samples),
        'generation_wind_onshore': np.random.uniform(1000, 5000, n_samples),
        'total_load_actual': np.random.uniform(20000, 35000, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def comprehensive_config_files():
    """Create comprehensive config files for integration testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Complete city property config
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
                'distance_decay': {
                    'max_distance_km': 400,
                    'half_effect_distance_km': 200
                },
                'scale_impact': {
                    'small': 0.1,
                    'medium': 0.3,
                    'large': 0.6
                },
                'time_effects': {
                    'preparation_days_before': 3,
                    'aftermath_days_after': 2,
                    'evening_start_hour': 19,
                    'evening_end_hour': 23
                },
                'interactions': {
                    'holiday_amplification_factor': 1.5,
                    'rain_threshold_mm': 5.0,
                    'indoor_displacement_rate': 0.3,
                    'weekday_afternoon_reduction': -0.2,
                    'weekday_evening_increase': 0.3
                }
            },
            'feature_selection': {
                'max_features': 15,
                'cv_folds': 5,
                'alpha_range': {
                    'min': 1e-4,
                    'max': 1.0,
                    'num_alphas': 50
                },
                'multicollinearity_threshold': 30,
                'correlation_threshold': 0.8
            }
        }
        
        # Complete festival and holiday config
        festival_config = {
            'national_holidays': {
                '2015': {
                    '2015-01-01': 'New Year\'s Day',
                    '2015-01-06': 'Epiphany',
                    '2015-04-03': 'Good Friday',
                    '2015-05-01': 'Labor Day',
                    '2015-08-15': 'Assumption of Mary',
                    '2015-10-12': 'Hispanic Day',
                    '2015-11-01': 'All Saints Day',
                    '2015-12-06': 'Constitution Day',
                    '2015-12-08': 'Immaculate Conception',
                    '2015-12-25': 'Christmas Day'
                },
                '2016': {
                    '2016-01-01': 'New Year\'s Day',
                    '2016-01-06': 'Epiphany',
                    '2016-03-25': 'Good Friday',
                    '2016-05-01': 'Labor Day',
                    '2016-08-15': 'Assumption of Mary',
                    '2016-10-12': 'Hispanic Day',
                    '2016-11-01': 'All Saints Day',
                    '2016-12-06': 'Constitution Day',
                    '2016-12-08': 'Immaculate Conception',
                    '2016-12-25': 'Christmas Day'
                },
                '2017': {
                    '2017-01-01': 'New Year\'s Day',
                    '2017-01-06': 'Epiphany',
                    '2017-04-14': 'Good Friday',
                    '2017-05-01': 'Labor Day',
                    '2017-08-15': 'Assumption of Mary',
                    '2017-10-12': 'Hispanic Day',
                    '2017-11-01': 'All Saints Day',
                    '2017-12-06': 'Constitution Day',
                    '2017-12-08': 'Immaculate Conception',
                    '2017-12-25': 'Christmas Day'
                },
                '2018': {
                    '2018-01-01': 'New Year\'s Day',
                    '2018-01-06': 'Epiphany',
                    '2018-03-30': 'Good Friday',
                    '2018-05-01': 'Labor Day',
                    '2018-08-15': 'Assumption of Mary',
                    '2018-10-12': 'Hispanic Day',
                    '2018-11-01': 'All Saints Day',
                    '2018-12-06': 'Constitution Day',
                    '2018-12-08': 'Immaculate Conception',
                    '2018-12-25': 'Christmas Day'
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
                    },
                    '2016': {
                        'start_date': '2016-03-20',
                        'end_date': '2016-03-27',
                        'primary_cities': ['Sevilla', 'Valencia'],
                        'scale': 'large',
                        'outdoor_rate': 0.7
                    },
                    '2017': {
                        'start_date': '2017-04-09',
                        'end_date': '2017-04-16',
                        'primary_cities': ['Sevilla', 'Valencia'],
                        'scale': 'large',
                        'outdoor_rate': 0.7
                    },
                    '2018': {
                        'start_date': '2018-03-25',
                        'end_date': '2018-04-01',
                        'primary_cities': ['Sevilla', 'Valencia'],
                        'scale': 'large',
                        'outdoor_rate': 0.7
                    }
                },
                'feria_abril': {
                    '2015': {
                        'start_date': '2015-04-21',
                        'end_date': '2015-04-26',
                        'primary_cities': ['Sevilla'],
                        'scale': 'medium',
                        'outdoor_rate': 0.8
                    },
                    '2016': {
                        'start_date': '2016-04-12',
                        'end_date': '2016-04-17',
                        'primary_cities': ['Sevilla'],
                        'scale': 'medium',
                        'outdoor_rate': 0.8
                    },
                    '2017': {
                        'start_date': '2017-05-02',
                        'end_date': '2017-05-07',
                        'primary_cities': ['Sevilla'],
                        'scale': 'medium',
                        'outdoor_rate': 0.8
                    },
                    '2018': {
                        'start_date': '2018-04-17',
                        'end_date': '2018-04-22',
                        'primary_cities': ['Sevilla'],
                        'scale': 'medium',
                        'outdoor_rate': 0.8
                    }
                },
                'san_fermin': {
                    '2015': {
                        'start_date': '2015-07-06',
                        'end_date': '2015-07-14',
                        'primary_cities': ['Bilbao'],
                        'scale': 'medium',
                        'outdoor_rate': 0.9
                    },
                    '2016': {
                        'start_date': '2016-07-06',
                        'end_date': '2016-07-14',
                        'primary_cities': ['Bilbao'],
                        'scale': 'medium',
                        'outdoor_rate': 0.9
                    },
                    '2017': {
                        'start_date': '2017-07-06',
                        'end_date': '2017-07-14',
                        'primary_cities': ['Bilbao'],
                        'scale': 'medium',
                        'outdoor_rate': 0.9
                    },
                    '2018': {
                        'start_date': '2018-07-06',
                        'end_date': '2018-07-14',
                        'primary_cities': ['Bilbao'],
                        'scale': 'medium',
                        'outdoor_rate': 0.9
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
                    },
                    '2017': {
                        'start_date': '2017-08-30',
                        'end_date': '2017-08-30',
                        'primary_cities': ['Valencia'],
                        'scale': 'small',
                        'outdoor_rate': 1.0
                    },
                    '2018': {
                        'start_date': '2018-08-29',
                        'end_date': '2018-08-29',
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
def comprehensive_data_processor(comprehensive_config_files, comprehensive_test_data):
    """Create DataProcessor with comprehensive test setup."""
    # Save data to temporary files
    train_file = comprehensive_config_files / 'train.csv'
    test_file = comprehensive_config_files / 'test.csv'
    
    # Split data for train/test
    train_data = comprehensive_test_data[comprehensive_test_data['time'].dt.year <= 2017]
    test_data = comprehensive_test_data[comprehensive_test_data['time'].dt.year == 2018]
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    config = {
        'population_type': 'admin',
        'train_config': {
            'general': {
                'target': 'price_actual',
                'drop_columns': [],
                'use_time_split': True,
                'time_column': 'time',
                'train_years': [2015, 2016],
                'validation_year': 2017
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
    processor.config_loader = ConfigLoader(config_dir=str(comprehensive_config_files))
    processor.cities_config = processor.config_loader.get_cities_config()
    processor.feature_config = processor.config_loader.get_feature_engineering_config()
    processor.population_admin = processor.config_loader.get_population_config("admin")
    processor.population_metro = processor.config_loader.get_population_config("metro")
    processor.population_weights = processor.population_admin
    
    return processor


class TestCompleteIntegration:
    """Test complete integration of all festival features."""
    
    def test_full_pipeline_execution(self, comprehensive_data_processor):
        """Test complete pipeline execution without errors."""
        processor = comprehensive_data_processor
        
        # Run the complete pipeline
        train_df, val_df, test_df = processor.process_all()
        
        # Basic sanity checks
        assert train_df is not None
        assert val_df is not None
        assert test_df is not None
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        
        # Check that target column exists in train/val but not test
        assert 'price_actual' in train_df.columns
        assert 'price_actual' in val_df.columns
        assert 'price_actual' not in test_df.columns
        
    def test_feature_categories_present(self, comprehensive_data_processor):
        """Test that all major feature categories are present."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Check time features
        time_features = ['year', 'month', 'day', 'hour', 'day_of_week']
        for feature in time_features:
            assert feature in train_df.columns
            
        # Check population features
        pop_features = [col for col in train_df.columns if 'population_weight' in col]
        assert len(pop_features) >= 5  # One for each city
        
        # Check holiday features
        assert 'is_national_holiday' in train_df.columns
        assert 'holiday_name' in train_df.columns
        
        # Check festival features
        festival_features = [col for col in train_df.columns if '_active' in col]
        assert len(festival_features) > 0
        
        # Check interaction features
        interaction_features = [col for col in train_df.columns if any(keyword in col for keyword in ['amplified', 'pop_weighted', 'rain_reduction', 'evening_effect'])]
        assert len(interaction_features) > 0
        
    def test_data_consistency_across_splits(self, comprehensive_data_processor):
        """Test data consistency across train/val/test splits."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # All datasets should have same columns (except target in test)
        train_cols = set(train_df.columns) - {'price_actual'}
        val_cols = set(val_df.columns) - {'price_actual'}
        test_cols = set(test_df.columns)
        
        assert train_cols == val_cols == test_cols
        
        # Population weights should be identical across all datasets
        pop_weight_cols = [col for col in train_cols if 'population_weight' in col]
        for col in pop_weight_cols:
            assert train_df[col].iloc[0] == val_df[col].iloc[0] == test_df[col].iloc[0]
            
    def test_temporal_consistency(self, comprehensive_data_processor):
        """Test temporal consistency of features."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Train data should be from 2015-2016
        train_years = train_df['year'].unique()
        assert set(train_years).issubset({2015, 2016})
        
        # Val data should be from 2017
        val_years = val_df['year'].unique()
        assert set(val_years) == {2017}
        
        # Test data should be from 2018
        test_years = test_df['year'].unique()
        assert set(test_years) == {2018}
        
    def test_festival_temporal_accuracy(self, comprehensive_data_processor):
        """Test that festivals occur at correct times."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Combine all data for checking
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # Check Semana Santa (should occur in March/April)
        semana_santa_cols = [col for col in all_data.columns if 'semana_santa_active' in col]
        for col in semana_santa_cols:
            active_dates = all_data[all_data[col] == 1]['time']
            if len(active_dates) > 0:
                months = active_dates.dt.month.unique()
                assert all(month in [3, 4] for month in months)
                
        # Check La Tomatina (should occur in August)
        tomatina_cols = [col for col in all_data.columns if 'la_tomatina_active' in col]
        for col in tomatina_cols:
            active_dates = all_data[all_data[col] == 1]['time']
            if len(active_dates) > 0:
                months = active_dates.dt.month.unique()
                assert all(month == 8 for month in months)
                
    def test_holiday_distribution(self, comprehensive_data_processor):
        """Test holiday distribution across years."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Each year should have reasonable number of holidays (8-12)
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
        yearly_holidays = all_data.groupby('year')['is_national_holiday'].sum()
        
        for year, holiday_count in yearly_holidays.items():
            assert 8 <= holiday_count <= 12
            
    def test_population_weight_normalization(self, comprehensive_data_processor):
        """Test that population weights are properly normalized."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Check population weight normalization
        weight_cols = [col for col in train_df.columns if 'population_weight' in col]
        
        for df in [train_df, val_df, test_df]:
            total_weight = df[weight_cols].iloc[0].sum()
            assert abs(total_weight - 1.0) < 0.001
            
    def test_interaction_feature_logic(self, comprehensive_data_processor):
        """Test interaction feature logic."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Test holiday amplification logic
        amplified_cols = [col for col in train_df.columns if 'holiday_amplified' in col]
        
        for col in amplified_cols:
            base_col = col.replace('_holiday_amplified', '')
            if base_col in train_df.columns:
                # Where both festival and holiday are 0, amplified should be 0
                no_festival_no_holiday = (train_df[base_col] == 0) & (train_df['is_national_holiday'] == 0)
                if no_festival_no_holiday.any():
                    assert (train_df.loc[no_festival_no_holiday, col] == 0).all()


class TestPopulationTypeComparison:
    """Test comparison between admin and metro population types."""
    
    def test_admin_vs_metro_population_differences(self, comprehensive_config_files, comprehensive_test_data):
        """Test differences between admin and metro population processing."""
        # Test admin population
        admin_processor = self._create_processor(comprehensive_config_files, comprehensive_test_data, 'admin')
        admin_train, admin_val, admin_test = admin_processor.process_all()
        
        # Test metro population
        metro_processor = self._create_processor(comprehensive_config_files, comprehensive_test_data, 'metro')
        metro_train, metro_val, metro_test = metro_processor.process_all()
        
        # Population raw values should be different
        madrid_admin = admin_train['madrid_population_raw'].iloc[0]
        madrid_metro = metro_train['madrid_population_raw'].iloc[0]
        assert madrid_metro > madrid_admin  # Metro should be larger
        
        # Weights should be different but still sum to 1
        admin_weight_cols = [col for col in admin_train.columns if 'population_weight' in col]
        metro_weight_cols = [col for col in metro_train.columns if 'population_weight' in col]
        
        admin_madrid_weight = admin_train['madrid_population_weight'].iloc[0]
        metro_madrid_weight = metro_train['madrid_population_weight'].iloc[0]
        
        # Due to different population bases, weights should be different
        assert admin_madrid_weight != metro_madrid_weight
        
        # Both should sum to 1
        assert abs(admin_train[admin_weight_cols].iloc[0].sum() - 1.0) < 0.001
        assert abs(metro_train[metro_weight_cols].iloc[0].sum() - 1.0) < 0.001
        
    def _create_processor(self, config_files, test_data, pop_type):
        """Helper to create processor with specified population type."""
        train_file = config_files / f'train_{pop_type}.csv'
        test_file = config_files / f'test_{pop_type}.csv'
        
        train_data = test_data[test_data['time'].dt.year <= 2017]
        test_data_split = test_data[test_data['time'].dt.year == 2018]
        
        train_data.to_csv(train_file, index=False)
        test_data_split.to_csv(test_file, index=False)
        
        config = {
            'population_type': pop_type,
            'train_config': {
                'general': {
                    'target': 'price_actual',
                    'drop_columns': [],
                    'use_time_split': True,
                    'time_column': 'time',
                    'train_years': [2015, 2016],
                    'validation_year': 2017
                }
            }
        }
        
        processor = DataProcessor(
            train_file_path=train_file,
            test_file_path=test_file,
            config=config
        )
        
        from config.config_loader import ConfigLoader
        processor.config_loader = ConfigLoader(config_dir=str(config_files))
        processor.cities_config = processor.config_loader.get_cities_config()
        processor.feature_config = processor.config_loader.get_feature_engineering_config()
        processor.population_admin = processor.config_loader.get_population_config("admin")
        processor.population_metro = processor.config_loader.get_population_config("metro")
        processor.population_weights = processor.population_admin if pop_type == 'admin' else processor.population_metro
        
        return processor


class TestValidationIntegration:
    """Test integration with validation utilities."""
    
    def test_comprehensive_validation_report(self, comprehensive_data_processor):
        """Test comprehensive validation report on processed data."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Load config for validation
        config_data = {
            'cities': processor.cities_config,
            'population_weights': {
                'admin_population': processor.population_admin,
                'metro_population': processor.population_metro
            },
            'feature_engineering': processor.feature_config
        }
        
        # Test validation on train data
        train_report = generate_validation_report(train_df, config_data)
        
        # Should pass most validations
        assert train_report['date_range'] is True
        assert train_report['required_columns'] is True
        assert train_report['population_features'] is True
        assert train_report['holiday_features'] is True
        
        # Test validation on val data
        val_report = generate_validation_report(val_df, config_data)
        assert val_report['date_range'] is True
        assert val_report['population_features'] is True
        
    def test_feature_statistics_comprehensive(self, comprehensive_data_processor):
        """Test feature statistics on comprehensive dataset."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        from utils.logging_utils import log_feature_statistics
        
        # Test on different feature types
        numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = train_df.select_dtypes(include=['object']).columns.tolist()
        
        # Should not raise errors
        log_feature_statistics(train_df, numeric_features[:10])  # Test subset
        log_feature_statistics(train_df, categorical_features[:5])


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    def test_processing_time_reasonable(self, comprehensive_data_processor):
        """Test that processing time is reasonable for large dataset."""
        import time
        
        processor = comprehensive_data_processor
        
        start_time = time.time()
        train_df, val_df, test_df = processor.process_all()
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert processing_time < 60  # Should complete within 1 minute
        
        # Should produce reasonable amount of features
        assert len(train_df.columns) > 50  # Should have many features
        assert len(train_df.columns) < 500  # But not too many
        
    def test_memory_usage_reasonable(self, comprehensive_data_processor):
        """Test that memory usage is reasonable."""
        processor = comprehensive_data_processor
        train_df, val_df, test_df = processor.process_all()
        
        # Check memory usage of dataframes
        train_memory = train_df.memory_usage(deep=True).sum()
        val_memory = val_df.memory_usage(deep=True).sum()
        test_memory = test_df.memory_usage(deep=True).sum()
        
        # Should be reasonable (less than 100MB for test data)
        assert train_memory < 100 * 1024 * 1024  # 100MB
        assert val_memory < 100 * 1024 * 1024
        assert test_memory < 100 * 1024 * 1024


@pytest.mark.slow
def test_full_pipeline_with_real_config():
    """Test full pipeline with actual config files (if available)."""
    # This test uses actual config files if they exist
    config_dir = Path(__file__).parent.parent / 'config'
    
    if not (config_dir / 'city_property.yml').exists():
        pytest.skip("Real config files not available")
        
    # Test with minimal real data
    sample_data = pd.DataFrame({
        'time': pd.date_range('2015-01-01', '2015-01-31', freq='H'),
        'price_actual': np.random.uniform(40, 80, 24*31)
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        train_file = Path(temp_dir) / 'train.csv'
        test_file = Path(temp_dir) / 'test.csv'
        
        sample_data.to_csv(train_file, index=False)
        sample_data.iloc[:100].to_csv(test_file, index=False)
        
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
        
        # Should not raise errors with real config
        train_df, val_df, test_df = processor.process_all()
        
        assert train_df is not None
        assert val_df is not None
        assert test_df is not None


@pytest.mark.integration  
def test_end_to_end_feature_pipeline(comprehensive_data_processor):
    """Test complete end-to-end feature pipeline."""
    processor = comprehensive_data_processor
    
    # Track original data shape
    original_train_shape = processor.train_data.shape
    original_test_shape = processor.test_data.shape
    
    # Run complete pipeline
    train_df, val_df, test_df = processor.process_all()
    
    # Verify significant feature expansion
    assert train_df.shape[1] > original_train_shape[1] * 3
    assert test_df.shape[1] > original_test_shape[1] * 3
    
    # Verify no data leakage (same number of samples)
    assert len(train_df) + len(val_df) == original_train_shape[0]
    assert len(test_df) == original_test_shape[0]
    
    # Verify feature consistency
    train_features = set(train_df.columns) - {'price_actual'}
    val_features = set(val_df.columns) - {'price_actual'}  
    test_features = set(test_df.columns)
    
    assert train_features == val_features == test_features
    
    # Verify no missing values in critical features
    critical_features = ['is_national_holiday', 'madrid_population_weight', 'year', 'month', 'day']
    for feature in critical_features:
        if feature in train_df.columns:
            assert train_df[feature].notna().all()
            assert val_df[feature].notna().all()
            assert test_df[feature].notna().all()
            
    print(f"Pipeline completed successfully:")
    print(f"  Train: {original_train_shape} → {train_df.shape}")
    print(f"  Val: {val_df.shape}")
    print(f"  Test: {original_test_shape} → {test_df.shape}")
    print(f"  Total features added: {train_df.shape[1] - original_train_shape[1]}")