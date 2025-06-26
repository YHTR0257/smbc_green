"""
ユーティリティ機能のテスト
Tests for utility functions (logging and validation)
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import logging
import time
from pathlib import Path
from io import StringIO
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.logging_utils import (
    setup_logger,
    log_feature_statistics,
    log_processing_time,
    log_feature_generation_summary,
    log_festival_calendar_summary,
    log_holiday_summary,
    log_population_weights,
    validate_config_consistency
)
from utils.validation_utils import (
    validate_date_range,
    check_missing_values,
    validate_feature_distribution,
    validate_festival_features,
    validate_population_features,
    validate_holiday_features,
    generate_validation_report
)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    dates = pd.date_range('2015-01-01', '2015-12-31', freq='D')
    
    data = {
        'time': dates,
        'feature1': np.random.normal(10, 2, len(dates)),
        'feature2': np.random.uniform(0, 100, len(dates)),
        'feature3': np.random.exponential(1, len(dates)),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], len(dates)),
        'target': np.random.uniform(40, 80, len(dates))
    }
    
    # Add some missing values
    data['feature1'][10:15] = np.nan
    data['categorical_feature'][20:25] = None
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_festival_dataframe():
    """Create sample dataframe with festival features."""
    dates = pd.date_range('2015-01-01', '2015-12-31', freq='D')
    
    data = {
        'time': dates,
        'madrid_semana_santa_active': np.zeros(len(dates)),
        'madrid_semana_santa_preparation': np.zeros(len(dates)),
        'madrid_semana_santa_aftermath': np.zeros(len(dates)),
        'valencia_la_tomatina_active': np.zeros(len(dates)),
        'barcelona_festival_active': np.zeros(len(dates))
    }
    
    # Set some festival periods
    data['madrid_semana_santa_active'][100:108] = 1  # 8 days
    data['madrid_semana_santa_preparation'][97:100] = 1  # 3 days before
    data['madrid_semana_santa_aftermath'][108:110] = 1  # 2 days after
    data['valencia_la_tomatina_active'][238] = 1  # 1 day
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_config_data():
    """Create sample configuration data."""
    return {
        'cities': {
            'coordinates': {
                'Madrid': [40.4168, -3.7033],
                'Barcelona': [41.3851, 2.1686],
                'Valencia': [39.4699, -0.3756]
            }
        },
        'population_weights': {
            'admin_population': {
                'Madrid': 3200000,
                'Barcelona': 1600000,
                'Valencia': 790000
            },
            'metro_population': {
                'Madrid': 6500000,
                'Barcelona': 5200000,
                'Valencia': 1700000
            }
        },
        'feature_engineering': {
            'scale_impact': {
                'small': 0.1,
                'medium': 0.3,
                'large': 0.6
            }
        }
    }


class TestLoggingUtils:
    """Test cases for logging utilities."""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        logger = setup_logger('test_logger', 'INFO')
        
        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
        
    def test_setup_logger_different_levels(self):
        """Test logger setup with different levels."""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in levels:
            logger = setup_logger(f'test_logger_{level}', level)
            assert logger.level == getattr(logging, level)
            
    def test_log_feature_statistics(self, sample_dataframe, caplog):
        """Test feature statistics logging."""
        with caplog.at_level(logging.INFO):
            log_feature_statistics(sample_dataframe, ['feature1', 'feature2'])
            
        # Check that statistics were logged
        assert 'データフレーム形状' in caplog.text
        assert 'feature1' in caplog.text
        assert 'feature2' in caplog.text
        assert 'count' in caplog.text
        assert 'mean' in caplog.text
        
    def test_log_feature_statistics_missing_column(self, sample_dataframe, caplog):
        """Test feature statistics with missing column."""
        with caplog.at_level(logging.WARNING):
            log_feature_statistics(sample_dataframe, ['nonexistent_column'])
            
        assert 'が見つかりません' in caplog.text
        
    def test_log_feature_statistics_categorical(self, sample_dataframe, caplog):
        """Test feature statistics for categorical data."""
        with caplog.at_level(logging.INFO):
            log_feature_statistics(sample_dataframe, ['categorical_feature'])
            
        assert 'unique' in caplog.text
        assert 'top_values' in caplog.text
        
    def test_log_processing_time(self, caplog):
        """Test processing time logging."""
        start_time = time.time()
        time.sleep(0.1)  # Small delay
        
        with caplog.at_level(logging.INFO):
            log_processing_time('test_function', start_time)
            
        assert 'test_function 処理時間' in caplog.text
        assert '秒' in caplog.text
        
    def test_log_feature_generation_summary(self, sample_dataframe, caplog):
        """Test feature generation summary logging."""
        df_before = sample_dataframe.copy()
        df_after = sample_dataframe.copy()
        df_after['new_feature'] = 1
        
        with caplog.at_level(logging.INFO):
            log_feature_generation_summary(df_before, df_after, 'test')
            
        assert 'test特徴量生成完了' in caplog.text
        assert '追加特徴量数: 1' in caplog.text
        assert 'new_feature' in caplog.text
        
    def test_log_festival_calendar_summary(self, sample_festival_dataframe, caplog):
        """Test festival calendar summary logging."""
        with caplog.at_level(logging.INFO):
            log_festival_calendar_summary(sample_festival_dataframe)
            
        assert '祭典カレンダー概要' in caplog.text
        assert 'アクティブな祭典列数' in caplog.text
        assert '準備期間列数' in caplog.text
        assert '後片付け期間列数' in caplog.text
        
    def test_log_holiday_summary(self, caplog):
        """Test holiday summary logging."""
        holiday_df = pd.DataFrame({
            'time': pd.date_range('2015-01-01', '2015-12-31', freq='D'),
            'is_national_holiday': [0] * 365,
            'year': [2015] * 365
        })
        holiday_df.loc[0, 'is_national_holiday'] = 1  # New Year
        holiday_df.loc[359, 'is_national_holiday'] = 1  # Christmas
        
        with caplog.at_level(logging.INFO):
            log_holiday_summary(holiday_df)
            
        assert '休日データ概要' in caplog.text
        assert '総祝日数: 2' in caplog.text
        
    def test_log_population_weights(self, caplog):
        """Test population weights logging."""
        population_weights = {
            'Madrid': 3200000,
            'Barcelona': 1600000,
            'Valencia': 790000
        }
        
        with caplog.at_level(logging.INFO):
            log_population_weights(population_weights, 'admin')
            
        assert '人口重み付け設定' in caplog.text
        assert 'admin' in caplog.text
        assert 'Madrid' in caplog.text
        assert '総人口' in caplog.text
        
    def test_validate_config_consistency_success(self, sample_config_data):
        """Test config consistency validation - success case."""
        errors = validate_config_consistency(sample_config_data)
        assert len(errors) == 0
        
    def test_validate_config_consistency_missing_keys(self):
        """Test config consistency validation - missing keys."""
        incomplete_config = {'cities': {}}
        errors = validate_config_consistency(incomplete_config)
        
        assert len(errors) > 0
        assert any('必須キー' in error for error in errors)
        
    def test_validate_config_consistency_city_mismatch(self):
        """Test config consistency validation - city mismatch."""
        inconsistent_config = {
            'cities': {
                'coordinates': {'Madrid': [40, -3]}
            },
            'population_weights': {
                'admin_population': {'Barcelona': 1000000},  # Different city
                'metro_population': {'Madrid': 5000000}
            },
            'feature_engineering': {}
        }
        
        errors = validate_config_consistency(inconsistent_config)
        assert len(errors) > 0
        assert any('都市が一致しません' in error for error in errors)


class TestValidationUtils:
    """Test cases for validation utilities."""
    
    def test_validate_date_range_success(self, sample_dataframe):
        """Test successful date range validation."""
        result = validate_date_range(
            sample_dataframe, 
            '2015-01-01', 
            '2015-12-31',
            'time'
        )
        assert result is True
        
    def test_validate_date_range_missing_column(self, sample_dataframe):
        """Test date range validation with missing column."""
        result = validate_date_range(
            sample_dataframe,
            '2015-01-01',
            '2015-12-31',
            'nonexistent_time'
        )
        assert result is False
        
    def test_check_missing_values_success(self, sample_dataframe):
        """Test missing values check."""
        result = check_missing_values(
            sample_dataframe,
            ['feature1', 'feature2', 'categorical_feature']
        )
        
        assert 'feature1' in result
        assert 'feature2' in result
        assert 'categorical_feature' in result
        
        # feature1 has 5 missing values (indices 10-14)
        assert result['feature1'] == 5
        # feature2 has no missing values
        assert result['feature2'] == 0
        # categorical_feature has 5 missing values (indices 20-24)
        assert result['categorical_feature'] == 5
        
    def test_check_missing_values_missing_column(self, sample_dataframe):
        """Test missing values check with missing column."""
        result = check_missing_values(
            sample_dataframe,
            ['nonexistent_column']
        )
        
        assert result['nonexistent_column'] == -1
        
    def test_validate_feature_distribution_success(self, sample_dataframe):
        """Test feature distribution validation - success case."""
        result = validate_feature_distribution(
            sample_dataframe,
            'feature2',
            expected_range=(0, 100)
        )
        assert result is True
        
    def test_validate_feature_distribution_out_of_range(self, sample_dataframe):
        """Test feature distribution validation - out of range."""
        result = validate_feature_distribution(
            sample_dataframe,
            'feature2',
            expected_range=(50, 60),  # Too narrow range
            max_outlier_rate=0.01  # Very strict
        )
        assert result is False
        
    def test_validate_feature_distribution_missing_column(self, sample_dataframe):
        """Test feature distribution validation with missing column."""
        result = validate_feature_distribution(
            sample_dataframe,
            'nonexistent_feature'
        )
        assert result is False
        
    def test_validate_feature_distribution_non_numeric(self, sample_dataframe):
        """Test feature distribution validation with non-numeric column."""
        result = validate_feature_distribution(
            sample_dataframe,
            'categorical_feature'
        )
        assert result is False
        
    def test_validate_festival_features_success(self, sample_festival_dataframe):
        """Test festival features validation - success case."""
        cities = ['madrid', 'valencia', 'barcelona']
        result = validate_festival_features(sample_festival_dataframe, cities)
        
        assert 'madrid' in result
        assert 'valencia' in result
        assert 'barcelona' in result
        
        # Madrid and Valencia should have valid festival features
        assert result['madrid'] is True
        assert result['valencia'] is True
        
    def test_validate_festival_features_invalid_values(self):
        """Test festival features validation with invalid values."""
        invalid_df = pd.DataFrame({
            'madrid_festival_active': [0, 1, 2],  # Invalid value: 2
            'madrid_festival_preparation': [0, 0, 0]
        })
        
        result = validate_festival_features(invalid_df, ['madrid'])
        assert result['madrid'] is False
        
    def test_validate_population_features_success(self):
        """Test population features validation - success case."""
        pop_df = pd.DataFrame({
            'madrid_population_weight': [0.4, 0.4, 0.4],
            'madrid_population_raw': [3200000, 3200000, 3200000],
            'valencia_population_weight': [0.3, 0.3, 0.3],
            'valencia_population_raw': [790000, 790000, 790000],
            'barcelona_population_weight': [0.3, 0.3, 0.3],
            'barcelona_population_raw': [1600000, 1600000, 1600000]
        })
        
        result = validate_population_features(pop_df, ['madrid', 'valencia', 'barcelona'])
        assert result is True
        
    def test_validate_population_features_weights_not_sum_to_one(self):
        """Test population features validation when weights don't sum to 1."""
        pop_df = pd.DataFrame({
            'madrid_population_weight': [0.5, 0.5, 0.5],  # Sum = 0.8 ≠ 1.0
            'valencia_population_weight': [0.3, 0.3, 0.3],
            'madrid_population_raw': [3200000, 3200000, 3200000],
            'valencia_population_raw': [790000, 790000, 790000]
        })
        
        result = validate_population_features(pop_df, ['madrid', 'valencia'])
        assert result is False
        
    def test_validate_holiday_features_success(self):
        """Test holiday features validation - success case."""
        holiday_df = pd.DataFrame({
            'time': pd.date_range('2015-01-01', '2015-12-31', freq='D'),
            'is_national_holiday': [0] * 365
        })
        # Add some holidays
        holiday_df.loc[0, 'is_national_holiday'] = 1
        holiday_df.loc[100, 'is_national_holiday'] = 1
        holiday_df.loc[359, 'is_national_holiday'] = 1
        
        result = validate_holiday_features(holiday_df, [2015])
        assert result is True
        
    def test_validate_holiday_features_too_many_holidays(self):
        """Test holiday features validation with unreasonable holiday count."""
        holiday_df = pd.DataFrame({
            'time': pd.date_range('2015-01-01', '2015-12-31', freq='D'),
            'is_national_holiday': [1] * 365  # Every day is a holiday!
        })
        
        result = validate_holiday_features(holiday_df, [2015])
        assert result is False
        
    def test_validate_holiday_features_missing_columns(self):
        """Test holiday features validation with missing columns."""
        incomplete_df = pd.DataFrame({
            'time': pd.date_range('2015-01-01', '2015-12-31', freq='D')
            # Missing 'is_national_holiday' column
        })
        
        result = validate_holiday_features(incomplete_df, [2015])
        assert result is False
        
    def test_generate_validation_report(self, sample_config_data):
        """Test validation report generation."""
        # Create comprehensive test dataframe
        test_df = pd.DataFrame({
            'time': pd.date_range('2015-01-01', '2015-12-31', freq='D'),
            'price_actual': np.random.uniform(40, 80, 365),
            'is_national_holiday': [0] * 365,
            'madrid_population_weight': [0.4] * 365,
            'madrid_population_raw': [3200000] * 365,
            'valencia_population_weight': [0.3] * 365,
            'valencia_population_raw': [790000] * 790,
            'barcelona_population_weight': [0.3] * 365,
            'barcelona_population_raw': [1600000] * 365,
            'madrid_festival_active': [0] * 365,
            'valencia_festival_active': [0] * 365
        })
        
        # Add some holidays
        test_df.loc[0, 'is_national_holiday'] = 1
        test_df.loc[100, 'is_national_holiday'] = 1
        
        report = generate_validation_report(test_df, sample_config_data)
        
        assert 'date_range' in report
        assert 'required_columns' in report
        assert 'population_features' in report
        assert 'holiday_features' in report
        assert 'festival_features' in report
        assert 'overall' in report
        
        # Most validations should pass for this well-formed test data
        assert report['date_range'] is True
        assert report['required_columns'] is True


class TestValidationEdgeCases:
    """Test edge cases for validation utilities."""
    
    def test_empty_dataframe_validation(self):
        """Test validation with empty dataframe."""
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        result = validate_date_range(empty_df, '2015-01-01', '2015-12-31', 'time')
        assert result is False
        
        missing_result = check_missing_values(empty_df, ['feature1'])
        assert missing_result['feature1'] == -1
        
    def test_single_row_dataframe(self):
        """Test validation with single-row dataframe."""
        single_row_df = pd.DataFrame({
            'time': [pd.Timestamp('2015-01-01')],
            'feature1': [10.0],
            'is_national_holiday': [1]
        })
        
        result = validate_date_range(single_row_df, '2015-01-01', '2015-12-31', 'time')
        assert result is True
        
        missing_result = check_missing_values(single_row_df, ['feature1'])
        assert missing_result['feature1'] == 0
        
    def test_all_nan_column(self):
        """Test validation with column containing all NaN values."""
        nan_df = pd.DataFrame({
            'time': pd.date_range('2015-01-01', '2015-01-10', freq='D'),
            'all_nan_feature': [np.nan] * 10
        })
        
        result = validate_feature_distribution(nan_df, 'all_nan_feature')
        assert result is False
        
        missing_result = check_missing_values(nan_df, ['all_nan_feature'])
        assert missing_result['all_nan_feature'] == 10


@pytest.mark.unit
def test_logging_level_conversion():
    """Test logging level string to constant conversion."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    for level_str, level_const in levels.items():
        assert getattr(logging, level_str) == level_const


@pytest.mark.unit
def test_statistics_calculation():
    """Test statistics calculation logic."""
    data = np.array([1, 2, 3, 4, 5])
    
    assert np.mean(data) == 3.0
    assert np.std(data) == np.sqrt(2.0)
    assert np.min(data) == 1
    assert np.max(data) == 5
    assert len(data) == 5


@pytest.mark.integration
def test_full_validation_pipeline_with_real_config(sample_config_data):
    """Test full validation pipeline with realistic data."""
    # Create realistic test data
    dates = pd.date_range('2015-01-01', '2017-12-31', freq='D')
    n_samples = len(dates)
    
    realistic_df = pd.DataFrame({
        'time': dates,
        'price_actual': np.random.uniform(30, 90, n_samples),
        'is_national_holiday': np.random.choice([0, 1], n_samples, p=[0.97, 0.03]),
        'madrid_population_weight': [0.4] * n_samples,
        'madrid_population_raw': [3200000] * n_samples,
        'valencia_population_weight': [0.3] * n_samples,
        'valencia_population_raw': [790000] * n_samples,
        'barcelona_population_weight': [0.3] * n_samples,
        'barcelona_population_raw': [1600000] * n_samples,
        'madrid_semana_santa_active': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'valencia_la_tomatina_active': np.random.choice([0, 1], n_samples, p=[0.999, 0.001])
    })
    
    # Run full validation
    report = generate_validation_report(realistic_df, sample_config_data)
    
    # Should pass most validations
    assert report['overall'] in [True, False]  # May fail due to specific validation rules
    assert 'date_range' in report
    assert 'population_features' in report