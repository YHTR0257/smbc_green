"""
ConfigLoader機能のテスト
Tests for ConfigLoader functionality
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config_loader import ConfigLoader


@pytest.fixture
def temp_config_dir():
    """Create temporary config directory with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Create test city_property.yml
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
                'interactions': {
                    'holiday_amplification_factor': 1.5,
                    'rain_threshold_mm': 5.0,
                    'indoor_displacement_rate': 0.3
                }
            },
            'feature_selection': {
                'max_features': 15,
                'cv_folds': 5,
                'alpha_range': {
                    'min': 1e-4,
                    'max': 1.0,
                    'num_alphas': 50
                }
            }
        }
        
        # Create test holiday_festrival_config.yml
        festival_config = {
            'national_holidays': {
                '2015': {
                    '2015-01-01': 'New Year\'s Day',
                    '2015-01-06': 'Epiphany',
                    '2015-12-25': 'Christmas Day'
                },
                '2016': {
                    '2016-01-01': 'New Year\'s Day',
                    '2016-01-06': 'Epiphany',
                    '2016-12-25': 'Christmas Day'
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
def config_loader(temp_config_dir):
    """Create ConfigLoader instance with temporary config directory."""
    return ConfigLoader(config_dir=str(temp_config_dir))


class TestConfigLoader:
    """Test cases for ConfigLoader class."""
    
    def test_load_yaml_config_success(self, config_loader):
        """Test successful YAML config loading."""
        config = config_loader.load_yaml_config('city_property.yml')
        
        assert config is not None
        assert 'cities' in config
        assert 'population_weights' in config
        assert 'feature_engineering' in config
        
    def test_load_yaml_config_file_not_found(self, config_loader):
        """Test YAML config loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            config_loader.load_yaml_config('non_existent.yml')
            
    def test_load_yaml_config_caching(self, config_loader):
        """Test that config loading uses caching."""
        # Load same config twice
        config1 = config_loader.load_yaml_config('city_property.yml')
        config2 = config_loader.load_yaml_config('city_property.yml')
        
        # Should be the same object (cached)
        assert config1 is config2
        
    def test_get_cities_config_success(self, config_loader):
        """Test successful cities config retrieval."""
        cities_config = config_loader.get_cities_config()
        
        assert 'coordinates' in cities_config
        coordinates = cities_config['coordinates']
        
        # Check all expected cities
        expected_cities = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao']
        for city in expected_cities:
            assert city in coordinates
            assert isinstance(coordinates[city], list)
            assert len(coordinates[city]) == 2  # lat, lng
            
    def test_get_population_config_admin(self, config_loader):
        """Test population config retrieval for admin type."""
        pop_config = config_loader.get_population_config('admin')
        
        expected_cities = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao']
        for city in expected_cities:
            assert city in pop_config
            assert isinstance(pop_config[city], int)
            assert pop_config[city] > 0
            
    def test_get_population_config_metro(self, config_loader):
        """Test population config retrieval for metro type."""
        pop_config = config_loader.get_population_config('metro')
        
        expected_cities = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Bilbao']
        for city in expected_cities:
            assert city in pop_config
            assert isinstance(pop_config[city], int)
            assert pop_config[city] > 0
            
        # Metro population should be larger than admin for major cities
        admin_config = config_loader.get_population_config('admin')
        assert pop_config['Madrid'] > admin_config['Madrid']
        assert pop_config['Barcelona'] > admin_config['Barcelona']
        
    def test_get_population_config_invalid_type(self, config_loader):
        """Test population config retrieval with invalid type."""
        with pytest.raises(ValueError, match="weight_typeは 'admin' または 'metro' である必要があります"):
            config_loader.get_population_config('invalid')
            
    def test_get_holidays_config_2015(self, config_loader):
        """Test holiday config retrieval for 2015."""
        holidays = config_loader.get_holidays_config(2015)
        
        assert '2015-01-01' in holidays
        assert holidays['2015-01-01'] == 'New Year\'s Day'
        assert '2015-01-06' in holidays
        assert '2015-12-25' in holidays
        
    def test_get_holidays_config_invalid_year(self, config_loader):
        """Test holiday config retrieval with invalid year."""
        with pytest.raises(ValueError, match="対象年は2015-2018の範囲である必要があります"):
            config_loader.get_holidays_config(2010)
            
        with pytest.raises(ValueError, match="対象年は2015-2018の範囲である必要があります"):
            config_loader.get_holidays_config(2020)
            
    def test_get_festivals_config_2015(self, config_loader):
        """Test festival config retrieval for 2015."""
        festivals = config_loader.get_festivals_config(2015)
        
        assert 'semana_santa' in festivals
        assert 'la_tomatina' in festivals
        
        # Check semana_santa details
        semana_santa = festivals['semana_santa']
        assert semana_santa['start_date'] == '2015-03-29'
        assert semana_santa['end_date'] == '2015-04-05'
        assert semana_santa['scale'] == 'large'
        assert semana_santa['outdoor_rate'] == 0.7
        assert 'Sevilla' in semana_santa['primary_cities']
        assert 'Valencia' in semana_santa['primary_cities']
        
    def test_get_festivals_config_invalid_year(self, config_loader):
        """Test festival config retrieval with invalid year."""
        with pytest.raises(ValueError, match="対象年は2015-2018の範囲である必要があります"):
            config_loader.get_festivals_config(2010)
            
    def test_get_feature_engineering_config(self, config_loader):
        """Test feature engineering config retrieval."""
        fe_config = config_loader.get_feature_engineering_config()
        
        assert 'distance_decay' in fe_config
        assert 'scale_impact' in fe_config
        assert 'interactions' in fe_config
        
        # Check specific values
        assert fe_config['distance_decay']['max_distance_km'] == 400
        assert fe_config['scale_impact']['large'] == 0.6
        assert fe_config['interactions']['holiday_amplification_factor'] == 1.5
        
    def test_get_feature_selection_config(self, config_loader):
        """Test feature selection config retrieval."""
        fs_config = config_loader.get_feature_selection_config()
        
        assert 'max_features' in fs_config
        assert 'cv_folds' in fs_config
        assert 'alpha_range' in fs_config
        
        assert fs_config['max_features'] == 15
        assert fs_config['cv_folds'] == 5
        
    def test_validate_all_configs_success(self, config_loader):
        """Test validation of all configs - success case."""
        result = config_loader.validate_all_configs()
        assert result is True
        
    def test_festival_config_validation(self, config_loader):
        """Test festival config validation."""
        festivals = config_loader.get_festivals_config(2015)
        
        for festival_name, festival_data in festivals.items():
            # Check required fields
            required_fields = ['start_date', 'end_date', 'primary_cities', 'scale', 'outdoor_rate']
            for field in required_fields:
                assert field in festival_data, f"Festival {festival_name} missing {field}"
                
            # Check scale values
            assert festival_data['scale'] in ['small', 'medium', 'large']
            
            # Check outdoor_rate range
            assert 0 <= festival_data['outdoor_rate'] <= 1
            
            # Check primary_cities is list
            assert isinstance(festival_data['primary_cities'], list)
            assert len(festival_data['primary_cities']) > 0


class TestConfigLoaderEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_yaml_file(self, temp_config_dir):
        """Test handling of empty YAML file."""
        # Create empty file
        empty_file = temp_config_dir / 'empty.yml'
        empty_file.touch()
        
        config_loader = ConfigLoader(config_dir=str(temp_config_dir))
        
        with pytest.raises(ValueError, match="設定ファイルが空です"):
            config_loader.load_yaml_config('empty.yml')
            
    def test_malformed_yaml_file(self, temp_config_dir):
        """Test handling of malformed YAML file."""
        # Create malformed YAML file
        malformed_file = temp_config_dir / 'malformed.yml'
        with open(malformed_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed bracket")
            
        config_loader = ConfigLoader(config_dir=str(temp_config_dir))
        
        with pytest.raises(yaml.YAMLError):
            config_loader.load_yaml_config('malformed.yml')
            
    def test_missing_required_keys(self, temp_config_dir):
        """Test handling of missing required configuration keys."""
        # Create config with missing keys
        incomplete_config = {'cities': {'coordinates': {}}}
        
        with open(temp_config_dir / 'incomplete.yml', 'w') as f:
            yaml.dump(incomplete_config, f)
            
        config_loader = ConfigLoader(config_dir=str(temp_config_dir))
        
        # Should raise error when trying to get population config
        with pytest.raises(KeyError):
            config_loader.get_population_config('admin')
            
    def test_invalid_coordinate_format(self, temp_config_dir):
        """Test handling of invalid coordinate format."""
        invalid_config = {
            'cities': {
                'coordinates': {
                    'Madrid': [40.4168]  # Missing longitude
                }
            }
        }
        
        with open(temp_config_dir / 'invalid_coords.yml', 'w') as f:
            yaml.dump(invalid_config, f)
            
        config_loader = ConfigLoader(config_dir=str(temp_config_dir))
        
        with pytest.raises(ValueError, match="座標形式が不正です"):
            config_loader.get_cities_config()
            
    def test_negative_population_values(self, temp_config_dir):
        """Test handling of negative population values."""
        invalid_config = {
            'cities': {
                'coordinates': {'Madrid': [40.4168, -3.7033]}
            },
            'population_weights': {
                'admin_population': {
                    'Madrid': -1000  # Invalid negative population
                }
            }
        }
        
        with open(temp_config_dir / 'invalid_pop.yml', 'w') as f:
            yaml.dump(invalid_config, f)
            
        config_loader = ConfigLoader(config_dir=str(temp_config_dir))
        
        with pytest.raises(ValueError, match="人口データが不正です"):
            config_loader.get_population_config('admin')


@pytest.mark.unit
def test_config_loader_initialization():
    """Test ConfigLoader initialization with different parameters."""
    # Test default initialization
    loader1 = ConfigLoader()
    assert loader1.config_dir == Path('config')
    
    # Test custom directory
    loader2 = ConfigLoader('/custom/path')
    assert loader2.config_dir == Path('/custom/path')
    
    # Test that cache is initially empty
    assert len(loader1._cached_configs) == 0
    assert len(loader2._cached_configs) == 0