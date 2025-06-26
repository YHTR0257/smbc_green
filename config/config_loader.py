"""
Spanish electricity price prediction configuration management system
Festival feature engineering configuration loader
"""

import yaml
import os
from typing import Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Class for loading and managing configuration files"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Args:
            config_dir: Configuration files directory path
        """
        self.config_dir = Path(config_dir)
        self._cached_configs = {}
        
    def load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load YAML file and return as dictionary
        
        Args:
            config_path: Configuration file path
            
        Returns:
            Dictionary of loaded configuration
            
        Raises:
            FileNotFoundError: When file is not found
            yaml.YAMLError: When YAML parsing error occurs
        """
        if config_path in self._cached_configs:
            return self._cached_configs[config_path]
            
        file_path = self.config_dir / config_path
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            if config is None:
                raise ValueError(f"Configuration file is empty: {file_path}")
                
            self._cached_configs[config_path] = config
            logger.info(f"Configuration file loaded: {file_path}")
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML parsing error in {file_path}: {e}")
    
    def get_cities_config(self) -> Dict[str, Any]:
        """
        Get city configuration
        
        Returns:
            Dictionary of city coordinates and basic information
        """
        config = self.load_yaml_config("city_property.yml")
        
        if "cities" not in config:
            raise KeyError("Cities configuration not found")
            
        cities_config = config["cities"]
        
        # Validation
        if "coordinates" not in cities_config:
            raise KeyError("Cities coordinates configuration not found")
            
        coordinates = cities_config["coordinates"]
        expected_cities = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao"]
        
        for city in expected_cities:
            if city not in coordinates:
                raise KeyError(f"Coordinates for city {city} not configured")
            if not isinstance(coordinates[city], list) or len(coordinates[city]) != 2:
                raise ValueError(f"Invalid coordinate format for city {city} (must be [latitude, longitude] list)")
                
        return cities_config
    
    def get_population_config(self, weight_type: str) -> Dict[str, int]:
        """
        人口重み付け設定を取得
        
        Args:
            weight_type: 'admin' または 'metro'
            
        Returns:
            都市別人口の辞書
            
        Raises:
            ValueError: weight_typeが不正な場合
            KeyError: 設定が見つからない場合
        """
        if weight_type not in ["admin", "metro"]:
            raise ValueError(f"weight_typeは 'admin' または 'metro' である必要があります: {weight_type}")
            
        config = self.load_yaml_config("city_property.yml")
        
        if "population_weights" not in config:
            raise KeyError("population_weights設定が見つかりません")
            
        pop_config = config["population_weights"]
        key = f"{weight_type}_population"
        
        if key not in pop_config:
            raise KeyError(f"{key}設定が見つかりません")
            
        population_data = pop_config[key]
        expected_cities = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Bilbao"]
        
        # Validation
        for city in expected_cities:
            if city not in population_data:
                raise KeyError(f"都市 {city} の人口データが設定されていません")
            if not isinstance(population_data[city], int) or population_data[city] <= 0:
                raise ValueError(f"都市 {city} の人口データが不正です (正の整数である必要があります)")
                
        return population_data
    
    def get_holidays_config(self, year: int) -> Dict[str, str]:
        """
        指定年の祝日設定を取得
        
        Args:
            year: 対象年 (2015-2018)
            
        Returns:
            日付文字列をキー、祝日名を値とする辞書
            
        Raises:
            ValueError: 対象年が範囲外の場合
            KeyError: 設定が見つからない場合
        """
        if year < 2015 or year > 2018:
            raise ValueError(f"対象年は2015-2018の範囲である必要があります: {year}")
            
        config = self.load_yaml_config("holiday_festrival_config.yml")
        
        if "national_holidays" not in config:
            raise KeyError("National holidays configuration not found")
            
        holidays_config = config["national_holidays"]
        year_key = str(year)
        
        if year_key not in holidays_config:
            raise KeyError(f"Holiday configuration for year {year} not found")
            
        year_holidays = holidays_config[year_key]
        
        # Validation
        for date_str, holiday_name in year_holidays.items():
            try:
                # 日付形式の簡単なチェック
                if not date_str.startswith(str(year)):
                    raise ValueError(f"日付 {date_str} が年 {year} と一致しません")
            except:
                raise ValueError(f"不正な日付形式: {date_str}")
                
        return year_holidays
    
    def get_festivals_config(self, year: int) -> Dict[str, Dict[str, Any]]:
        """
        指定年の祭典設定を取得
        
        Args:
            year: 対象年 (2015-2018)
            
        Returns:
            祭典名をキー、祭典情報を値とする辞書
            
        Raises:
            ValueError: 対象年が範囲外の場合
            KeyError: 設定が見つからない場合
        """
        if year < 2015 or year > 2018:
            raise ValueError(f"対象年は2015-2018の範囲である必要があります: {year}")
            
        config = self.load_yaml_config("holiday_festrival_config.yml")
        
        if "festivals" not in config:
            raise KeyError("Festivals configuration not found")
            
        festivals_config = config["festivals"]
        year_key = str(year)
        
        result = {}
        for festival_name, festival_data in festivals_config.items():
            if year_key not in festival_data:
                logger.warning(f"Festival {festival_name} data not found for year {year}")
                continue
                
            festival_year_data = festival_data[year_key]
            
            # Validation
            required_fields = ["start_date", "end_date", "primary_cities", "scale", "outdoor_rate"]
            for field in required_fields:
                if field not in festival_year_data:
                    raise KeyError(f"祭典 {festival_name} の {field} が設定されていません")
                    
            # scale値の検証
            if festival_year_data["scale"] not in ["small", "medium", "large"]:
                raise ValueError(f"祭典 {festival_name} のscaleが不正です: {festival_year_data['scale']}")
                
            # outdoor_rate値の検証
            outdoor_rate = festival_year_data["outdoor_rate"]
            if not isinstance(outdoor_rate, (int, float)) or outdoor_rate < 0 or outdoor_rate > 1:
                raise ValueError(f"祭典 {festival_name} のoutdoor_rateが不正です: {outdoor_rate}")
                
            result[festival_name] = festival_year_data
            
        return result
    
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """
        特徴量エンジニアリング設定を取得
        
        Returns:
            特徴量エンジニアリング設定の辞書
        """
        config = self.load_yaml_config("city_property.yml")
        
        if "feature_engineering" not in config:
            raise KeyError("feature_engineering設定が見つかりません")
            
        return config["feature_engineering"]
    
    def get_feature_selection_config(self) -> Dict[str, Any]:
        """
        特徴量選択設定を取得
        
        Returns:
            特徴量選択設定の辞書
        """
        config = self.load_yaml_config("city_property.yml")
        
        if "feature_selection" not in config:
            raise KeyError("feature_selection設定が見つかりません")
            
        return config["feature_selection"]
    
    def validate_all_configs(self) -> bool:
        """
        全ての設定ファイルの整合性をチェック
        
        Returns:
            全ての設定が正常な場合True
            
        Raises:
            各種エラー: 設定に問題がある場合
        """
        try:
            # 都市設定の検証
            cities_config = self.get_cities_config()
            logger.info("都市設定の検証: OK")
            
            # 人口設定の検証
            admin_pop = self.get_population_config("admin")
            metro_pop = self.get_population_config("metro")
            logger.info("人口設定の検証: OK")
            
            # 2015-2018年の祝日・祭典設定の検証
            for year in range(2015, 2019):
                holidays = self.get_holidays_config(year)
                festivals = self.get_festivals_config(year)
                logger.info(f"{year}年の祝日・祭典設定の検証: OK")
                
            # 特徴量設定の検証
            fe_config = self.get_feature_engineering_config()
            fs_config = self.get_feature_selection_config()
            logger.info("特徴量設定の検証: OK")
            
            logger.info("全ての設定ファイルの検証が完了しました")
            return True
            
        except Exception as e:
            logger.error(f"設定検証エラー: {e}")
            raise