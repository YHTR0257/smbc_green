"""
スペイン電力価格予測用 ログ・ユーティリティ
Logging and utility functions for Spanish electricity price prediction
"""

import logging
import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any
from pathlib import Path


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    ログ設定を初期化
    
    Args:
        name: ロガー名
        level: ログレベル ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        
    Returns:
        設定済みのロガー
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger


def log_feature_statistics(df: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    特徴量の統計情報をログ出力
    
    Args:
        df: データフレーム
        feature_cols: 統計を出力する特徴量のリスト
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"データフレーム形状: {df.shape}")
    
    for col in feature_cols:
        if col not in df.columns:
            logger.warning(f"列 {col} が見つかりません")
            continue
            
        if pd.api.types.is_numeric_dtype(df[col]):
            stats = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'null_count': df[col].isnull().sum(),
                'zero_count': (df[col] == 0).sum()
            }
            logger.info(f"{col}: {stats}")
        else:
            value_counts = df[col].value_counts()
            null_count = df[col].isnull().sum()
            logger.info(f"{col}: unique={df[col].nunique()}, null={null_count}, top_values={dict(value_counts.head())}")


def log_processing_time(func_name: str, start_time: float) -> None:
    """
    処理時間をログ出力
    
    Args:
        func_name: 関数名
        start_time: 開始時刻 (time.time()の戻り値)
    """
    logger = logging.getLogger(__name__)
    elapsed_time = time.time() - start_time
    logger.info(f"{func_name} 処理時間: {elapsed_time:.2f}秒")


def log_feature_generation_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, 
                                 feature_type: str) -> None:
    """
    特徴量生成の概要をログ出力
    
    Args:
        df_before: 処理前のデータフレーム
        df_after: 処理後のデータフレーム
        feature_type: 特徴量の種類
    """
    logger = logging.getLogger(__name__)
    
    shape_before = df_before.shape
    shape_after = df_after.shape
    added_features = shape_after[1] - shape_before[1]
    
    logger.info(f"{feature_type}特徴量生成完了:")
    logger.info(f"  処理前: {shape_before}")
    logger.info(f"  処理後: {shape_after}")
    logger.info(f"  追加特徴量数: {added_features}")
    
    # 新しく追加された列の確認
    if added_features > 0:
        new_cols = [col for col in df_after.columns if col not in df_before.columns]
        logger.info(f"  新規列: {new_cols[:10]}{'...' if len(new_cols) > 10 else ''}")


def log_festival_calendar_summary(df: pd.DataFrame) -> None:
    """
    祭典カレンダーの概要をログ出力
    
    Args:
        df: 祭典特徴量が追加されたデータフレーム
    """
    logger = logging.getLogger(__name__)
    
    # 祭典関連列の検出
    festival_active_cols = [col for col in df.columns if '_active' in col]
    festival_prep_cols = [col for col in df.columns if '_preparation' in col]
    festival_aftermath_cols = [col for col in df.columns if '_aftermath' in col]
    
    logger.info("祭典カレンダー概要:")
    logger.info(f"  アクティブな祭典列数: {len(festival_active_cols)}")
    logger.info(f"  準備期間列数: {len(festival_prep_cols)}")
    logger.info(f"  後片付け期間列数: {len(festival_aftermath_cols)}")
    
    # 各祭典の発生回数
    for col in festival_active_cols[:5]:  # 最初の5つのみ表示
        active_count = df[col].sum()
        logger.info(f"  {col}: {active_count}回発生")


def log_holiday_summary(df: pd.DataFrame) -> None:
    """
    休日データの概要をログ出力
    
    Args:
        df: 休日特徴量が追加されたデータフレーム
    """
    logger = logging.getLogger(__name__)
    
    if 'is_national_holiday' in df.columns:
        holiday_count = df['is_national_holiday'].sum()
        total_rows = len(df)
        holiday_rate = holiday_count / total_rows * 100
        
        logger.info("休日データ概要:")
        logger.info(f"  総祝日数: {holiday_count}")
        logger.info(f"  全データに占める割合: {holiday_rate:.2f}%")
        
        # 年別祝日数
        if 'year' in df.columns:
            yearly_holidays = df.groupby('year')['is_national_holiday'].sum()
            logger.info(f"  年別祝日数: {dict(yearly_holidays)}")


def log_population_weights(population_weights: Dict[str, int], population_type: str) -> None:
    """
    人口重み付け情報をログ出力
    
    Args:
        population_weights: 都市別人口辞書
        population_type: 人口タイプ ('admin' または 'metro')
    """
    logger = logging.getLogger(__name__)
    
    total_pop = sum(population_weights.values())
    normalized_weights = {city: pop/total_pop for city, pop in population_weights.items()}
    
    logger.info(f"人口重み付け設定 (タイプ: {population_type}):")
    logger.info(f"  総人口: {total_pop:,}")
    
    for city, (raw_pop, norm_weight) in zip(population_weights.keys(), 
                                           zip(population_weights.values(), normalized_weights.values())):
        logger.info(f"  {city}: {raw_pop:,} ({norm_weight:.3f})")


def validate_config_consistency(config_data: Dict[str, Any]) -> List[str]:
    """
    設定の整合性チェック
    
    Args:
        config_data: 設定データ
        
    Returns:
        エラーメッセージのリスト
    """
    errors = []
    
    # 必須キーの存在チェック
    required_keys = ['cities', 'population_weights', 'feature_engineering']
    for key in required_keys:
        if key not in config_data:
            errors.append(f"必須キー '{key}' が見つかりません")
    
    # 都市の整合性チェック
    if 'cities' in config_data and 'population_weights' in config_data:
        cities_in_coords = set(config_data['cities']['coordinates'].keys())
        cities_in_admin_pop = set(config_data['population_weights']['admin_population'].keys())
        cities_in_metro_pop = set(config_data['population_weights']['metro_population'].keys())
        
        if cities_in_coords != cities_in_admin_pop:
            errors.append(f"座標と行政人口の都市が一致しません: {cities_in_coords} vs {cities_in_admin_pop}")
        
        if cities_in_coords != cities_in_metro_pop:
            errors.append(f"座標と都市圏人口の都市が一致しません: {cities_in_coords} vs {cities_in_metro_pop}")
    
    return errors