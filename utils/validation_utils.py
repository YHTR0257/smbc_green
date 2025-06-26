"""
スペイン電力価格予測用 データ検証ユーティリティ
Data validation utilities for Spanish electricity price prediction
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


def validate_date_range(df: pd.DataFrame, start_date: str, end_date: str, 
                       date_column: str = 'time') -> bool:
    """
    データフレームの日付範囲を検証
    
    Args:
        df: 検証するデータフレーム
        start_date: 期待される開始日 (YYYY-MM-DD形式)
        end_date: 期待される終了日 (YYYY-MM-DD形式)
        date_column: 日付列名
        
    Returns:
        検証結果 (True: 正常, False: 異常)
    """
    if date_column not in df.columns:
        logger.error(f"日付列 '{date_column}' が見つかりません")
        return False
    
    try:
        # 日付列をdatetimeに変換
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df_dates = pd.to_datetime(df[date_column])
        else:
            df_dates = df[date_column]
        
        actual_start = df_dates.min().date()
        actual_end = df_dates.max().date()
        expected_start = pd.to_datetime(start_date).date()
        expected_end = pd.to_datetime(end_date).date()
        
        if actual_start < expected_start:
            logger.warning(f"開始日が期待値より早い: {actual_start} < {expected_start}")
            
        if actual_end > expected_end:
            logger.warning(f"終了日が期待値より遅い: {actual_end} > {expected_end}")
            
        logger.info(f"日付範囲検証: {actual_start} ～ {actual_end}")
        return True
        
    except Exception as e:
        logger.error(f"日付範囲検証エラー: {e}")
        return False


def check_missing_values(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, int]:
    """
    必須列の欠損値をチェック
    
    Args:
        df: チェックするデータフレーム
        required_cols: 必須列のリスト
        
    Returns:
        列名をキー、欠損値数を値とする辞書
    """
    missing_summary = {}
    
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"必須列 '{col}' が見つかりません")
            missing_summary[col] = -1  # -1は列が存在しないことを示す
        else:
            missing_count = df[col].isnull().sum()
            missing_summary[col] = missing_count
            
            if missing_count > 0:
                missing_rate = missing_count / len(df) * 100
                logger.warning(f"列 '{col}' に {missing_count}個の欠損値 ({missing_rate:.2f}%)")
    
    return missing_summary


def validate_feature_distribution(df: pd.DataFrame, feature_col: str, 
                                expected_range: Optional[Tuple[float, float]] = None,
                                max_outlier_rate: float = 0.05) -> bool:
    """
    特徴量の分布を検証
    
    Args:
        df: データフレーム
        feature_col: 検証する特徴量列名
        expected_range: 期待される値の範囲 (min, max)
        max_outlier_rate: 外れ値の最大許容率
        
    Returns:
        検証結果 (True: 正常, False: 異常)
    """
    if feature_col not in df.columns:
        logger.error(f"特徴量列 '{feature_col}' が見つかりません")
        return False
    
    if not pd.api.types.is_numeric_dtype(df[feature_col]):
        logger.warning(f"列 '{feature_col}' は数値型ではありません")
        return False
    
    series = df[feature_col].dropna()
    
    if len(series) == 0:
        logger.error(f"列 '{feature_col}' に有効な値がありません")
        return False
    
    # 基本統計
    stats = {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'median': series.median()
    }
    
    logger.info(f"特徴量 '{feature_col}' の統計: {stats}")
    
    # 範囲チェック
    is_valid = True
    if expected_range:
        min_val, max_val = expected_range
        out_of_range = ((series < min_val) | (series > max_val)).sum()
        out_of_range_rate = out_of_range / len(series)
        
        if out_of_range_rate > max_outlier_rate:
            logger.error(f"範囲外の値が多すぎます: {out_of_range}個 ({out_of_range_rate:.2%})")
            is_valid = False
    
    # 外れ値チェック (IQR法)
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    outlier_rate = outliers / len(series)
    
    if outlier_rate > max_outlier_rate:
        logger.warning(f"外れ値が多い: {outliers}個 ({outlier_rate:.2%})")
    
    return is_valid


def validate_festival_features(df: pd.DataFrame, cities: List[str]) -> Dict[str, bool]:
    """
    祭典特徴量の妥当性を検証
    
    Args:
        df: データフレーム
        cities: 都市のリスト
        
    Returns:
        都市をキー、検証結果を値とする辞書
    """
    validation_results = {}
    
    for city in cities:
        city_lower = city.lower()
        if city_lower == 'sevilla':
            city_lower = 'seville'
            
        # 各都市の祭典関連列を検索
        festival_cols = [col for col in df.columns if col.startswith(f'{city_lower}_') and '_active' in col]
        prep_cols = [col for col in df.columns if col.startswith(f'{city_lower}_') and '_preparation' in col]
        aftermath_cols = [col for col in df.columns if col.startswith(f'{city_lower}_') and '_aftermath' in col]
        
        city_valid = True
        
        # 基本チェック
        if not festival_cols:
            logger.warning(f"都市 {city} の祭典列が見つかりません")
            city_valid = False
        
        # 値の妥当性チェック
        for col in festival_cols + prep_cols + aftermath_cols:
            if col in df.columns:
                unique_vals = df[col].unique()
                if not set(unique_vals).issubset({0, 1}):
                    logger.error(f"祭典列 '{col}' に不正な値: {unique_vals}")
                    city_valid = False
        
        # 論理的整合性チェック
        for festival_col in festival_cols:
            base_name = festival_col.replace('_active', '')
            prep_col = f"{base_name}_preparation"
            aftermath_col = f"{base_name}_aftermath"
            
            if prep_col in df.columns and aftermath_col in df.columns:
                # 準備期間とアクティブ期間が重複していないかチェック
                overlap_prep_active = ((df[prep_col] == 1) & (df[festival_col] == 1)).sum()
                overlap_active_aftermath = ((df[festival_col] == 1) & (df[aftermath_col] == 1)).sum()
                
                if overlap_prep_active > 0:
                    logger.warning(f"準備期間とアクティブ期間が重複: {festival_col}")
                    
                if overlap_active_aftermath > 0:
                    logger.warning(f"アクティブ期間と後片付け期間が重複: {festival_col}")
        
        validation_results[city] = city_valid
    
    return validation_results


def validate_population_features(df: pd.DataFrame, cities: List[str]) -> bool:
    """
    人口特徴量の妥当性を検証
    
    Args:
        df: データフレーム
        cities: 都市のリスト
        
    Returns:
        検証結果
    """
    is_valid = True
    
    # 人口重み特徴量の存在チェック
    for city in cities:
        city_lower = city.lower()
        if city_lower == 'sevilla':
            city_lower = 'seville'
            
        weight_col = f'{city_lower}_population_weight'
        raw_col = f'{city_lower}_population_raw'
        
        if weight_col not in df.columns:
            logger.error(f"人口重み列 '{weight_col}' が見つかりません")
            is_valid = False
            
        if raw_col not in df.columns:
            logger.error(f"人口raw列 '{raw_col}' が見つかりません")
            is_valid = False
    
    # 重みの合計チェック
    weight_cols = [col for col in df.columns if col.endswith('_population_weight')]
    if weight_cols:
        weight_sum = df[weight_cols].iloc[0].sum()  # 最初の行の重みの合計
        
        if not np.isclose(weight_sum, 1.0, rtol=1e-3):
            logger.error(f"人口重みの合計が1.0になりません: {weight_sum}")
            is_valid = False
        else:
            logger.info(f"人口重みの合計: {weight_sum}")
    
    return is_valid


def validate_holiday_features(df: pd.DataFrame, years: List[int]) -> bool:
    """
    休日特徴量の妥当性を検証
    
    Args:
        df: データフレーム
        years: 対象年のリスト
        
    Returns:
        検証結果
    """
    if 'is_national_holiday' not in df.columns:
        logger.error("休日列 'is_national_holiday' が見つかりません")
        return False
    
    if 'time' not in df.columns:
        logger.error("時間列 'time' が見つかりません")
        return False
    
    is_valid = True
    
    # 年別休日数の妥当性チェック
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df_time = pd.to_datetime(df['time'])
    else:
        df_time = df['time']
    
    df_with_year = df.copy()
    df_with_year['year'] = df_time.dt.year
    
    yearly_holidays = df_with_year.groupby('year')['is_national_holiday'].sum()
    
    for year in years:
        if year in yearly_holidays.index:
            holiday_count = yearly_holidays[year]
            # スペインの祝日は通常8-12日程度
            if holiday_count < 5 or holiday_count > 15:
                logger.warning(f"{year}年の祝日数が異常: {holiday_count}")
                is_valid = False
            else:
                logger.info(f"{year}年の祝日数: {holiday_count}")
    
    return is_valid


def generate_validation_report(df: pd.DataFrame, config_data: Dict) -> Dict[str, bool]:
    """
    総合的な検証レポートを生成
    
    Args:
        df: データフレーム
        config_data: 設定データ
        
    Returns:
        検証結果の辞書
    """
    report = {}
    
    logger.info("=== データ検証レポート開始 ===")
    
    # 基本データ情報
    logger.info(f"データ形状: {df.shape}")
    logger.info(f"データ期間: {df['time'].min()} ～ {df['time'].max()}")
    
    # 日付範囲検証
    report['date_range'] = validate_date_range(df, '2015-01-01', '2018-12-31')
    
    # 必須列チェック
    required_cols = ['time', 'price_actual']
    missing_vals = check_missing_values(df, required_cols)
    report['required_columns'] = all(count == 0 for count in missing_vals.values() if count != -1)
    
    # 人口特徴量検証
    cities = list(config_data.get('cities', {}).get('coordinates', {}).keys())
    report['population_features'] = validate_population_features(df, cities)
    
    # 休日特徴量検証
    report['holiday_features'] = validate_holiday_features(df, [2015, 2016, 2017, 2018])
    
    # 祭典特徴量検証
    festival_results = validate_festival_features(df, cities)
    report['festival_features'] = all(festival_results.values())
    
    # 全体結果
    all_valid = all(report.values())
    report['overall'] = all_valid
    
    logger.info(f"=== 検証結果: {'PASS' if all_valid else 'FAIL'} ===")
    for check, result in report.items():
        status = "✓" if result else "✗"
        logger.info(f"{status} {check}: {'PASS' if result else 'FAIL'}")
    
    return report