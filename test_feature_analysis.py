#!/usr/bin/env python3
"""
Test script for feature analysis functionality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add source directory to path
sys.path.append(str(Path(__file__).parent / 'src'))
from model_training import (
    get_feature_groups, 
    expand_feature_patterns, 
    get_features_by_group,
    analyze_feature_contribution
)

def test_feature_groups():
    """Test feature group definitions and pattern matching."""
    print("Testing feature group functionality...")
    
    # Test feature groups definition
    feature_groups = get_feature_groups()
    print(f"Defined feature groups: {list(feature_groups.keys())}")
    
    # Sample column names (simulating a real dataset)
    sample_columns = [
        'generation_solar', 'generation_wind_onshore', 'generation_nuclear',
        'total_load_actual', 'madrid_temp', 'barcelona_temp', 'valencia_temp',
        'madrid_pressure', 'barcelona_pressure', 'madrid_humidity',
        'madrid_icon_number', 'madrid_is_day', 'madrid_has_time_info',
        'madrid_is_clear', 'madrid_has_clouds', 'madrid_has_precipitation',
        'barcelona_main_clear', 'barcelona_main_clouds', 'madrid_main_rain',
        'madrid_desc_clouds', 'madrid_precipitation_intensity', 'madrid_has_shower',
        'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend',
        'madrid_discomfort1', 'barcelona_diff_temp',
        'madrid_has_festival', 'is_national_holiday', 'holiday_name',
        'madrid_population_weight', 'total_population',
        'supply_demand_balance_ratio', 'supply_sufficiency_ratio',
        'madrid_festival_holiday_amplified', 'madrid_festival_rain_reduction'
    ]
    
    print(f"\nTesting with {len(sample_columns)} sample columns")
    
    # Test pattern expansion
    for group_name, patterns in feature_groups.items():
        print(f"\n--- Testing group: {group_name} ---")
        print(f"Patterns: {patterns}")
        
        expanded = expand_feature_patterns(patterns, sample_columns)
        print(f"Matched features ({len(expanded)}): {expanded}")
        
        # Test helper function
        features_by_group = get_features_by_group(group_name, sample_columns)
        assert expanded == features_by_group, f"Mismatch in {group_name}"
    
    print("\n✅ Feature group tests passed!")

def test_contribution_analysis():
    """Test contribution analysis functionality."""
    print("\nTesting contribution analysis...")
    
    # Create mock results
    mock_results = [
        {
            'group_added': 'baseline',
            'val_rmse': 15.0,
            'val_r2': 0.85,
            'feature_count': 50,
            'features_added': ['generation_solar', 'madrid_temp']
        },
        {
            'group_added': 'weather_icon',
            'val_rmse': 14.5,
            'val_r2': 0.87,
            'feature_count': 65,
            'rmse_improvement': 0.5,
            'relative_improvement': 3.33,
            'features_added_count': 15,
            'efficiency': 0.033,
            'features_added': ['madrid_icon_number', 'madrid_is_day']
        },
        {
            'group_added': 'weather_main',
            'val_rmse': 14.8,
            'val_r2': 0.86,
            'feature_count': 60,
            'rmse_improvement': 0.2,
            'relative_improvement': 1.33,
            'features_added_count': 10,
            'efficiency': 0.02,
            'features_added': ['madrid_main_clear', 'madrid_main_clouds']
        },
        {
            'group_added': 'temporal',
            'val_rmse': 15.1,
            'val_r2': 0.84,
            'feature_count': 56,
            'rmse_improvement': -0.1,
            'relative_improvement': -0.67,
            'features_added_count': 6,
            'efficiency': -0.017,
            'features_added': ['year', 'month', 'day']
        }
    ]
    
    # Test analysis
    analysis = analyze_feature_contribution(mock_results)
    
    print("Analysis results:")
    print(f"Baseline: {analysis['baseline']}")
    print(f"Best improvement: {analysis['summary']['best_improvement']}")
    print(f"Most efficient: {analysis['summary']['most_efficient']}")
    print(f"Positive improvements: {analysis['summary']['positive_improvements']}")
    print(f"Negative improvements: {analysis['summary']['negative_improvements']}")
    
    # Verify rankings
    assert len(analysis['contribution_ranking']) == 3  # Excluding baseline
    assert analysis['contribution_ranking'][0]['group'] == 'weather_icon'  # Best improvement
    assert analysis['summary']['positive_improvements'] == 2
    assert analysis['summary']['negative_improvements'] == 1
    
    print("✅ Contribution analysis tests passed!")

def test_usage_examples():
    """Show usage examples."""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("""
# 1. Run feature analysis only (no training)
python src/model_training.py

# 2. Run analysis with LightGBM and continue training
python -c "
from src.model_training import main
main('dataset_20250625_01', 
     model_type='lightgbm',
     run_feature_analysis=True,
     analysis_only=False)
"

# 3. Run cumulative analysis (each group builds on previous)
python -c "
from src.model_training import main
main('dataset_20250625_01',
     run_feature_analysis=True,
     cumulative=True,
     analysis_only=True)
"

# 4. Test different baseline
python -c "
from src.model_training import main
main('dataset_20250625_01',
     baseline_group='temporal',
     run_feature_analysis=True)
"
""")

if __name__ == "__main__":
    print("Feature Analysis System Test")
    print("="*50)
    
    try:
        test_feature_groups()
        test_contribution_analysis()
        test_usage_examples()
        
        print(f"\n{'='*50}")
        print("🎉 ALL TESTS PASSED!")
        print("The feature analysis system is ready to use.")
        print(f"{'='*50}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()