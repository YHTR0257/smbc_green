import pandas as pd
import joblib
import yaml
import dotenv
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import LightGBM, LSTMTrainer, EnsembleModel, PYTORCH_AVAILABLE

# Load environment variables
dotenv.load_dotenv()


def load_data(file_path: Path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def get_feature_groups() -> Dict[str, List[str]]:
    """Define feature groups for incremental analysis."""
    return {
        "baseline": [
            "generation_*", "total_load_*", "total_supply", "supply_*", 
            "*_temp", "*_pressure", "*_humidity", "*_wind_speed"
        ],
        "weather_icon": [
            "*_icon_number", "*_is_day", "*_has_time_info", 
            "*_is_clear", "*_has_clouds", "*_has_precipitation", "*_is_extreme"
        ],
        "weather_main": [
            "*_main_*"
        ],
        "weather_description": [
            "*_desc_*", "*_precipitation_intensity", 
            "*_has_shower", "*_has_proximity", "*_is_ragged"
        ],
        "temporal": [
            "year", "month", "day", "hour", "day_of_week", "is_weekend", "is_weekday", "is_evening"
        ],
        "discomfort": [
            "*_discomfort*", "*_diff_temp"
        ],
        "festivals": [
            "*_festival*", "*_holiday*", "is_national_holiday", "holiday_name"
        ],
        "population": [
            "*_population*", "total_population"
        ],
        "supply_demand": [
            "*_supply*", "*_demand*", "*_ratio", "*_balance*", "*_sufficiency*", "*_surplus*"
        ],
        "interactions": [
            "*_amplified", "*_weighted", "*_reduction", "*_displacement", "*_effect"
        ]
    }

def expand_feature_patterns(pattern_list: List[str], available_columns: List[str]) -> List[str]:
    """Expand wildcard patterns to actual column names."""
    expanded_features = []
    
    for pattern in pattern_list:
        if '*' in pattern:
            # Convert wildcard pattern to regex
            regex_pattern = pattern.replace('*', '.*')
            regex = re.compile(f'^{regex_pattern}$')
            
            # Find matching columns
            matches = [col for col in available_columns if regex.match(col)]
            expanded_features.extend(matches)
        else:
            # Exact match
            if pattern in available_columns:
                expanded_features.append(pattern)
    
    # Remove duplicates while preserving order
    return list(dict.fromkeys(expanded_features))

def get_features_by_group(group_name: str, available_columns: List[str]) -> List[str]:
    """Get actual feature names for a specific group."""
    feature_groups = get_feature_groups()
    
    if group_name not in feature_groups:
        raise ValueError(f"Unknown feature group: {group_name}. Available groups: {list(feature_groups.keys())}")
    
    return expand_feature_patterns(feature_groups[group_name], available_columns)

def train_and_evaluate_subset(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, 
                             feature_subset: List[str], config: Dict[str, Any], model_type: str = "lightgbm") -> Dict[str, Any]:
    """Train and evaluate model with a specific feature subset."""
    # Select features
    X_train_subset = X_train[feature_subset]
    X_val_subset = X_val[feature_subset]
    
    print(f"Training with {len(feature_subset)} features...")
    
    # Train model
    model_config = config['train_config'][model_type]
    model = train_model(X_train_subset, y_train, X_val_subset, y_val, model_config, model_type, optimize=False)
    
    # Evaluate
    val_scores = evaluate_model(model, X_val_subset, y_val)
    
    return {
        'model': model,
        'val_rmse': val_scores['rmse'],
        'val_r2': val_scores['r2'],
        'val_mae': val_scores['mae'],
        'feature_count': len(feature_subset),
        'features': feature_subset
    }

def run_incremental_feature_analysis(dataset_name: str, cumulative: bool, config: dict, 
                                    model_type: str = "lightgbm", baseline_group: str = "baseline",) -> List[Dict[str, Any]]:
    """Run incremental feature analysis to measure contribution of each feature group."""
    
    print(f"Starting incremental feature analysis...")
    print(f"Dataset: {dataset_name}, Model: {model_type}, Baseline: {baseline_group}")
    print(f"Cumulative mode: {cumulative}")

    if not config:
        # Load default config
        config_path = os.path.join(Path(__file__).parent.parent, 'config', 'config.yml')
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    
    # Load data
    processed_data_path = Path(config['data_path']['processed_data'])
    train_path = processed_data_path / dataset_name / 'train_data.csv'
    val_path = processed_data_path / dataset_name / 'val_data.csv'
    
    print("Loading data...")
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    
    # Prepare features and targets
    target_col = config['train_config']['general']['target']
    drop_cols = [target_col, 'time', 'year']
    
    X_train = train_data.drop(columns=drop_cols, errors='ignore')
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=drop_cols, errors='ignore')
    y_val = val_data[target_col]
    
    available_columns = X_train.columns.tolist()
    
    # Get baseline features
    baseline_features = get_features_by_group(baseline_group, available_columns)
    
    print(f"\nBaseline features ({len(baseline_features)}): {baseline_group}")
    
    # Evaluate baseline
    print("\nEvaluating baseline...")
    baseline_result = train_and_evaluate_subset(X_train, y_train, X_val, y_val, 
                                               baseline_features, config, model_type)
    
    results = [{
        'group_added': baseline_group,
        'features_added': baseline_features,
        'total_features': baseline_features,
        **baseline_result
    }]
    
    print(f"Baseline RMSE: {baseline_result['val_rmse']:.4f}")
    
    # Get feature groups to test
    feature_groups = get_feature_groups()
    groups_to_test = [group for group in feature_groups.keys() if group != baseline_group]
    
    current_features = baseline_features.copy()
    
    # Test each feature group
    for group_name in groups_to_test:
        print(f"\n--- Testing feature group: {group_name} ---")
        
        # Get features for this group
        group_features = get_features_by_group(group_name, available_columns)
        
        if not group_features:
            print(f"No features found for group {group_name}, skipping...")
            continue
        
        # Features to use for this experiment
        if cumulative:
            # Add to existing features (cumulative)
            experiment_features = current_features + [f for f in group_features if f not in current_features]
            features_added = [f for f in group_features if f not in current_features]
        else:
            # Add only to baseline (non-cumulative)
            experiment_features = baseline_features + [f for f in group_features if f not in baseline_features]
            features_added = [f for f in group_features if f not in baseline_features]
        
        if not features_added:
            print(f"All features from {group_name} already included, skipping...")
            continue
            
        print(f"Adding {len(features_added)} new features from {group_name}")
        
        # Evaluate with added features
        result = train_and_evaluate_subset(X_train, y_train, X_val, y_val, 
                                          experiment_features, config, model_type)
        
        # Calculate improvement
        improvement = baseline_result['val_rmse'] - result['val_rmse']
        relative_improvement = improvement / baseline_result['val_rmse'] * 100
        
        result_entry = {
            'group_added': group_name,
            'features_added': features_added,
            'total_features': experiment_features,
            'rmse_improvement': improvement,
            'relative_improvement': relative_improvement,
            'features_added_count': len(features_added),
            'efficiency': improvement / len(features_added) if len(features_added) > 0 else 0,
            **result
        }
        
        results.append(result_entry)
        
        print(f"RMSE: {result['val_rmse']:.4f} (improvement: {improvement:+.4f}, {relative_improvement:+.2f}%)")
        print(f"Efficiency: {result_entry['efficiency']:.6f} improvement per feature")
        
        # Update current features for next iteration if cumulative
        if cumulative:
            current_features = experiment_features.copy()
    
    return results

def analyze_feature_contribution(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze feature group contribution from incremental analysis results."""
    if len(results) <= 1:
        return {"error": "Need at least baseline + 1 feature group for analysis"}
    
    baseline = results[0]
    experiments = results[1:]
    
    # Sort by improvement (best first)
    sorted_experiments = sorted(experiments, key=lambda x: x.get('rmse_improvement', 0), reverse=True)
    
    analysis = {
        "baseline": {
            "group": baseline['group_added'],
            "rmse": baseline['val_rmse'],
            "features": baseline['feature_count']
        },
        "contribution_ranking": [],
        "efficiency_ranking": [],
        "summary": {
            "best_improvement": None,
            "most_efficient": None,
            "total_experiments": len(experiments),
            "positive_improvements": 0,
            "negative_improvements": 0
        }
    }
    
    # Analyze each experiment
    for exp in sorted_experiments:
        improvement = exp.get('rmse_improvement', 0)
        relative_improvement = exp.get('relative_improvement', 0)
        efficiency = exp.get('efficiency', 0)
        features_added = exp.get('features_added_count', 0)
        
        contribution = {
            "group": exp['group_added'],
            "rmse_improvement": improvement,
            "relative_improvement": relative_improvement,
            "features_added": features_added,
            "efficiency": efficiency,
            "final_rmse": exp['val_rmse'],
            "final_r2": exp['val_r2'],
            "significant": abs(improvement) > 0.01  # Consider >0.01 RMSE change significant
        }
        
        analysis["contribution_ranking"].append(contribution)
        
        # Count improvements
        if improvement > 0:
            analysis["summary"]["positive_improvements"] += 1
        elif improvement < 0:
            analysis["summary"]["negative_improvements"] += 1
    
    # Efficiency ranking (separate sort)
    efficiency_sorted = sorted(experiments, key=lambda x: x.get('efficiency', 0), reverse=True)
    for exp in efficiency_sorted:
        efficiency_entry = {
            "group": exp['group_added'],
            "efficiency": exp.get('efficiency', 0),
            "rmse_improvement": exp.get('rmse_improvement', 0),
            "features_added": exp.get('features_added_count', 0)
        }
        analysis["efficiency_ranking"].append(efficiency_entry)
    
    # Summary statistics
    if sorted_experiments:
        analysis["summary"]["best_improvement"] = {
            "group": sorted_experiments[0]['group_added'],
            "improvement": sorted_experiments[0].get('rmse_improvement', 0),
            "relative": sorted_experiments[0].get('relative_improvement', 0)
        }
        
        best_efficiency = max(efficiency_sorted, key=lambda x: x.get('efficiency', 0))
        analysis["summary"]["most_efficient"] = {
            "group": best_efficiency['group_added'],
            "efficiency": best_efficiency.get('efficiency', 0),
            "improvement": best_efficiency.get('rmse_improvement', 0)
        }
    
    return analysis

def generate_feature_contribution_report(results: List[Dict[str, Any]], analysis: Dict[str, Any], 
                                        output_path: str, model_type: str = "lightgbm") -> str:
    """Generate detailed feature contribution analysis report."""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_content = f"""# Feature Group Contribution Analysis Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Experiment Configuration
- **Model Type**: {model_type.upper()}
- **Baseline Group**: {analysis['baseline']['group']}
- **Baseline RMSE**: {analysis['baseline']['rmse']:.4f}
- **Baseline Features**: {analysis['baseline']['features']}
- **Total Experiments**: {analysis['summary']['total_experiments']}

## Summary
- **Positive Improvements**: {analysis['summary']['positive_improvements']} groups
- **Negative Improvements**: {analysis['summary']['negative_improvements']} groups
"""

    if analysis['summary']['best_improvement']:
        best = analysis['summary']['best_improvement']
        report_content += f"- **Best Improvement**: {best['group']} ({best['improvement']:+.4f} RMSE, {best['relative']:+.2f}%)\n"
    
    if analysis['summary']['most_efficient']:
        efficient = analysis['summary']['most_efficient']
        report_content += f"- **Most Efficient**: {efficient['group']} ({efficient['efficiency']:.6f} improvement/feature)\n"

    report_content += """
## Contribution Ranking (by RMSE Improvement)

| Rank | Feature Group | RMSE Improvement | Relative (%) | Features Added | Efficiency | Final RMSE | R² | Significant |
|------|---------------|------------------|--------------|----------------|------------|------------|----|-----------| 
"""
    
    for i, contrib in enumerate(analysis['contribution_ranking'], 1):
        significant = "✓" if contrib['significant'] else "-"
        report_content += f"| {i} | {contrib['group']} | {contrib['rmse_improvement']:+.4f} | {contrib['relative_improvement']:+.2f}% | {contrib['features_added']} | {contrib['efficiency']:.6f} | {contrib['final_rmse']:.4f} | {contrib['final_r2']:.3f} | {significant} |\n"

    report_content += """
## Efficiency Ranking (by Improvement per Feature)

| Rank | Feature Group | Efficiency | RMSE Improvement | Features Added |
|------|---------------|------------|------------------|----------------|
"""
    
    for i, eff in enumerate(analysis['efficiency_ranking'], 1):
        report_content += f"| {i} | {eff['group']} | {eff['efficiency']:.6f} | {eff['rmse_improvement']:+.4f} | {eff['features_added']} |\n"

    # Detailed analysis section
    report_content += """
## Detailed Analysis

### Highly Contributing Groups
"""
    
    significant_groups = [c for c in analysis['contribution_ranking'] if c['significant'] and c['rmse_improvement'] > 0]
    if significant_groups:
        for contrib in significant_groups[:3]:  # Top 3
            report_content += f"- **{contrib['group']}**: {contrib['rmse_improvement']:+.4f} RMSE improvement ({contrib['relative_improvement']:+.2f}%) with {contrib['features_added']} features\n"
    else:
        report_content += "- No significantly contributing feature groups found (>0.01 RMSE improvement)\n"

    report_content += """
### Recommendations

"""
    
    # Generate recommendations based on analysis
    recommendations = []
    
    # Best absolute improvement
    if analysis['summary']['best_improvement'] and analysis['summary']['best_improvement']['improvement'] > 0:
        best_group = analysis['summary']['best_improvement']['group']
        recommendations.append(f"1. **Include {best_group}**: Provides the largest absolute improvement ({analysis['summary']['best_improvement']['improvement']:+.4f} RMSE)")
    
    # Best efficiency
    if analysis['summary']['most_efficient'] and analysis['summary']['most_efficient']['efficiency'] > 0:
        efficient_group = analysis['summary']['most_efficient']['group']
        if efficient_group != analysis['summary']['best_improvement']['group']:
            recommendations.append(f"2. **Consider {efficient_group}**: Most efficient feature group ({analysis['summary']['most_efficient']['efficiency']:.6f} improvement/feature)")
    
    # Negative contributors
    negative_groups = [c for c in analysis['contribution_ranking'] if c['rmse_improvement'] < -0.005]  # More than 0.005 degradation
    if negative_groups:
        for neg in negative_groups[:2]:  # Top 2 worst
            recommendations.append(f"- **Avoid {neg['group']}**: Degrades performance ({neg['rmse_improvement']:+.4f} RMSE)")
    
    # Low efficiency warning
    low_efficiency = [c for c in analysis['contribution_ranking'] if 0 < c['rmse_improvement'] < 0.01 and c['features_added'] > 10]
    if low_efficiency:
        for low in low_efficiency[:2]:
            recommendations.append(f"- **Question {low['group']}**: Small improvement ({low['rmse_improvement']:+.4f}) for many features ({low['features_added']})")
    
    if not recommendations:
        recommendations.append("- No clear recommendations - all feature groups show minimal impact")
    
    for rec in recommendations:
        report_content += rec + "\n"

    # Feature details section
    report_content += """
## Feature Group Details

"""
    
    for result in results:
        if 'features_added' in result and isinstance(result['features_added'], list):
            report_content += f"### {result['group_added']}\n"
            report_content += f"**Features Added ({len(result['features_added'])}):**\n"
            if len(result['features_added']) <= 10:
                for feature in result['features_added']:
                    report_content += f"- {feature}\n"
            else:
                for feature in result['features_added'][:7]:
                    report_content += f"- {feature}\n"
                report_content += f"- ... and {len(result['features_added']) - 7} more\n"
            report_content += "\n"

    report_content += f"""
---
*Report generated by Feature Analysis System at {timestamp}*
"""

    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Feature contribution report saved to: {output_path}")
    return output_path

def train_model(X_train, y_train, X_val, y_val, config: dict, model_type="lightgbm", optimize=False):
    """Train the model using the training data with validation."""
    
    if model_type.lower() == "lightgbm":
        model = LightGBM(config)  # Initialize LightGBM model
        
        # Run hyperparameter optimization if requested
        if optimize:
            print("Running hyperparameter optimization...")
            model.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            
            # Get optimization results
            results = model.get_optimization_results()
            if results['best_params']:
                print("Optimization completed. Best parameters:")
                for param, value in results['best_params'].items():
                    print(f"  {param}: {value}")
        
        model.train(X_train, y_train, X_val, y_val)  # Fit the model with validation
        
    elif model_type.lower() == "lstm":
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is not available. Please install PyTorch to use LSTM models.")
        
        model = LSTMTrainer(config)  # Initialize LSTM trainer
        print("Training LSTM model...")
        model.train(X_train, y_train, X_val, y_val)  # Train LSTM model
        
    elif model_type.lower() == "xgboost":
        raise ValueError(f"XGBoost is no longer supported. Use 'lightgbm' instead.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'lightgbm', 'lstm'")
    
    return model

def lasso_feature_selection(X_train, y_train, X_val, X_test, config):
    """LASSO回帰による特徴量選択"""
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    
    # 設定を取得
    feature_config = config.get('feature_selection', {})
    if not feature_config.get('enabled', False):
        return X_train, X_val, X_test, X_train.columns.tolist()
    
    cv_folds = feature_config.get('lasso_cv_folds', 5)
    max_features = feature_config.get('max_features', 100)
    
    print(f"Running LASSO feature selection...")
    print(f"  Original features: {X_train.shape[1]}")
    
    # 標準化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # LASSO CV
    lasso = LassoCV(cv=cv_folds, random_state=42, max_iter=2000)
    lasso.fit(X_train_scaled, y_train)
    
    # 重要な特徴量を選択
    selected_mask = lasso.coef_ != 0
    selected_features = X_train.columns[selected_mask].tolist()
    
    # max_featuresを超える場合は係数の絶対値で上位を選択
    if len(selected_features) > max_features:
        feature_importance = abs(lasso.coef_[selected_mask])
        top_indices = feature_importance.argsort()[-max_features:]
        selected_features = [selected_features[i] for i in top_indices]
    
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Reduction rate: {(1 - len(selected_features)/X_train.shape[1])*100:.1f}%")
    
    # 選択された特徴量でデータセットを更新
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

def train_ensemble_model(X_train, y_train, X_val, y_val, config: dict):
    """Train ensemble model with multiple base models."""
    ensemble_config = config['ensemble']
    
    # Initialize ensemble
    ensemble = EnsembleModel(ensemble_config)
    
    # Prepare model configurations
    model_configs = {
        'lightgbm': config['train_config']['lightgbm'],
        'lstm': config['train_config']['lstm']
    }
    
    # Train ensemble
    print("Training ensemble models...")
    ensemble.train_ensemble(X_train, y_train, X_val, y_val, model_configs)
    
    # Optimize weights if enabled
    if ensemble_config.get('optimize_weights', True):
        print("Optimizing ensemble weights...")
        ensemble.optimize_weights(X_val, y_val)
    
    return ensemble

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    scores = model.score(X_test, y_test)
    return scores

def main(dataset_name: str, model_name: Optional[str] = None, model_type: str = "lightgbm", 
         run_feature_analysis: bool = False, baseline_group: str = "baseline", 
         cumulative: bool = False, analysis_only: bool = False) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    
    if run_feature_analysis:
        print(f"Starting Feature Analysis...")
        print(f"Dataset: {dataset_name}, Model: {model_type}, Baseline: {baseline_group}")
        print(f"Cumulative mode: {cumulative}, Analysis only: {analysis_only}")
    else:
        print(f"Starting {model_type.upper()} training with preprocessed data...")
    
    print(f"Current working directory: {os.getcwd()}")

    # Load configuration
    config_path = os.path.join(Path(__file__).parent.parent, 'config', 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Please check the path.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("Configuration loaded successfully")
    
    # Feature Analysis Mode
    if run_feature_analysis:
        print("\n" + "="*60)
        print("FEATURE CONTRIBUTION ANALYSIS")
        print("="*60)
        
        # Run incremental feature analysis
        results = run_incremental_feature_analysis(
            dataset_name=dataset_name,
            model_type=model_type,
            baseline_group=baseline_group,
            cumulative=cumulative,
            config=config
        )
        
        # Analyze contributions
        print("\nAnalyzing feature contributions...")
        analysis = analyze_feature_contribution(results)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"feature_analysis_{dataset_name}_{model_type}_{baseline_group}_{timestamp}.md"
        report_path = os.path.join(config['data_path']['reports'], report_filename)
        
        generate_feature_contribution_report(results, analysis, report_path, model_type)
        
        # Print summary to console
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        baseline_info = analysis.get('baseline', {})
        summary_info = analysis.get('summary', {})
        
        print(f"Baseline ({baseline_group}): {baseline_info.get('rmse', 0):.4f} RMSE with {baseline_info.get('features', 0)} features")
        
        best_improvement = summary_info.get('best_improvement')
        if best_improvement:
            print(f"Best improvement: {best_improvement.get('group', 'N/A')} ({best_improvement.get('improvement', 0):+.4f} RMSE, {best_improvement.get('relative', 0):+.2f}%)")
        
        most_efficient = summary_info.get('most_efficient')
        if most_efficient:
            print(f"Most efficient: {most_efficient.get('group', 'N/A')} ({most_efficient.get('efficiency', 0):.6f} improvement/feature)")
        
        print(f"\nPositive improvements: {summary_info.get('positive_improvements', 0)}")
        print(f"Negative improvements: {summary_info.get('negative_improvements', 0)}")
        print(f"\nDetailed report saved to: {report_path}")
        
        # If analysis_only mode, return here
        if analysis_only:
            print("\nFeature analysis completed. Exiting (analysis_only=True).")
            return results, analysis
        else:
            print("\nContinuing with full model training using baseline features...")
            # Continue with normal training using baseline
    
    # Generate model name if not provided
    if model_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_feature_analysis and not analysis_only:
            model_name = f"{model_type}_after_analysis_{dataset_name}_{timestamp}"
        else:
            model_name = f"{model_type}_model_{dataset_name}_{timestamp}"

    # Define paths to processed data
    processed_data_path = Path(config['data_path']['processed_data'])
    train_path = processed_data_path / dataset_name / 'train_data.csv'
    val_path = processed_data_path / dataset_name / 'val_data.csv'
    test_path = processed_data_path / dataset_name / 'test_data.csv'

    # Check if processed data files exist
    if not train_path.exists():
        raise FileNotFoundError(f"Processed train data not found at {train_path}. Please run data preprocessing first.")
    if not val_path.exists():
        raise FileNotFoundError(f"Processed validation data not found at {val_path}. Please run data preprocessing first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Processed test data not found at {test_path}. Please run data preprocessing first.")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    train_data = load_data(train_path)
    val_data = load_data(val_path)
    test_data = load_data(test_path)
    
    print(f"Data loaded successfully:")
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Validation data shape: {val_data.shape}")
    print(f"  Test data shape: {test_data.shape}")
    
    # Prepare features and targets
    target_col = config['train_config']['general']['target']
    
    # Separate features and targets for train and validation
    # Consistently drop the same columns across all datasets
    drop_cols = [target_col, 'time', 'year']
        
    X_train = train_data.drop(columns=drop_cols, errors='ignore')
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=drop_cols, errors='ignore')
    y_val = val_data[target_col]
    
    # Test data doesn't have target column
    time_column = test_data["time"].copy()
    X_test = test_data.drop(columns=drop_cols, errors='ignore')
    
    # Clean target variables - remove NaN values if any
    print(f"Checking for NaN values - Train: {y_train.isna().sum()}, Val: {y_val.isna().sum()}")
    
    if y_train.isna().sum() > 0 or y_val.isna().sum() > 0:
        print("Warning: Found NaN values in target data, cleaning...")
        train_valid_mask = ~y_train.isna()
        val_valid_mask = ~y_val.isna()
        
        X_train = X_train[train_valid_mask]
        y_train = y_train[train_valid_mask]
        X_val = X_val[val_valid_mask]
        y_val = y_val[val_valid_mask]
    
    print(f"Features prepared:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Data ready for training")
    
    if X_train.empty or X_val.empty:
        raise ValueError("Training or validation data is empty. Please check the preprocessed data.")
    
    # LASSO feature selection
    X_train, X_val, X_test, _ = lasso_feature_selection(X_train, y_train, X_val, X_test, config)
    
    print(f"Features after LASSO selection:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  X_test shape: {X_test.shape}")
    
    # Get model configuration based on model type
    if model_type.lower() == "lightgbm":
        model_config = config['train_config']['lightgbm']
        optuna_enabled = model_config.get('optuna', {}).get('enabled', False)
        
        # Train the model with validation
        if optuna_enabled:
            print("Training LightGBM model with hyperparameter optimization...")
            model = train_model(X_train, y_train, X_val, y_val, model_config, model_type, optimize=True)
        else:
            print("Training LightGBM model...")
            model = train_model(X_train, y_train, X_val, y_val, model_config, model_type, optimize=False)
            
    elif model_type.lower() == "lstm":
        model_config = config['train_config']['lstm']
        
        # LSTM doesn't use LASSO feature selection - needs time series structure
        print("Note: LSTM training requires time series structure, skipping LASSO feature selection...")
        
        # Use original features for LSTM (before LASSO selection)
        # Re-prepare features without LASSO
        X_train_orig = train_data.drop(columns=drop_cols, errors='ignore')
        X_val_orig = val_data.drop(columns=drop_cols, errors='ignore')
        
        # Add input_size to config for model creation
        model_config['input_size'] = X_train_orig.shape[1]
        
        print("Training LSTM model...")
        model = train_model(X_train_orig, y_train, X_val_orig, y_val, model_config, model_type, optimize=False)
        
        # Update X_train, X_val for evaluation
        X_train, X_val = X_train_orig, X_val_orig
        
    elif model_type.lower() == "ensemble":
        # Check if ensemble is enabled in config
        if not config.get('ensemble', {}).get('enabled', False):
            raise ValueError("Ensemble training is disabled in configuration. Set ensemble.enabled=true in config.yml")
        
        print("Training ensemble model...")
        
        # For ensemble, prepare both LASSO-selected and original features
        X_train_orig = train_data.drop(columns=drop_cols, errors='ignore')
        X_val_orig = val_data.drop(columns=drop_cols, errors='ignore')
        # X_test_orig not used in current implementation
        
        # Add input sizes to config
        lstm_config = config['train_config']['lstm'].copy()
        lstm_config['input_size'] = X_train_orig.shape[1]
        config['train_config']['lstm'] = lstm_config
        
        # Train ensemble model (it will handle feature selection internally)
        model = train_ensemble_model(X_train, y_train, X_val, y_val, config)
        
        # For test predictions, we'll need to prepare data accordingly
        X_test = X_test  # Keep LASSO-selected features for XGBoost models in ensemble
        
    elif model_type.lower() == "xgboost":
        raise ValueError(f"XGBoost is no longer supported. Use 'lightgbm' instead.")
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'lightgbm', 'lstm', 'ensemble'")
    
    # Evaluate the model on both training and validation sets
    print("\nEvaluating model performance...")
    train_scores = evaluate_model(model, X_train, y_train)
    val_scores = evaluate_model(model, X_val, y_val)
    
    print("\nModel Performance:")
    print(f"Training RMSE: {train_scores['rmse']:.4f}")
    print(f"Training R²: {train_scores['r2']:.4f}")
    print(f"Training MAE: {train_scores['mae']:.4f}")
    print(f"Validation RMSE: {val_scores['rmse']:.4f}")
    print(f"Validation R²: {val_scores['r2']:.4f}")
    print(f"Validation MAE: {val_scores['mae']:.4f}")
    
    # Generate predictions on test data
    print("\nGenerating test predictions...")
    test_predictions = model.predict(X_test)
    
    # Save test predictions
    predictions_path = Path(config['data_path']['submits']) /  f'{model_name}.csv'
    
    # Convert time back to Europe/Madrid timezone for submission
    time_column_madrid = pd.to_datetime(time_column)
    if hasattr(time_column_madrid, 'dt') and time_column_madrid.dt.tz is not None and str(time_column_madrid.dt.tz) == 'UTC':
        time_column_madrid = time_column_madrid.dt.tz_convert('Europe/Madrid')
    
    submit_df = pd.DataFrame({
        'time': time_column_madrid,
        'target': test_predictions
    })
    submit_df.to_csv(predictions_path, index=False, header=False)
    print(f"Test predictions saved to {predictions_path}")
    
    if hasattr(model, 'best_iteration') and getattr(model, 'best_iteration', None):
        print(f"Best iteration: {getattr(model, 'best_iteration', 'N/A')}")
    
    # Save the model
    model_path = Path(config['data_path']['model_checkpoints']) / f'{model_name}'
    os.makedirs(model_path.parent, exist_ok=True)
    
    try:
        if model_type.lower() == "lightgbm":
            # Save LightGBM model using joblib
            pkl_path = model_path.with_suffix('.pkl')
            joblib.dump(model, pkl_path)
            print(f"\nModel saved to {pkl_path}")
            
            # Save model using LightGBM native format as well
            if hasattr(model, 'save_model'):
                native_model_path = model_path.with_suffix('.txt')
                model.save_model(str(native_model_path))
                print(f"Model also saved in LightGBM native format to {native_model_path}")
            
        elif model_type.lower() == "lstm":
            # Save LSTM model using PyTorch format
            if hasattr(model, 'save_model'):
                pth_path = model_path.with_suffix('.pth')
                model.save_model(str(pth_path))
                print(f"\nLSTM model saved to {pth_path}")
            
        elif model_type.lower() == "ensemble":
            # Save ensemble configuration
            if hasattr(model, 'save_ensemble'):
                ensemble_path = model_path.with_suffix('.json')
                model.save_ensemble(str(ensemble_path))
                print(f"\nEnsemble configuration saved to {ensemble_path}")
                
                # Individual models are saved internally during ensemble training
                print("Individual ensemble models saved in their respective formats")
    except Exception as e:
        print(f"Warning: Could not save model - {e}")
    
    print("\nTraining and prediction completed successfully!")
    print(f"Summary:")
    print(f"  - Model trained on {len(X_train)} samples")
    print(f"  - Validated on {len(X_val)} samples")
    print(f"  - Generated predictions for {len(X_test)} test samples")
    print(f"  - Final validation RMSE: {val_scores['rmse']:.4f}")
    
    return None

if __name__ == "__main__":
    dataset_name = "dataset_20250625_01"  # Example dataset name, replace with actual if needed
    
    # Choose model type: "lightgbm", "lstm", or "ensemble"
    model_type = "lightgbm"  # Change to "lstm" for LSTM model or "ensemble" for ensemble training
    
    # Feature analysis options
    run_feature_analysis = True  # Set to True to run feature contribution analysis
    baseline_group = "baseline"  # Baseline feature group
    cumulative = False  # False: each group vs baseline, True: cumulative addition
    analysis_only = True  # True: only run analysis, False: continue with training after analysis
    
    if run_feature_analysis:
        print("Running feature contribution analysis...")
        result = main(
            dataset_name=dataset_name,
            model_type=model_type,
            run_feature_analysis=run_feature_analysis,
            baseline_group=baseline_group,
            cumulative=cumulative,
            analysis_only=analysis_only
        )
        if result is not None:
            results, analysis = result
        else:
            print("Analysis returned no results")
    else:
        # Normal training mode
        model_name = f"{model_type}_model_{dataset_name}_002"
        main(dataset_name, model_name=model_name, model_type=model_type)