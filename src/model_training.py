import pandas as pd
import joblib

from pathlib import Path
import yaml
import dotenv
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

def main(dataset_name: str, model_name: str, model_type: str = "lightgbm"):
    print(f"Starting {model_type.upper()} training with preprocessed data...")
    print(f"Current working directory: {os.getcwd()}")

    # Load configuration
    config_path = os.path.join(Path(__file__).parent.parent, 'config', 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Please check the path.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("Configuration loaded successfully")

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
    X_train, X_val, X_test, selected_features = lasso_feature_selection(X_train, y_train, X_val, X_test, config)
    
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
        X_test_orig = test_data.drop(columns=drop_cols, errors='ignore')
        
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
    if time_column_madrid.dt.tz is not None and str(time_column_madrid.dt.tz) == 'UTC':
        time_column_madrid = time_column_madrid.dt.tz_convert('Europe/Madrid')
    
    submit_df = pd.DataFrame({
        'time': time_column_madrid,
        'target': test_predictions
    })
    submit_df.to_csv(predictions_path, index=False, header=False)
    print(f"Test predictions saved to {predictions_path}")
    
    if hasattr(model, 'best_iteration') and model.best_iteration:
        print(f"Best iteration: {model.best_iteration}")
    
    # Save the model
    model_path = Path(config['data_path']['model_checkpoints']) / f'{model_name}'
    os.makedirs(model_path.parent, exist_ok=True)
    
    if model_type.lower() == "lightgbm":
        # Save LightGBM model using joblib
        pkl_path = model_path.with_suffix('.pkl')
        joblib.dump(model, pkl_path)
        print(f"\nModel saved to {pkl_path}")
        
        # Save model using LightGBM native format as well
        native_model_path = model_path.with_suffix('.txt')
        model.save_model(str(native_model_path))
        print(f"Model also saved in LightGBM native format to {native_model_path}")
        
    elif model_type.lower() == "lstm":
        # Save LSTM model using PyTorch format
        pth_path = model_path.with_suffix('.pth')
        model.save_model(str(pth_path))
        print(f"\nLSTM model saved to {pth_path}")
        
    elif model_type.lower() == "ensemble":
        # Save ensemble configuration
        ensemble_path = model_path.with_suffix('.json')
        model.save_ensemble(str(ensemble_path))
        print(f"\nEnsemble configuration saved to {ensemble_path}")
        
        # Individual models are saved internally during ensemble training
        print("Individual ensemble models saved in their respective formats")
    
    print("\nTraining and prediction completed successfully!")
    print(f"Summary:")
    print(f"  - Model trained on {len(X_train)} samples")
    print(f"  - Validated on {len(X_val)} samples")
    print(f"  - Generated predictions for {len(X_test)} test samples")
    print(f"  - Final validation RMSE: {val_scores['rmse']:.4f}")

if __name__ == "__main__":
    dataset_name = "dataset_20250625_01"  # Example dataset name, replace with actual if needed
    
    # Choose model type: "xgboost", "lstm", or "ensemble"
    model_type = "ensemble"  # Change to "lstm" for LSTM model or "ensemble" for ensemble training
    
    model_name = f"{model_type}_model_{dataset_name}_002"
    main(dataset_name, model_name=model_name, model_type=model_type)