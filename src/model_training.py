import pandas as pd
import joblib

from pathlib import Path
import yaml
import dotenv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import XGBoost

# Load environment variables
dotenv.load_dotenv()


def load_data(file_path: Path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def train_model(X_train, y_train, X_val, y_val, config: dict, optimize=False):
    """Train the model using the training data with validation."""
    model = XGBoost(config)  # Initialize the model
    
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
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    scores = model.score(X_test, y_test)
    return scores

def main(dataset_name: str):
    print("Starting XGBoost training with preprocessed data...")
    print(f"Current working directory: {os.getcwd()}")
    model_name = f"xgboost_model_{dataset_name}"

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
    
    # Get XGBoost configuration
    xgb_config = config['train_config']['xgboost']
    
    # Check if hyperparameter optimization is enabled
    optuna_enabled = xgb_config.get('optuna', {}).get('enabled', False)
    
    # Train the model with validation
    if optuna_enabled:
        print("Training XGBoost model with hyperparameter optimization...")
        model = train_model(X_train, y_train, X_val, y_val, xgb_config, optimize=True)
    else:
        print("Training XGBoost model...")
        model = train_model(X_train, y_train, X_val, y_val, xgb_config, optimize=False)
    
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
    
    if model.best_iteration:
        print(f"Best iteration: {model.best_iteration}")
    
    # Save the model
    model_path = Path(config['data_path']['model_checkpoints']) / f'{model_name}.pkl'
    os.makedirs(model_path.parent, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save model using XGBoost native format as well
    native_model_path = Path(config['data_path']['model_checkpoints']) / f'{model_name}.json'
    model.save_model(str(native_model_path))
    print(f"Model also saved in XGBoost native format to {native_model_path}")
    
    print("\nTraining and prediction completed successfully!")
    print(f"Summary:")
    print(f"  - Model trained on {len(X_train)} samples")
    print(f"  - Validated on {len(X_val)} samples")
    print(f"  - Generated predictions for {len(X_test)} test samples")
    print(f"  - Final validation RMSE: {val_scores['rmse']:.4f}")

if __name__ == "__main__":
    dataset_name = "dataset_20250625_01"  # Example dataset name, replace with actual if needed
    main(dataset_name)