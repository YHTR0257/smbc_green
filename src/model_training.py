import pandas as pd
import joblib

from pathlib import Path
import yaml
import dotenv
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models import XGBoost
from src.data_processing import DataProcessor

# Load environment variables
dotenv.load_dotenv()


def load_data(file_path: Path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def train_model(X_train, y_train, X_val, y_val, config: dict):
    """Train the model using the training data with validation."""
    model = XGBoost(config)  # Initialize the model
    model.train(X_train, y_train, X_val, y_val)  # Fit the model with validation
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    scores = model.score(X_test, y_test)
    return scores

def main():
    print("Starting XGBoost training with time-based validation...")
    print(f"Current working directory: {os.getcwd()}")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}. Please check the path.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    print("Configuration loaded successfully")
    
    # Initialize data processor with time-based splitting
    processor = DataProcessor(
        train_file_path=Path(config['data_path']['raw_data']) / 'train.csv',
        test_file_path=Path(config['data_path']['raw_data']) / 'test.csv',
        config=config
    )
    
    print("DataProcessor initialized")
    
    # Process data with time-based split (2015-2016 train, 2017 validation)
    train_data, val_data = processor.process_with_time_split()
    
    print(f"Data processed:")
    print(f"  Train data shape: {train_data.shape}")
    print(f"  Validation data shape: {val_data.shape}")
    
    # Prepare features and targets
    target_col = config['train_config']['general']['target']
    
    # Separate features and targets
    X_train = train_data.drop(columns=[target_col, 'time'], errors='ignore')
    y_train = train_data[target_col]
    X_val = val_data.drop(columns=[target_col, 'time'], errors='ignore')
    y_val = val_data[target_col]
    
    # Clean target variables - remove NaN values
    print(f"Before cleaning - Train NaNs: {y_train.isna().sum()}, Val NaNs: {y_val.isna().sum()}")
    
    train_valid_mask = ~y_train.isna()
    val_valid_mask = ~y_val.isna()
    
    X_train = X_train[train_valid_mask]
    y_train = y_train[train_valid_mask]
    X_val = X_val[val_valid_mask]
    y_val = y_val[val_valid_mask]
    
    print(f"Features prepared:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape: {X_val.shape}")
    print(f"  Target data cleaned successfully")
    
    if X_train.empty or X_val.empty:
        raise ValueError("Training or validation data is empty after processing. Please check the data.")
    
    # Get XGBoost configuration
    xgb_config = config['train_config']['xgboost']
    
    # Train the model with validation
    print("Training XGBoost model...")
    model = train_model(X_train, y_train, X_val, y_val, xgb_config)
    
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
    
    if model.best_iteration:
        print(f"Best iteration: {model.best_iteration}")
    
    # Save the model
    model_path = Path(config['data_path']['model_checkpoints']) / 'xgboost_model.pkl'
    os.makedirs(model_path.parent, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save model using XGBoost native format as well
    native_model_path = Path(config['data_path']['model_checkpoints']) / 'xgboost_model.json'
    model.save_model(str(native_model_path))
    print(f"Model also saved in XGBoost native format to {native_model_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()