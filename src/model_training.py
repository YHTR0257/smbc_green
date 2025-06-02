# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from src.models import Model1  # Assuming a Model class is defined in models.py
from pathlib import Path
import yaml
import dotenv

# Load environment variables
dotenv.load_dotenv()


def load_data(file_path:Path):
    """Load dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def train_model(X_train, y_train, config:dict):
    """Train the model using the training data."""
    model = Model1(config=config)  # Initialize the model
    model.fit(X_train, y_train)  # Fit the model
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test data."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

def main():
    # Load configuration
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    train_config = dict(**config['train_config']['general'], 
                        **config['train_config']['model1']) # Adjust based on the model you want to train
    
    # Load data
    data = load_data(config['data_path']['processed_data'] + 'dataset.csv') # Processed data path
    if data.empty:
        raise ValueError("Loaded data is empty. Please check the data path and file.")
    X = data.drop(train_config['target'], axis=1)  # Assuming train_config['target'] is the label column
    y = data[train_config['target']]

    # Split data into training, validation, and testing sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=config['test_size'], random_state=config['random_seed'][0])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config['validation_size'], random_state=config['random_seed'][0])
    
    # Train the model
    model = train_model(X_train, y_train, train_config)
    
    # Evaluate the model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Model accuracy: {accuracy:.2f}')
    
    # Save the model
    joblib.dump(model, config['data_path']['model_checkpoints'] + 'model.pkl')  # Save the model to the specified path
    print(f'Model saved to {config["data_path"]["model_checkpoints"] + "model.pkl"}')

if __name__ == "__main__":
    main()