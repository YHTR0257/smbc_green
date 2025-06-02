import pandas as pd
from sklearn.model_selection import train_test_split
import os
import datetime

def load_data(file_path):
    """Load data from a given file path."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save data to a given file path."""
    data.to_csv(file_path, index=False)

def split_data(data, train_size=0.8):
    """Split data into training and testing sets."""
    train, test = train_test_split(data, train_size=train_size)
    return train, test

def create_directory(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_message(message):
    """Log a message to the console."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} - {message}")
    