import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

def load_data(file_path):
    """Load data from a specified file path."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save data to a specified file path."""
    data.to_csv(file_path, index=False)

def preprocess_data(data):
    """Preprocess the data by handling missing values and encoding categorical variables."""
    # Example preprocessing steps
    # data = pd.get_dummies(data)
    return data

def split_data(data, train_size=0.8):
    """Split the data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, train_size=train_size)
    return train_data, test_data

class DataProcessor:
    """Class for handling data processing tasks."""
    
    def __init__(self, train_file_path:Path, test_file_path:Path):
        self.train_data = load_data(train_file_path)
        self.test_data = load_data(test_file_path)
    
    def process(self):
        """Process the data."""
        self.train_data = preprocess_data(self.train_data)
        self.test_data = preprocess_data(self.test_data)
        return self.train_data, self.test_data

    def save_processed_data(self, output_path:Path):
        """Save the processed data to the specified output path."""
        save_data(self.train_data, output_path + '_train.csv')
        save_data(self.test_data, output_path + '_test.csv')