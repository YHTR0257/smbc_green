import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path

def load_data(file_path)-> pd.DataFrame:
    """Load data from a specified file path."""
    import pandas as pd
    return pd.read_csv(file_path)

def save_data(data, file_path):
    """Save data to a specified file path."""
    data.to_csv(file_path, index=False)

def cal_discomfort(df: pd.DataFrame):
    """Calculate discomfort index based on temperature and humidity."""
    areas = ["bilbao", "barcelona", "seville", "madrid", "valencia"]
    for area in areas:
        temp_col = f"{area}_temp"
        humidity_col = f"{area}_humidity"
        temp_max_col = f"{area}_temp_max"
        rain_col = f"{area}_rain"
        
        # Only calculate if required columns exist
        if temp_col in df.columns and humidity_col in df.columns:
            df[f"{area}_discomfort1"] = df[temp_col] * 0.81 + df[humidity_col] * 0.01 * (0.99 * df[temp_col] - 14.3) + 46.3
            
        if temp_max_col in df.columns and humidity_col in df.columns and rain_col in df.columns:
            df[f"{area}_discomfort2"] = df[temp_max_col] * 0.82 + df[humidity_col] * (0.98 * df[temp_max_col] - 14.4) * df[rain_col]
    
    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['discomfort'] = 0.81 * df['temperature'] + 0.01 * df['humidity'] * (0.99 * df['temperature'] - 14.3) + 46.3
    else:
        warnings.warn("Columns 'temperature' and 'humidity' are required for discomfort calculation.")
        df['discomfort'] = np.nan
    return df

def cal_gene_sum(df: pd.DataFrame):
    """Calculate the sum of gene expression levels."""
    gene_cols = [col for col in df.columns if 'gene' in col]
    df['gene_sum'] = df[gene_cols].sum(axis=1)
    return df

def preprocess_data(test_df: pd.DataFrame, train_df: pd.DataFrame):
    """Preprocess the df by handling missing values and encoding categorical variables.
    
    Args:
        test_df: The test dataframe to preprocess
        train_df: Training dataframe to use for normalization statistics
    """
    # Handle categorical variables - drop object type columns that are not useful for modeling
    categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
    # Keep time column, drop weather description columns that are string-based
    weather_text_cols = [col for col in categorical_cols if any(x in col for x in ['weather_main', 'weather_description', 'weather_icon'])]
    
    train_df = train_df.drop(columns=weather_text_cols, errors='ignore')
    test_df = test_df.drop(columns=weather_text_cols, errors='ignore')
    
    # Get numeric columns for scaling
    scale_cols = test_df.select_dtypes(include=['number']).columns.tolist()
    test_scaled_cols = []
    train_scaled_cols = []

    # Discomfort cols
    train_df = cal_discomfort(train_df)
    test_df = cal_discomfort(test_df)
    
    for col in scale_cols:
        if col in train_df.columns:
            # Calculate statistics from reference dataframe
            mean_col = train_df[col].mean()
            std_col = train_df[col].std(ddof=0)

            # Apply normalization to target dataframe
            if std_col != 0:
                test_scaled = (test_df[col] - mean_col) / std_col
                train_scaled = (train_df[col] - mean_col) / std_col
            else:
                test_scaled = test_df[col] * 0  # Set to 0 if std is 0
                train_scaled = train_df[col] * 0  # Set to 0 if std is 0

            test_scaled_cols.append(pd.DataFrame({
                f"{col}_mean_to_t": [mean_col] * len(test_df),
                f"{col}_std_to_t": [std_col] * len(test_df),
                f"{col}_scaled": test_scaled,
            }, index=test_df.index))
            train_scaled_cols.append(pd.DataFrame({
                f"{col}_mean_to_t": [mean_col] * len(train_df),
                f"{col}_std_to_t": [std_col] * len(train_df),
                f"{col}_scaled": train_scaled,
            }, index=train_df.index))
    
    train_df = pd.concat([train_df] + train_scaled_cols, axis=1)
    test_df = pd.concat([test_df] + test_scaled_cols, axis=1)
    
    # Fill any remaining NaN values
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    
    return test_df, train_df


def split_data(data, train_size=0.8):
    """Split the data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, train_size=train_size)
    return train_data, test_data

def time_based_split(data, config):
    """Split data based on time periods for time series validation.
    
    Args:
        data: DataFrame with time column
        config: Configuration dictionary with time split settings
        
    Returns:
        train_data, validation_data: Split datasets
    """
    if not config.get('use_time_split', False):
        return split_data(data, train_size=0.8)
    
    time_col = config.get('time_column', 'time')
    train_years = config.get('train_years', [2015, 2016])
    val_year = config.get('validation_year', 2017)
    
    # Make a copy to avoid modifying original data
    data_copy = data.copy()
    
    # Convert time column to datetime if it's not already
    if time_col not in data_copy.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    
    if not pd.api.types.is_datetime64_any_dtype(data_copy[time_col]):
        data_copy[time_col] = pd.to_datetime(data_copy[time_col], utc=True)
    
    # Extract year from time column
    data_copy['year'] = data_copy[time_col].dt.year
    
    # Split based on years
    train_mask = data_copy['year'].isin(train_years)
    val_mask = data_copy['year'] == val_year
    
    train_data = data_copy[train_mask].copy()
    val_data = data_copy[val_mask].copy()
    
    # Remove the temporary year column
    train_data = train_data.drop('year', axis=1)
    val_data = val_data.drop('year', axis=1)
    
    print(f"Train data: {len(train_data)} samples from years {train_years}")
    print(f"Validation data: {len(val_data)} samples from year {val_year}")
    
    return train_data, val_data

class DataProcessor:
    """Class for handling data processing tasks."""
    
    def __init__(self, train_file_path:Path, test_file_path:Path, config:dict):
        self.train_data = load_data(train_file_path)
        self.test_data = load_data(test_file_path)
        self.config = config
    
    def process(self):
        """Process the data."""
        self.train_data = self.train_data.drop(columns=self.config.get('train_config', {}).get('general', {}).get('drop_columns', []), errors='ignore')
        self.test_data = self.test_data.drop(columns=self.config.get('test_config', {}).get('general', {}).get('drop_columns', []), errors='ignore')
        _target = self.train_data[self.config['train_config']['general']['target']]
        self.train_data.drop(columns=[self.config['train_config']['general']['target']], inplace=True)
        
        # Process training data first
        self.train_data, self.test_data = preprocess_data(self.test_data, self.train_data)
        
        self.train_data = pd.concat([self.train_data, _target], axis=1)
        return self.train_data, self.test_data
    
    def process_with_time_split(self):
        """Process the data with time-based train/validation split."""
        general_config = self.config.get('train_config', {}).get('general', {})
        
        # Drop specified columns
        drop_columns = general_config.get('drop_columns', [])
        self.train_data = self.train_data.drop(columns=drop_columns, errors='ignore')
        
        # Extract target before preprocessing
        target_col = general_config.get('target', 'price_actual')
        target_data = self.train_data[target_col].copy()
        features_data = self.train_data.drop(columns=[target_col])
        
        # Time-based split
        train_features, val_features = time_based_split(features_data, general_config)
        
        # Get corresponding target values for train/val splits
        train_indices = train_features.index
        val_indices = val_features.index
        train_target = target_data.loc[train_indices]
        val_target = target_data.loc[val_indices]
        
        # Preprocess features using train data for normalization
        train_features_processed, val_features_processed = preprocess_data(val_features, train_features)
        
        # Combine features and target
        train_data_final = pd.concat([train_features_processed, train_target], axis=1)
        val_data_final = pd.concat([val_features_processed, val_target], axis=1)
        
        return train_data_final, val_data_final

    def save_processed_data(self, output_path:Path):
        """Save the processed data to the specified output path."""
        train_path = Path(output_path) / 'train_data.csv'
        test_path = Path(output_path) / 'test_data.csv'
        save_data(self.train_data, train_path)
        save_data(self.test_data, test_path)