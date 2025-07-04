import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
import sys
import logging
from datetime import datetime
import re

from sklearn.preprocessing import LabelEncoder

# Add config directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config.config_loader import ConfigLoader

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
        temp_min_col = f"{area}_temp_min"
        rain_col = f"{area}_rain"

        # Change temp celcius to kelvin
        if temp_col in df.columns:
            df[temp_col] = df[temp_col] + 273.15
        if temp_max_col in df.columns:
            df[temp_max_col] = df[temp_max_col] + 273.15
        if temp_min_col in df.columns:
            df[temp_min_col] = df[temp_min_col] + 273.15
        
        # Only calculate if required columns exist
        if temp_col in df.columns and humidity_col in df.columns:
            df[f"{area}_discomfort1"] = df[temp_col] * 0.81 + df[humidity_col] * 0.01 * (0.99 * df[temp_col] - 14.3) + 46.3
        if temp_max_col in df.columns and humidity_col in df.columns and rain_col in df.columns:
            df[f"{area}_discomfort2"] = df[temp_max_col] * 0.82 + df[humidity_col] * (0.99 * df[temp_max_col] - 14.3) + 46.3
        if temp_min_col in df.columns and humidity_col in df.columns:
            df[f"{area}_discomfort3"] = df[temp_min_col] * 0.82 + df[humidity_col] * (0.99 * df[temp_min_col] - 14.3) + 46.3
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

    # Discomfort cols
    train_df = cal_discomfort(train_df)
    test_df = cal_discomfort(test_df)
    
    # Prepare dictionaries to collect all scaled columns at once
    test_scaled_dict = {}
    train_scaled_dict = {}
    
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

            # Store all columns in dictionaries
            test_scaled_dict[f"{col}_mean_to_t"] = [mean_col] * len(test_df)
            test_scaled_dict[f"{col}_std_to_t"] = [std_col] * len(test_df)
            test_scaled_dict[f"{col}_scaled"] = test_scaled
            
            train_scaled_dict[f"{col}_mean_to_t"] = [mean_col] * len(train_df)
            train_scaled_dict[f"{col}_std_to_t"] = [std_col] * len(train_df)
            train_scaled_dict[f"{col}_scaled"] = train_scaled
    
    # Create single DataFrames with all scaled columns
    if test_scaled_dict:  # Only if there are columns to add
        test_scaled_df = pd.DataFrame(test_scaled_dict, index=test_df.index)
        train_scaled_df = pd.DataFrame(train_scaled_dict, index=train_df.index)
        
        train_df = pd.concat([train_df, train_scaled_df], axis=1)
        test_df = pd.concat([test_df, test_scaled_df], axis=1)
    
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
    
    def __init__(self, train_file_path:Path, test_file_path:Path, config:dict, fes_config:dict={}):
        self.train_data = load_data(train_file_path)
        self.test_data = load_data(test_file_path)
        self.config = config
        self.fes_config = fes_config if fes_config else {}
        
        # Initialize config loader for festival and holiday data
        self.config_loader = ConfigLoader()
        
        # Load configuration data
        try:
            self.cities_config = self.config_loader.get_cities_config()
            self.feature_config = self.config_loader.get_feature_engineering_config()
            self.population_admin = self.config_loader.get_population_config("admin")  
            self.population_metro = self.config_loader.get_population_config("metro")
            self.weather_config = self.config_loader.get_weather_config()
            
            # Set population type based on config
            self.population_type = config.get('population_type', 'admin')
            self.population_weights = self.population_admin if self.population_type == 'admin' else self.population_metro
            
            logging.info(f"DataProcessor initialized with population type: {self.population_type}")
            
        except Exception as e:
            logging.warning(f"Failed to load configuration: {e}")
            self.cities_config = {}
            self.feature_config = {}
            self.population_weights = {}
            self.weather_config = {}
    
    def cal_discomfort(self):
        """Calculate discomfort index based on temperature and humidity."""
        areas = ["bilbao", "barcelona", "seville", "madrid", "valencia"]
        for df in [self.train_data, self.test_data]:
            # Collect all new columns to add at once
            new_columns = {}
            
            for area in areas:
                temp_col = f"{area}_temp"
                humidity_col = f"{area}_humidity"
                temp_max_col = f"{area}_temp_max"
                temp_min_col = f"{area}_temp_min"
                rain_col = f"{area}_rain"

                # Change temp celsius to kelvin (modify existing columns)
                if temp_col in df.columns:
                    df[temp_col] = df[temp_col] + 273.15
                if temp_max_col in df.columns:
                    df[temp_max_col] = df[temp_max_col] + 273.15
                if temp_min_col in df.columns:
                    df[temp_min_col] = df[temp_min_col] + 273.15
                
                # Calculate new columns and store in dictionary
                if temp_col in df.columns and humidity_col in df.columns:
                    new_columns[f"{area}_discomfort1"] = df[temp_col] * 0.81 + df[humidity_col] * 0.01 * (0.99 * df[temp_col] - 14.3) + 46.3
                
                if temp_max_col in df.columns and humidity_col in df.columns and rain_col in df.columns:
                    new_columns[f"{area}_discomfort2"] = df[temp_max_col] * 0.82 + df[humidity_col] * (0.99 * df[temp_max_col] - 14.3) + 46.3
                
                if temp_min_col in df.columns and humidity_col in df.columns:
                    new_columns[f"{area}_discomfort3"] = df[temp_min_col] * 0.82 + df[humidity_col] * (0.99 * df[temp_min_col] - 14.3) + 46.3
                
                if temp_min_col in df.columns and temp_max_col in df.columns:
                    new_columns[f"{area}_diff_temp"] = df[temp_max_col] - df[temp_min_col]
            
            # Add all new columns at once if any were calculated
            if new_columns:
                new_df = pd.DataFrame(new_columns, index=df.index)
                # Update the original dataframe reference
                if df is self.train_data:
                    self.train_data = pd.concat([df, new_df], axis=1)
                else:
                    self.test_data = pd.concat([df, new_df], axis=1)

    def cal_gene_sum(self):
        """Calculate the sum of gene expression levels."""
        for i, df in enumerate([self.train_data, self.test_data]):
            # Identify gene columns and calculate their sum
            gene_cols = [col for col in df.columns if 'gene' in col]
            if gene_cols:  # Only add if gene columns exist
                new_df = pd.DataFrame({'gene_sum': df[gene_cols].sum(axis=1)}, index=df.index)
                # Update the original dataframe reference
                if i == 0:  # train_data
                    self.train_data = pd.concat([df, new_df], axis=1)
                else:  # test_data
                    self.test_data = pd.concat([df, new_df], axis=1)

    def create_festival_calendar(self):
        _combined_df = pd.concat([self.train_data, self.test_data], ignore_index=True)
        split_index = len(self.train_data)
        cities = ["bilbao", "barcelona", "seville", "madrid", "valencia"]
        for city in cities:
            _combined_df = self.add_city_festivals(_combined_df, city)
        
        self.train_data = _combined_df.iloc[:split_index].reset_index(drop=True)
        self.test_data = _combined_df.iloc[split_index:].reset_index(drop=True)


    def add_city_festivals(self, df:pd.DataFrame, city:str, years:list = [2015, 2016, 2017, 2018]):
        """Add city festival information to the datasets.
        Focus on whether any festival is happening in the city, not individual festivals.
        
        Args:
            df: DataFrame to add festival features to
            city: Name of the city to add festivals for
            years: List of years to process
            
        Returns:
            DataFrame with festival features added
        """
        if 'time' not in df.columns:
            logging.warning("Time column not found, skipping festival features")
            return df
            
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'], utc=True)
        
        # Capitalize city name to match config
        city_cap = city.capitalize()
        if city_cap == 'Seville':
            city_cap = 'Sevilla'
            
        # Initialize city festival columns
        festival_col = f"{city}_has_festival"
        festival_intensity_col = f"{city}_festival_intensity"
        outdoor_impact_col = f"{city}_festival_outdoor_impact"
        
        if festival_col not in df.columns:
            df[festival_col] = 0
        if festival_intensity_col not in df.columns:
            df[festival_intensity_col] = 0.0
        if outdoor_impact_col not in df.columns:
            df[outdoor_impact_col] = 0.0
            
        # Get scale impact weights from config
        scale_impact = self.feature_config.get('scale_impact', {'small': 0.1, 'medium': 0.3, 'large': 0.6})
        
        for year in years:
            try:
                # Get festival data for this year
                festivals_data = self.config_loader.get_festivals_config(year)
                
                for festival_name, festival_info in festivals_data.items():
                    # Check if this city is involved in the festival
                    primary_cities = festival_info.get('primary_cities', [])
                    if city_cap not in primary_cities:
                        continue
                        
                    # Parse festival dates
                    start_date = pd.to_datetime(festival_info['start_date']).date()
                    end_date = pd.to_datetime(festival_info['end_date']).date()
                    scale = festival_info.get('scale', 'medium')
                    outdoor_rate = festival_info.get('outdoor_rate', 0.5)
                    
                    # Create date masks for this year
                    year_mask = df['time'].dt.year == year
                    date_mask = (df['time'].dt.date >= start_date) & (df['time'].dt.date <= end_date)
                    combined_mask = year_mask & date_mask
                    
                    # Mark that city has festival during this period
                    df.loc[combined_mask, festival_col] = 1
                    
                    # Accumulate festival intensity (can have multiple festivals)
                    scale_weight = scale_impact.get(scale, 0.3)
                    current_intensity = df.loc[combined_mask, festival_intensity_col]
                    df.loc[combined_mask, festival_intensity_col] = np.maximum(
                        current_intensity, 
                        scale_weight
                    )
                    
                    # Accumulate outdoor impact
                    outdoor_impact = scale_weight * outdoor_rate
                    current_outdoor = df.loc[combined_mask, outdoor_impact_col]
                    df.loc[combined_mask, outdoor_impact_col] = np.maximum(
                        current_outdoor, 
                        outdoor_impact
                    )
                    
            except Exception as e:
                logging.warning(f"Failed to add festival data for {city} in {year}: {e}")
                
        return df

    def time_handling(self):
        """Handle time column in the datasets."""
        # Ensure time column exists
        if 'time' not in self.train_data.columns or 'time' not in self.test_data.columns:
            raise ValueError("Time column 'time' is missing from the datasets")

        # Convert time column to datetime if it's not already
        for df in [self.train_data, self.test_data]:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                # Use utc=True to handle mixed timezones, then keep the timezone info
                df['time'] = pd.to_datetime(df['time'], utc=True)
                
            df['year'] = df['time'].dt.year
            df['month'] = df['time'].dt.month
            df['day'] = df['time'].dt.day
            df['day_of_week'] = df['time'].dt.dayofweek
            df['hour'] = df['time'].dt.hour
    
    def add_holiday_features(self):
        """Add national holiday features to both datasets."""
        for df in [self.train_data, self.test_data]:
            if 'time' not in df.columns:
                continue
                
            # Ensure time column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'], utc=True)
            
            # Initialize holiday columns
            df['is_national_holiday'] = 0
            df['holiday_name'] = 'none'
            
            # Add holiday data for each year
            for year in range(2015, 2019):
                try:
                    holidays = self.config_loader.get_holidays_config(year)
                    year_mask = df['time'].dt.year == year
                    
                    for date_str, holiday_name in holidays.items():
                        holiday_date = pd.to_datetime(date_str).date()
                        date_mask = df['time'].dt.date == holiday_date
                        combined_mask = year_mask & date_mask
                        
                        df.loc[combined_mask, 'is_national_holiday'] = 1
                        
                except Exception as e:
                    logging.warning(f"Failed to add holiday data for {year}: {e}")
        
        # categorical encoding for holiday names
        le = LabelEncoder()
        le.fit(['none'] + list(self.config_loader.get_holidays_config(2015).values()))
        for df in [self.train_data, self.test_data]:
            df['holiday_name'] = le.transform(df['holiday_name'])

    def add_population_features(self):
        """Add population-weighted features to both datasets."""
        for df in [self.train_data, self.test_data]:
            # Get population totals
            total_pop = sum(self.population_weights.values())
            
            # Add normalized population weights for each city
            for city, population in self.population_weights.items():
                city_lower = city.lower()
                if city_lower == 'sevilla':
                    city_lower = 'seville'
                    
                # Normalized population weight
                df[f'{city_lower}_population_weight'] = population / total_pop
                df[f'{city_lower}_population_raw'] = population
                
            # Add population type info
            df['total_population'] = total_pop

    def add_basic_interactions(self):
        """Add basic interaction features between holidays, festivals, and time."""
        interactions_config = self.feature_config.get('interactions', {})
        holiday_amplification = interactions_config.get('holiday_amplification_factor', 1.5)
        rain_threshold = interactions_config.get('rain_threshold_mm', 5.0)
        indoor_displacement = interactions_config.get('indoor_displacement_rate', 0.3)
        
        for df in [self.train_data, self.test_data]:
            if 'time' not in df.columns:
                continue
                
            # Weekend/weekday flags
            if 'day_of_week' in df.columns:
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['is_weekday'] = (df['day_of_week'] < 5).astype(int)
            
            # Holiday-Festival amplification
            if 'is_national_holiday' in df.columns:
                for city in ['madrid', 'barcelona', 'valencia', 'seville', 'bilbao']:
                    festival_col = f'{city}_has_festival'
                    if festival_col in df.columns:
                        # Holiday amplification effect
                        amplified_col = f'{city}_festival_holiday_amplified'
                        df[amplified_col] = df[festival_col] * df['is_national_holiday'] * holiday_amplification
                        
                        # Population weighted effect
                        if f'{city}_population_weight' in df.columns:
                            pop_weighted_col = f'{city}_festival_pop_weighted'
                            df[pop_weighted_col] = df[festival_col] * df[f'{city}_population_weight']
            
            # Weather-Festival interactions (if weather data is available)
            for city in ['madrid', 'barcelona', 'valencia', 'seville', 'bilbao']:
                rain_col = f'{city}_rain'
                festival_col = f'{city}_has_festival'
                outdoor_impact_col = f'{city}_festival_outdoor_impact'
                
                if rain_col in df.columns and festival_col in df.columns and outdoor_impact_col in df.columns:
                    # Festival rain reduction effect
                    rain_reduction_col = f'{city}_festival_rain_reduction'
                    df[rain_reduction_col] = (
                        df[outdoor_impact_col] * 
                        np.clip(df[rain_col] / rain_threshold, 0, 1)
                    )
                    
                    # Indoor displacement effect
                    indoor_displacement_col = f'{city}_festival_indoor_displacement'
                    df[indoor_displacement_col] = df[rain_reduction_col] * indoor_displacement
            
            # Time-of-day festival effects
            if 'hour' in df.columns:
                # Evening festival effect (19-23時)
                df['is_evening'] = ((df['hour'] >= 19) & (df['hour'] <= 23)).astype(int)
                
                for city in ['madrid', 'barcelona', 'valencia', 'seville', 'bilbao']:
                    festival_col = f'{city}_has_festival'
                    if festival_col in df.columns:
                        evening_effect_col = f'{city}_festival_evening_effect'
                        df[evening_effect_col] = df[festival_col] * df['is_evening']

    def add_supply_demand_features(self):
        """Add supply-demand balance features to both datasets."""
        generation_columns = [
            'generation_biomass', 'generation_fossil_brown_coal/lignite', 
            'generation_fossil_gas', 'generation_fossil_hard_coal', 
            'generation_fossil_oil', 'generation_hydro_pumped_storage_consumption',
            'generation_hydro_run_of_river_and_poundage', 'generation_hydro_water_reservoir',
            'generation_nuclear', 'generation_other', 'generation_other_renewable',
            'generation_solar', 'generation_waste', 'generation_wind_onshore'
        ]
        
        for df in [self.train_data, self.test_data]:
            # Check which generation columns exist in the dataset
            available_gen_cols = [col for col in generation_columns if col in df.columns]
            
            if not available_gen_cols or 'total_load_actual' not in df.columns:
                logging.warning("Missing generation or load data for supply-demand features")
                continue
            
            # 1. Basic supply-demand indicators
            # Total supply (sum of all generation)
            df['total_supply'] = df[available_gen_cols].sum(axis=1)
            
            # Supply-demand balance ratio: (supply - demand) / demand
            df['supply_demand_balance_ratio'] = (df['total_supply'] - df['total_load_actual']) / df['total_load_actual']
            
            # Supply sufficiency ratio: supply / demand
            df['supply_sufficiency_ratio'] = df['total_supply'] / df['total_load_actual']
            
            # Supply surplus: supply - demand
            df['supply_surplus'] = df['total_supply'] - df['total_load_actual']
            
            # 2. Power source composition balance
            # Define power source categories
            baseload_cols = ['generation_nuclear', 'generation_fossil_hard_coal']
            renewable_cols = ['generation_solar', 'generation_wind_onshore', 'generation_hydro_run_of_river_and_poundage', 
                            'generation_hydro_water_reservoir', 'generation_other_renewable']
            flexible_cols = ['generation_fossil_gas', 'generation_fossil_oil']
            fossil_cols = ['generation_fossil_brown_coal/lignite', 'generation_fossil_hard_coal', 
                          'generation_fossil_gas', 'generation_fossil_oil']
            
            # Calculate ratios for each category
            for category_name, category_cols in [
                ('baseload', baseload_cols),
                ('renewable', renewable_cols), 
                ('flexible', flexible_cols),
                ('fossil', fossil_cols)
            ]:
                available_category_cols = [col for col in category_cols if col in df.columns]
                if available_category_cols:
                    category_sum = df[available_category_cols].sum(axis=1)
                    # Avoid division by zero
                    df[f'{category_name}_ratio'] = np.where(
                        df['total_supply'] > 0,
                        category_sum / df['total_supply'],
                        0
                    )
                else:
                    df[f'{category_name}_ratio'] = 0
    
    def wind_speed_processing(self):
        """Process wind speed columns in the datasets."""
        # Identify wind speed columns
        wind_speed_cols = [col for col in self.train_data.columns if 'wind_speed' in col]
        
        # Clip wind speed values to a reasonable range (0 to 19 m/s)
        self.train_data[wind_speed_cols] = self.train_data[wind_speed_cols].clip(lower=0, upper=19)
        self.test_data[wind_speed_cols] = self.test_data[wind_speed_cols].clip(lower=0, upper=19)

    def process_weather_icon_features(self):
        """Process weather icon features based on weather.md implementation plan."""
        cities = ["bilbao", "barcelona", "seville", "madrid", "valencia"]
        
        for df in [self.train_data, self.test_data]:
            new_columns = {}
            
            for city in cities:
                icon_col = f"{city}_weather_icon"
                if icon_col not in df.columns:
                    continue
                    
                # Extract icon features for each row
                icon_numbers = []
                day_nights = []
                has_time_infos = []
                is_clears = []
                has_clouds_list = []
                has_precipitations = []
                is_extremes = []
                
                for icon_code in df[icon_col]:
                    icon_number, day_night, has_time_info = self._extract_icon_features_robust(icon_code)
                    is_clear, has_clouds, has_precipitation, is_extreme = self._categorize_icon_number(icon_number)
                    
                    icon_numbers.append(icon_number if icon_number is not None else 0)
                    day_nights.append(day_night if day_night is not None else 0)
                    has_time_infos.append(has_time_info)
                    is_clears.append(is_clear)
                    has_clouds_list.append(has_clouds)
                    has_precipitations.append(has_precipitation)
                    is_extremes.append(is_extreme)
                
                # Add all icon-derived features
                new_columns[f"{city}_icon_number"] = icon_numbers
                new_columns[f"{city}_is_day"] = day_nights
                new_columns[f"{city}_has_time_info"] = has_time_infos
                new_columns[f"{city}_is_clear"] = is_clears
                new_columns[f"{city}_has_clouds"] = has_clouds_list
                new_columns[f"{city}_has_precipitation"] = has_precipitations
                new_columns[f"{city}_is_extreme"] = is_extremes
            
            # Add all new columns at once if any were calculated
            if new_columns:
                new_df = pd.DataFrame(new_columns, index=df.index)
                # Update the original dataframe reference
                if df is self.train_data:
                    self.train_data = pd.concat([df, new_df], axis=1)
                else:
                    self.test_data = pd.concat([df, new_df], axis=1)
    
    def _extract_icon_features_robust(self, icon_code):
        """Robust icon feature extraction based on weather.md plan."""
        if pd.isna(icon_code):
            return None, None, None
        
        icon_str = str(icon_code).strip()
        
        # Extract number part (first 1-2 digits)
        number_match = re.match(r'(\d{1,2})', icon_str)
        if number_match:
            icon_number = int(number_match.group(1))
        else:
            icon_number = None
        
        # d/n flag (check suffix)
        if icon_str.endswith('d'):
            day_night = 1  # day
            has_time_info = 1
        elif icon_str.endswith('n'):
            day_night = 0  # night  
            has_time_info = 1
        else:
            day_night = None  # unknown
            has_time_info = 0
        
        return icon_number, day_night, has_time_info
    
    def _categorize_icon_number(self, icon_number):
        """Meteorological classification of icon numbers based on OpenWeatherMap standard."""
        if pd.isna(icon_number) or icon_number is None:
            return 0, 0, 0, 0
        
        # OpenWeatherMap standard classification
        clear_sky = 1 if icon_number == 1 else 0
        clouds = 1 if icon_number in [2, 3, 4] else 0
        precipitation = 1 if icon_number in [9, 10, 13] else 0
        extreme = 1 if icon_number in [11, 50] else 0  # thunderstorm, mist/fog
        
        return clear_sky, clouds, precipitation, extreme
    
    def process_weather_main_features(self):
        """Process weather main features using config/weather.yml mapping."""
        cities = ["bilbao", "barcelona", "seville", "madrid", "valencia"]
        
        # Get weather main config
        weather_main_config = self.weather_config.get('weather_main', {})
        
        for df in [self.train_data, self.test_data]:
            new_columns = {}
            
            for city in cities:
                main_col = f"{city}_weather_main"
                if main_col not in df.columns:
                    continue
                
                # Extract main weather features for each row
                main_clouds_values = []
                
                for main_weather in df[main_col]:
                    cloud_ratio = self._extract_main_weather_features(main_weather, weather_main_config)
                    main_clouds_values.append(cloud_ratio)
                
                # Add main weather derived features
                new_columns[f"{city}_main_clouds"] = main_clouds_values
                
                # Add categorical features for each main weather type
                for weather_type in weather_main_config.keys():
                    weather_type_values = []
                    for main_weather in df[main_col]:
                        is_type = 1 if str(main_weather).lower() == weather_type else 0
                        weather_type_values.append(is_type)
                    new_columns[f"{city}_main_{weather_type}"] = weather_type_values
            
            # Add all new columns at once if any were calculated
            if new_columns:
                new_df = pd.DataFrame(new_columns, index=df.index)
                # Update the original dataframe reference
                if df is self.train_data:
                    self.train_data = pd.concat([df, new_df], axis=1)
                else:
                    self.test_data = pd.concat([df, new_df], axis=1)
    
    def _extract_main_weather_features(self, main_weather, weather_main_config):
        """Extract features from main weather using config mapping."""
        if pd.isna(main_weather):
            return 0.0
        
        main_weather_str = str(main_weather).lower().strip()
        
        # Get cloud ratio from config
        weather_info = weather_main_config.get(main_weather_str, {})
        cloud_ratio = weather_info.get('clouds', 0.0)
        
        return cloud_ratio
    
    def process_weather_description_features(self):
        """Process weather description features using config/weather.yml and intensity analysis."""
        cities = ["bilbao", "barcelona", "seville", "madrid", "valencia"]
        
        # Get weather description config
        weather_desc_config = self.weather_config.get('weather_description', {})
        
        for df in [self.train_data, self.test_data]:
            new_columns = {}
            
            for city in cities:
                desc_col = f"{city}_weather_description"
                if desc_col not in df.columns:
                    continue
                
                # Extract description features for each row
                desc_clouds_values = []
                precipitation_intensities = []
                has_showers = []
                has_proximities = []
                is_raggeds = []
                
                for description in df[desc_col]:
                    # Get cloud ratio from config
                    cloud_ratio = self._extract_description_cloud_ratio(description, weather_desc_config)
                    desc_clouds_values.append(cloud_ratio)
                    
                    # Extract intensity and special pattern features
                    intensity_features = self._extract_intensity_features(description)
                    precipitation_intensities.append(intensity_features['precipitation_intensity'])
                    has_showers.append(intensity_features['has_shower'])
                    has_proximities.append(intensity_features['has_proximity'])
                    is_raggeds.append(intensity_features['is_ragged'])
                
                # Add description derived features
                new_columns[f"{city}_desc_clouds"] = desc_clouds_values
                new_columns[f"{city}_precipitation_intensity"] = precipitation_intensities
                new_columns[f"{city}_has_shower"] = has_showers
                new_columns[f"{city}_has_proximity"] = has_proximities
                new_columns[f"{city}_is_ragged"] = is_raggeds
            
            # Add all new columns at once if any were calculated
            if new_columns:
                new_df = pd.DataFrame(new_columns, index=df.index)
                # Update the original dataframe reference
                if df is self.train_data:
                    self.train_data = pd.concat([df, new_df], axis=1)
                else:
                    self.test_data = pd.concat([df, new_df], axis=1)
    
    def _extract_description_cloud_ratio(self, description, weather_desc_config):
        """Extract cloud ratio from description using config mapping."""
        if pd.isna(description):
            return 0.0
        
        desc_str = str(description).lower().strip().replace(' ', '_')
        
        # Try exact match first
        if desc_str in weather_desc_config:
            return weather_desc_config[desc_str].get('clouds', 0.0)
        
        # Try partial matches for common patterns
        for config_key, config_value in weather_desc_config.items():
            if config_key in desc_str or desc_str in config_key:
                return config_value.get('clouds', 0.0)
        
        # Default fallback based on common keywords
        if any(keyword in desc_str for keyword in ['clear', 'sunny']):
            return 0.0
        elif any(keyword in desc_str for keyword in ['few_clouds', 'partly']):
            return 0.2
        elif any(keyword in desc_str for keyword in ['scattered', 'partly_cloudy']):
            return 0.4
        elif any(keyword in desc_str for keyword in ['broken', 'mostly_cloudy']):
            return 0.6
        elif any(keyword in desc_str for keyword in ['overcast', 'cloudy']):
            return 0.8
        elif any(keyword in desc_str for keyword in ['rain', 'snow', 'storm']):
            return 1.0
        
        return 0.5  # Default neutral value
    
    def _extract_intensity_features(self, description):
        """Extract precipitation intensity and special patterns from description."""
        if pd.isna(description):
            return {
                'precipitation_intensity': 0,
                'has_shower': 0,
                'has_proximity': 0,
                'is_ragged': 0
            }
        
        desc_str = str(description).lower().strip()
        
        # Precipitation intensity (0-4 scale) based on weather.md plan
        intensity = 0
        if 'very heavy' in desc_str:
            intensity = 4
        elif 'heavy intensity' in desc_str or 'heavy' in desc_str:
            intensity = 3
        elif 'moderate' in desc_str:
            intensity = 2
        elif 'light intensity' in desc_str or 'light' in desc_str:
            intensity = 1
        
        # Special patterns
        has_shower = 1 if 'shower' in desc_str else 0
        has_proximity = 1 if 'proximity' in desc_str else 0
        is_ragged = 1 if 'ragged' in desc_str else 0
        
        return {
            'precipitation_intensity': intensity,
            'has_shower': has_shower,
            'has_proximity': has_proximity,
            'is_ragged': is_ragged
        }

    def weather_col_processing(self):
        """Process weather-related cols in the datasets."""

        weather_cols = ['weather_description', 'weather_main', 'weather_icon']
        for col in weather_cols:
            if col in self.train_data.columns:
                self.train_data = self.train_data.drop(columns=col, errors='ignore')
            if col in self.test_data.columns:
                self.test_data = self.test_data.drop(columns=col, errors='ignore')

    def missing_value_handling(self):
        """Handle missing values in the datasets."""
        # Fill missing values with 0 for numeric columns
        numeric_cols = self.train_data.select_dtypes(include=['number']).columns.tolist()
        self.train_data[numeric_cols] = self.train_data[numeric_cols].fillna(0)
        self.test_data[numeric_cols] = self.test_data[numeric_cols].fillna(0)

        # For categorical columns, fill with 'unknown' or similar placeholder
        categorical_cols = self.train_data.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if col in self.train_data.columns:
                self.train_data[col] = self.train_data[col].fillna('unknown')
            if col in self.test_data.columns:
                self.test_data[col] = self.test_data[col].fillna('unknown')

        # pressure clipping
        _pressure_cols = [col for col in self.train_data.columns if 'pressure' in col]
        self.train_data[_pressure_cols] = self.train_data[_pressure_cols].clip(lower=900, upper=1100)
        self.test_data[_pressure_cols] = self.test_data[_pressure_cols].clip(lower=900, upper=1100)

    def process_all(self):
        """Process data and return train, validation, and test datasets.
        
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        general_config = self.config.get('train_config', {}).get('general', {})

        # Process weather features before dropping original columns
        self.process_weather_icon_features()    # Extract icon-based features
        self.process_weather_main_features()    # Extract main weather features  
        self.process_weather_description_features()  # Extract description features
        self.weather_col_processing() # Drop original weather columns
        
        self.time_handling()   # date, daytime, and time column handling
        
        # Add population-weighted features
        self.add_population_features()
        
        # Add holiday features
        self.add_holiday_features()
        
        # festival calendar creation
        self.create_festival_calendar()
        
        # Add basic interaction features
        self.add_basic_interactions()
        
        # Add supply-demand balance features
        self.add_supply_demand_features()

        # 欠損値の処理
        self.missing_value_handling()


        
        # Drop specified columns from both datasets
        drop_columns = general_config.get('drop_columns', [])
        train_data_clean = self.train_data.drop(columns=drop_columns, errors='ignore')
        test_data_clean = self.test_data.drop(columns=drop_columns, errors='ignore')
        
        # Extract target column before splitting
        target_col = general_config.get('target', 'price_actual')
        if target_col not in train_data_clean.columns:
            raise ValueError(f"Target column '{target_col}' not found in training data")
        
        train_target = train_data_clean[target_col].copy()
        train_features = train_data_clean.drop(columns=[target_col])
        
        # Time-based split of train data into train/val
        train_features_split, val_features_split = time_based_split(train_features, general_config)
        
        # Get corresponding target values
        train_target_split = train_target.loc[train_features_split.index]
        val_target_split = train_target.loc[val_features_split.index]
        
        # Handle categorical variables for all datasets
        categorical_cols = train_features_split.select_dtypes(include=['object']).columns.tolist()
        
        # Calculate discomfort indices
        self.cal_discomfort()
        self.cal_gene_sum()
        
        # Get numeric columns for scaling
        scale_cols = train_features_split.select_dtypes(include=['number']).columns.tolist()
        
        # Prepare dictionaries to collect all scaled columns at once
        train_scaled_dict = {}
        val_scaled_dict = {}
        test_scaled_dict = {}

        # Apply normalization using train statistics
        for col in scale_cols:
            # Calculate statistics from training data only
            mean_col = train_features_split[col].mean()
            std_col = train_features_split[col].std(ddof=0)

            # Apply normalization to all three datasets
            if std_col != 0:
                train_scaled = (train_features_split[col] - mean_col) / std_col
                val_scaled = (val_features_split[col] - mean_col) / std_col
                test_scaled = (test_data_clean[col] - mean_col) / std_col
            else:
                train_scaled = train_features_split[col] * 0
                val_scaled = val_features_split[col] * 0
                test_scaled = test_data_clean[col] * 0

            # Store all columns for each dataset
            train_scaled_dict[f"{col}_mean_to_t"] = [mean_col] * len(train_features_split)
            train_scaled_dict[f"{col}_std_to_t"] = [std_col] * len(train_features_split)
            train_scaled_dict[f"{col}_scaled"] = train_scaled
            
            val_scaled_dict[f"{col}_mean_to_t"] = [mean_col] * len(val_features_split)
            val_scaled_dict[f"{col}_std_to_t"] = [std_col] * len(val_features_split)
            val_scaled_dict[f"{col}_scaled"] = val_scaled
            
            test_scaled_dict[f"{col}_mean_to_t"] = [mean_col] * len(test_data_clean)
            test_scaled_dict[f"{col}_std_to_t"] = [std_col] * len(test_data_clean)
            test_scaled_dict[f"{col}_scaled"] = test_scaled
        
        # Create single DataFrames with all scaled columns
        train_scaled_df = pd.DataFrame(train_scaled_dict, index=train_features_split.index)
        val_scaled_df = pd.DataFrame(val_scaled_dict, index=val_features_split.index)
        test_scaled_df = pd.DataFrame(test_scaled_dict, index=test_data_clean.index)
        
        # Concatenate scaled columns (only 2 operations instead of many)
        train_processed = pd.concat([train_features_split, train_scaled_df], axis=1)
        val_processed = pd.concat([val_features_split, val_scaled_df], axis=1)
        test_processed = pd.concat([test_data_clean, test_scaled_df], axis=1)
        
        # Add target back to train and val
        self.train_df = pd.concat([train_processed, train_target_split], axis=1)
        self.val_df = pd.concat([val_processed, val_target_split], axis=1)
        self.test_df = test_processed
        
        return self.train_df, self.val_df, self.test_df

    def save_processed_data(self, output_path: Path):
        """Save the processed data to the specified output path.
        
        Args:
            output_path: Base path for saving the processed datasets
        
        Raises:
            ValueError: If data hasn't been processed yet
        """
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Data must be processed first. Call process_all() before saving.")
        
        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        train_path = output_path / 'train_data.csv'
        val_path = output_path / 'val_data.csv'
        test_path = output_path / 'test_data.csv'
        
        # Save all three datasets
        save_data(self.train_df, train_path)
        save_data(self.val_df, val_path)
        save_data(self.test_df, test_path)
        
        print(f"Saved processed data to {output_path}:")
        print(f"  - Train: {len(self.train_df)} samples -> {train_path}")
        print(f"  - Validation: {len(self.val_df)} samples -> {val_path}")
        print(f"  - Test: {len(self.test_df)} samples -> {test_path}")