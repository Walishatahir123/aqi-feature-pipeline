"""
Feature Engineering Module
Handles feature extraction, transformation, and engineering for AQI forecasting
"""
 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
 
 
class FeatureEngineer:
    """Feature engineering for air quality forecasting"""
    
    def __init__(self, target_col: str = 'aqi'):
        self.target_col = target_col
        self.feature_names = []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract temporal components
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['quarter'] = df['timestamp'].dt.quarter
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
        """Create lagged features for time series"""
        df = df.copy()
        
        for lag in lags:
            df[f'{self.target_col}_lag_{lag}'] = df[self.target_col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, 
                                windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """Create rolling window statistics"""
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{self.target_col}_rolling_mean_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{self.target_col}_rolling_std_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            df[f'{self.target_col}_rolling_min_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).min()
            )
            df[f'{self.target_col}_rolling_max_{window}'] = (
                df[self.target_col].rolling(window=window, min_periods=1).max()
            )
            
            # Rolling range
            df[f'{self.target_col}_rolling_range_{window}'] = (
                df[f'{self.target_col}_rolling_max_{window}'] - 
                df[f'{self.target_col}_rolling_min_{window}']
            )
        
        return df
    
    def create_pollutant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from pollutant measurements"""
        df = df.copy()
        
        pollutants = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3']
        
        for pollutant in pollutants:
            if pollutant in df.columns:
                # Lag features
                for lag in [1, 6, 24]:
                    df[f'{pollutant}_lag_{lag}'] = df[pollutant].shift(lag)
                
                # Rolling features
                for window in [6, 24]:
                    df[f'{pollutant}_rolling_mean_{window}'] = (
                        df[pollutant].rolling(window=window, min_periods=1).mean()
                    )
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from weather data"""
        df = df.copy()
        
        weather_vars = ['temperature', 'humidity', 'wind_speed', 'pressure']
        
        for var in weather_vars:
            if var in df.columns:
                # Interaction with time
                df[f'{var}_hour_interaction'] = df[var] * df['hour']
                
                # Polynomial features
                df[f'{var}_squared'] = df[var] ** 2
                
                # Rolling features
                df[f'{var}_rolling_mean_24'] = (
                    df[var].rolling(window=24, min_periods=1).mean()
                )
        
        # Weather combinations
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] * df['humidity'] / 100
        
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_chill'] = df['temperature'] - (df['wind_speed'] * 0.5)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        df = df.copy()
        
        # Time-AQI interactions
        if 'hour' in df.columns and f'{self.target_col}_lag_1' in df.columns:
            df['hour_aqi_interaction'] = df['hour'] * df[f'{self.target_col}_lag_1']
        
        # Weekend-AQI interaction
        if 'is_weekend' in df.columns and f'{self.target_col}_lag_24' in df.columns:
            df['weekend_aqi_interaction'] = df['is_weekend'] * df[f'{self.target_col}_lag_24']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features in proper order"""
        df = df.copy()
        
        # Temporal features
        df = self.create_temporal_features(df)
        
        # Lag features
        df = self.create_lag_features(df)
        
        # Rolling features
        df = self.create_rolling_features(df)
        
        # Pollutant features
        df = self.create_pollutant_features(df)
        
        # Weather features
        df = self.create_weather_features(df)
        
        # Interaction features
        df = self.create_interaction_features(df)
        
        # Drop rows with NaN (from lagging/rolling)
        df = df.dropna()
        
        # Store feature names (exclude timestamp and target)
        self.feature_names = [col for col in df.columns 
                            if col not in ['timestamp', self.target_col]]
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of created feature names"""
        return self.feature_names
 
 
def prepare_features_and_targets(df: pd.DataFrame, 
                                 target_col: str = 'aqi',
                                 forecast_horizon: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare features and multi-step forecast targets
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        forecast_horizon: Number of steps ahead to forecast
    
    Returns:
        X: Features DataFrame
        y: Targets DataFrame with columns for each forecast step
    """
    # Create targets for each forecast step
    targets = {}
    for h in range(1, forecast_horizon + 1):
        targets[f'{target_col}_t+{h}'] = df[target_col].shift(-h)
    
    y = pd.DataFrame(targets)
    
    # Features (exclude target)
    X = df.drop(columns=[target_col])
    
    # Drop rows with NaN in targets
    valid_idx = y.notna().all(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]
    
    return X, y