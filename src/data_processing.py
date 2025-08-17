"""
Main data processing module for the Credit Risk Model.
This module orchestrates the entire data processing pipeline.
"""
import pandas as pd
import os
from typing import Optional

from src.preprocessing import DataPreprocessor
from src.feature_engineering import (
    create_aggregate_features,
    create_time_features,
    calculate_rfm_metrics,
    assign_risk_labels
)

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw transaction data to create features and proxy target variable.
    
    This function orchestrates the entire data processing pipeline:
    1. Preprocess the raw data (imputation, scaling, encoding)
    2. Create aggregate customer features
    3. Extract time-based features
    4. Calculate RFM metrics
    5. Assign risk labels based on clustering
    
    Args:
        df: DataFrame containing raw transaction data
        
    Returns:
        Processed DataFrame with features and target variable
    """
    # Step 1: Preprocess the raw data
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(df)
    
    # Step 2: Create aggregate features
    agg_features = create_aggregate_features(df)
    processed_df = processed_df.merge(agg_features, on='CustomerId', how='left')
    
    # Step 3: Extract time-based features
    time_features = create_time_features(df)
    processed_df = processed_df.merge(time_features, on='CustomerId', how='left')
    
    # Step 4: Calculate RFM metrics
    rfm = calculate_rfm_metrics(df)
    
    # Step 5: Assign risk labels
    risk_labels = assign_risk_labels(rfm)
    processed_df = processed_df.merge(risk_labels[['is_high_risk']], on='CustomerId', how='left')
    
    # Fill any NaNs in the final dataset
    processed_df['is_high_risk'].fillna(0, inplace=True)
    processed_df.fillna(0, inplace=True)
    
    return processed_df

def process_and_save_data(input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load data from a file, process it, and optionally save the results.
    
    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the processed data (if None, data is not saved)
        
    Returns:
        Processed DataFrame
    """
    # Load the data
    df = pd.read_csv(input_file)
    
    # Process the data
    processed_df = preprocess_data(df.copy())
    
    # Save the processed data if output_file is provided
    if output_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        processed_df.to_csv(output_file, index=False)
        print(f"Data processing complete. Processed data saved to {output_file}")
    
    return processed_df

if __name__ == '__main__':
    # Load the data
    process_and_save_data('Data/data.csv', 'Data/processed/processed_data.csv')