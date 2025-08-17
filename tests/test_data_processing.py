import pandas as pd
import pytest
import sys
import os
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.cluster import KMeans

# Add the src directory to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.data_processing import preprocess_data

@pytest.fixture
def sample_data():
    """Create a sample DataFrame for testing."""
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5'],
        'BatchId': ['B1', 'B1', 'B2', 'B2', 'B3'],
        'AccountId': ['A1', 'A1', 'A2', 'A2', 'A3'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2', 'S3'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C1'],
        'CurrencyCode': ['UGX', 'UGX', 'UGX', 'UGX', 'UGX'],
        'CountryCode': [256, 256, 256, 256, 256],
        'ProviderId': ['P1', 'P1', 'P2', 'P2', 'P1'],
        'ProductId': ['ProductA', 'ProductB', 'ProductA', 'ProductC', 'ProductB'],
        'ProductCategory': ['Cat1', 'Cat2', 'Cat1', 'Cat3', 'Cat2'],
        'ChannelId': ['Web', 'Android', 'iOS', 'Web', 'Android'],
        'Amount': [1000, 2000, 500, 1500, 3000],
        'Value': [1000, 2000, 500, 1500, 3000],
        'TransactionStartTime': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-05 12:00:00', '2025-01-02 08:00:00', '2025-01-06 14:00:00', '2025-01-10 18:00:00']),
        'PricingStrategy': [1, 2, 1, 3, 2],
        'FraudResult': [0, 0, 0, 0, 0]
    }
    return pd.DataFrame(data)

def test_preprocess_data_creates_is_high_risk_column(sample_data):
    """Test if the is_high_risk column is created and contains binary values."""
    processed_df = preprocess_data(sample_data)
    assert 'is_high_risk' in processed_df.columns
    assert processed_df['is_high_risk'].isin([0, 1]).all()

def test_preprocess_data_returns_dataframe(sample_data):
    """Test if preprocess_data returns a pandas DataFrame."""
    processed_df = preprocess_data(sample_data)
    assert isinstance(processed_df, pd.DataFrame)

def test_aggregated_features_created(sample_data):
    """Test if aggregate features are correctly created."""
    processed_df = preprocess_data(sample_data)
    
    # Check that all aggregate features exist
    agg_features = ['TotalTransactionAmount', 'AvgTransactionAmount', 
                   'TransactionCount', 'StdDevTransactionAmount']
    for feature in agg_features:
        assert feature in processed_df.columns
    
    # Verify calculations for specific customers
    c1_data = processed_df[processed_df['CustomerId'] == 'C1']
    c1_total = sample_data[sample_data['CustomerId'] == 'C1']['Amount'].sum()
    assert c1_data['TotalTransactionAmount'].iloc[0] == c1_total
    
    c2_data = processed_df[processed_df['CustomerId'] == 'C2']
    c2_avg = sample_data[sample_data['CustomerId'] == 'C2']['Amount'].mean()
    assert c2_data['AvgTransactionAmount'].iloc[0] == c2_avg

def test_time_based_features_created(sample_data):
    """Test if time-based features are correctly extracted."""
    processed_df = preprocess_data(sample_data)
    
    # Check that all time features exist
    time_features = ['TransactionHour', 'TransactionDay', 
                    'TransactionMonth', 'TransactionYear']
    for feature in time_features:
        assert feature in processed_df.columns
    
    # Verify calculations for a specific transaction
    first_row = processed_df.iloc[0]
    original_time = sample_data['TransactionStartTime'].iloc[0]
    
    assert first_row['TransactionHour'] == original_time.hour
    assert first_row['TransactionDay'] == original_time.day
    assert first_row['TransactionMonth'] == original_time.month
    assert first_row['TransactionYear'] == original_time.year

def test_rfm_calculation(sample_data):
    """Test if RFM metrics are correctly calculated."""
    processed_df = preprocess_data(sample_data)
    
    # Since RFM calculation is internal, we can only check that the output makes sense
    # Each customer should have an is_high_risk value
    unique_customers = sample_data['CustomerId'].unique()
    for customer in unique_customers:
        customer_rows = processed_df[processed_df['CustomerId'] == customer]
        assert not customer_rows['is_high_risk'].isna().any()
        
    # Verify that the high-risk cluster is assigned consistently
    # (Same customers should have the same risk label)
    risk_by_customer = processed_df.groupby('CustomerId')['is_high_risk'].nunique()
    assert (risk_by_customer == 1).all()

def test_missing_values_handling(sample_data):
    """Test that the preprocessing handles missing values correctly."""
    # Create a copy with some NaN values
    data_with_nans = sample_data.copy()
    data_with_nans.loc[0, 'Amount'] = np.nan
    data_with_nans.loc[1, 'ProductCategory'] = np.nan
    
    processed_df = preprocess_data(data_with_nans)
    
    # Check that no NaN values remain in the processed data
    assert not processed_df.isna().any().any()
