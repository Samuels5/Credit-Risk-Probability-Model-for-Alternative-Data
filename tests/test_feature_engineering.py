"""
Unit tests for the feature engineering module.
"""
import pandas as pd
import pytest
import sys
import os
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.feature_engineering import (
    create_aggregate_features,
    create_time_features,
    calculate_rfm_metrics,
    assign_risk_labels
)

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

def test_create_aggregate_features(sample_data):
    """Test aggregate features creation."""
    agg_features = create_aggregate_features(sample_data)
    
    # Check that result is a DataFrame
    assert isinstance(agg_features, pd.DataFrame)
    
    # Check that all expected columns are present
    expected_columns = ['CustomerId', 'TotalTransactionAmount', 'AvgTransactionAmount', 
                       'TransactionCount', 'StdDevTransactionAmount']
    assert all(col in agg_features.columns for col in expected_columns)
    
    # Check number of unique customers
    assert len(agg_features) == len(sample_data['CustomerId'].unique())
    
    # Check aggregate calculations for customer C1
    c1_row = agg_features[agg_features['CustomerId'] == 'C1'].iloc[0]
    c1_data = sample_data[sample_data['CustomerId'] == 'C1']
    
    assert c1_row['TotalTransactionAmount'] == c1_data['Amount'].sum()
    assert c1_row['AvgTransactionAmount'] == c1_data['Amount'].mean()
    assert c1_row['TransactionCount'] == len(c1_data)
    
    # Check that there are no NaN values
    assert not agg_features.isna().any().any()

def test_create_time_features(sample_data):
    """Test time features extraction."""
    time_features = create_time_features(sample_data)
    
    # Check that result is a DataFrame
    assert isinstance(time_features, pd.DataFrame)
    
    # Check that all expected columns are present
    expected_columns = ['CustomerId', 'TransactionHour', 'TransactionDay', 
                       'TransactionMonth', 'TransactionYear', 'TransactionDayOfWeek']
    assert all(col in time_features.columns for col in expected_columns)
    
    # Check time feature calculations for the first row
    first_row = time_features.iloc[0]
    original_time = sample_data['TransactionStartTime'].iloc[0]
    
    assert first_row['TransactionHour'] == original_time.hour
    assert first_row['TransactionDay'] == original_time.day
    assert first_row['TransactionMonth'] == original_time.month
    assert first_row['TransactionYear'] == original_time.year
    
    # Check IsWeekend calculation
    # January 1, 2025 is a Wednesday (not weekend)
    assert time_features.iloc[0]['IsWeekend'] == 0

def test_calculate_rfm_metrics(sample_data):
    """Test RFM metrics calculation."""
    rfm = calculate_rfm_metrics(sample_data)
    
    # Check that result is a DataFrame
    assert isinstance(rfm, pd.DataFrame)
    
    # Check that all expected columns are present
    expected_columns = ['Recency', 'Frequency', 'MonetaryValue']
    assert all(col in rfm.columns for col in expected_columns)
    
    # Check number of unique customers
    assert len(rfm) == len(sample_data['CustomerId'].unique())
    
    # Check RFM calculations for customer C1
    c1_data = sample_data[sample_data['CustomerId'] == 'C1']
    c1_row = rfm.loc['C1']
    
    snapshot_date = sample_data['TransactionStartTime'].max() + pd.Timedelta(days=1)
    expected_recency = (snapshot_date - c1_data['TransactionStartTime'].max()).days
    expected_frequency = len(c1_data)
    expected_monetary = c1_data['Amount'].sum()
    
    assert c1_row['Recency'] == expected_recency
    assert c1_row['Frequency'] == expected_frequency
    assert c1_row['MonetaryValue'] == expected_monetary

def test_assign_risk_labels(sample_data):
    """Test risk label assignment."""
    rfm = calculate_rfm_metrics(sample_data)
    risk_labels = assign_risk_labels(rfm)
    
    # Check that result is a DataFrame
    assert isinstance(risk_labels, pd.DataFrame)
    
    # Check that all expected columns are present
    expected_columns = ['is_high_risk', 'Cluster', 'Recency', 'Frequency', 'MonetaryValue']
    assert all(col in risk_labels.columns for col in expected_columns)
    
    # Check that is_high_risk contains only 0 and 1
    assert risk_labels['is_high_risk'].isin([0, 1]).all()
    
    # Check that we have at least one high-risk customer
    assert risk_labels['is_high_risk'].sum() > 0
    
    # Check that Cluster values are integers and match number of clusters
    assert risk_labels['Cluster'].isin([0, 1, 2]).all()
