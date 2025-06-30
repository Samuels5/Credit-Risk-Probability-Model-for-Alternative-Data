import pandas as pd
import pytest
import sys
import os

# Add the src directory to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_processing import preprocess_data

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
