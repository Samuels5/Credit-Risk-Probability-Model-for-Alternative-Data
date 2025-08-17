"""
Unit tests for the preprocessing module.
"""
import pandas as pd
import pytest
import sys
import os
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.preprocessing import DataPreprocessor

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

def test_data_preprocessor_init():
    """Test DataPreprocessor initialization."""
    preprocessor = DataPreprocessor()
    
    assert preprocessor.numerical_transformer is None
    assert preprocessor.categorical_transformer is None
    assert preprocessor.preprocessor is None
    assert preprocessor.numerical_features is None
    assert preprocessor.categorical_features is None

def test_identify_feature_types(sample_data):
    """Test feature type identification."""
    preprocessor = DataPreprocessor()
    numerical_features, categorical_features = preprocessor._identify_feature_types(sample_data)
    
    # Check that numerical features include Amount, Value, etc.
    assert 'Amount' in numerical_features
    assert 'Value' in numerical_features
    assert 'PricingStrategy' in numerical_features
    
    # Check that categorical features include ProductCategory
    assert 'ProductCategory' in categorical_features
    
    # Check that excluded columns are not in the features
    assert 'CustomerId' not in categorical_features
    assert 'TransactionId' not in categorical_features
    assert 'FraudResult' not in numerical_features

def test_build_preprocessor(sample_data):
    """Test preprocessor pipeline building."""
    preprocessor = DataPreprocessor()
    numerical_features, categorical_features = preprocessor._identify_feature_types(sample_data)
    
    preprocessor_pipeline = preprocessor._build_preprocessor(numerical_features, categorical_features)
    
    # Check that the preprocessor has the correct structure
    assert 'num' in preprocessor_pipeline.transformers_
    assert 'cat' in preprocessor_pipeline.transformers_
    
    # Check that numerical transformer has imputer and scaler
    assert hasattr(preprocessor.numerical_transformer, 'named_steps')
    assert 'imputer' in preprocessor.numerical_transformer.named_steps
    assert 'scaler' in preprocessor.numerical_transformer.named_steps
    
    # Check that categorical transformer has imputer and onehot encoder
    assert hasattr(preprocessor.categorical_transformer, 'named_steps')
    assert 'imputer' in preprocessor.categorical_transformer.named_steps
    assert 'onehot' in preprocessor.categorical_transformer.named_steps

def test_fit_transform(sample_data):
    """Test the fit_transform method."""
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(sample_data)
    
    # Check that result is a DataFrame
    assert isinstance(processed_df, pd.DataFrame)
    
    # Check that essential columns are preserved
    assert 'CustomerId' in processed_df.columns
    assert 'TransactionStartTime' in processed_df.columns
    
    # Check that all numerical features are preserved (possibly transformed)
    for feature in ['Amount', 'Value', 'PricingStrategy']:
        assert feature in processed_df.columns
    
    # Check that categorical features are one-hot encoded
    assert any(col.startswith('ProductCategory_') for col in processed_df.columns)
    
    # Check that there are no NaN values
    assert not processed_df.isna().any().any()

def test_fit_transform_with_missing_values(sample_data):
    """Test preprocessing with missing values."""
    # Create a copy with some NaN values
    data_with_nans = sample_data.copy()
    data_with_nans.loc[0, 'Amount'] = np.nan
    data_with_nans.loc[1, 'ProductCategory'] = np.nan
    
    preprocessor = DataPreprocessor()
    processed_df = preprocessor.fit_transform(data_with_nans)
    
    # Check that there are no NaN values in the result
    assert not processed_df.isna().any().any()
