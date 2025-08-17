"""
Data preprocessing module for the Credit Risk Model.
This module contains functions for cleaning and preprocessing raw transaction data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    """
    A class to preprocess transaction data for credit risk modeling.
    
    This class handles data cleaning, feature transformation, and encoding
    for the transaction data.
    """
    
    def __init__(self):
        """Initialize the DataPreprocessor."""
        self.numerical_transformer = None
        self.categorical_transformer = None
        self.preprocessor = None
        self.numerical_features = None
        self.categorical_features = None
        
    def _identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify numerical and categorical features in the dataframe.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Tuple of (numerical_features, categorical_features)
        """
        numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target and ID columns
        if 'FraudResult' in numerical_features:
            numerical_features.remove('FraudResult')
            
        features_to_remove = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 
                             'CustomerId', 'CurrencyCode', 'CountryCode', 'ProviderId', 
                             'ProductId', 'ChannelId']
        
        for feature in features_to_remove:
            if feature in categorical_features:
                categorical_features.remove(feature)
                
        return numerical_features, categorical_features
    
    def _build_preprocessor(self, numerical_features: List[str], 
                           categorical_features: List[str]) -> ColumnTransformer:
        """
        Build the preprocessing pipeline.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            
        Returns:
            ColumnTransformer for preprocessing
        """
        self.numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numerical_transformer, numerical_features),
                ('cat', self.categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the input dataframe.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            Preprocessed DataFrame
        """
        # Ensure TransactionStartTime is datetime
        if 'TransactionStartTime' in df.columns:
            df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Identify feature types
        self.numerical_features, self.categorical_features = self._identify_feature_types(df)
        
        # Build and fit preprocessor
        self.preprocessor = self._build_preprocessor(self.numerical_features, self.categorical_features)
        preprocessor_pipeline = Pipeline(steps=[('preprocessor', self.preprocessor)])
        
        # Transform data
        processed_data = preprocessor_pipeline.fit_transform(df)
        
        # Get feature names after one-hot encoding
        ohe_feature_names = preprocessor_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(self.categorical_features)
        
        # Create column names for the processed dataframe
        all_feature_names = self.numerical_features + list(ohe_feature_names)
        
        # Create processed dataframe
        processed_df = pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)
        
        # Preserve ID columns needed for later processing
        processed_df['CustomerId'] = df['CustomerId']
        if 'TransactionStartTime' in df.columns:
            processed_df['TransactionStartTime'] = df['TransactionStartTime']
        
        return processed_df
