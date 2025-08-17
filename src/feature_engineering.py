"""
Feature engineering module for the Credit Risk Model.
This module contains functions for creating various features from raw transaction data.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate features for each customer based on their transaction history.
    
    Args:
        df: DataFrame containing transaction data with CustomerId and Amount columns
        
    Returns:
        DataFrame with customer-level aggregate features
    """
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std']
    }).reset_index()
    
    # Rename columns for clarity
    agg_features.columns = ['CustomerId', 'TotalTransactionAmount', 
                           'AvgTransactionAmount', 'TransactionCount', 
                           'StdDevTransactionAmount']
    
    # Fill missing values (e.g., if a customer has only one transaction, std will be NaN)
    agg_features = agg_features.fillna({
        'StdDevTransactionAmount': 0
    })
    
    return agg_features

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from transaction timestamps.
    
    Args:
        df: DataFrame containing transaction data with TransactionStartTime column
        
    Returns:
        DataFrame with time-based features
    """
    result_df = pd.DataFrame({
        'CustomerId': df['CustomerId'],
        'TransactionHour': df['TransactionStartTime'].dt.hour,
        'TransactionDay': df['TransactionStartTime'].dt.day,
        'TransactionMonth': df['TransactionStartTime'].dt.month,
        'TransactionYear': df['TransactionStartTime'].dt.year,
        'TransactionDayOfWeek': df['TransactionStartTime'].dt.dayofweek,
        'IsWeekend': df['TransactionStartTime'].dt.dayofweek.isin([5, 6]).astype(int)
    })
    
    return result_df

def calculate_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
    
    Args:
        df: DataFrame containing transaction data
        
    Returns:
        DataFrame with RFM metrics for each customer
    """
    # Set the snapshot date as one day after the most recent transaction
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    })
    
    rfm.rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Amount': 'MonetaryValue'
    }, inplace=True)
    
    return rfm

def assign_risk_labels(rfm: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> pd.DataFrame:
    """
    Cluster customers based on RFM metrics and assign risk labels.
    
    Args:
        rfm: DataFrame with RFM metrics
        n_clusters: Number of clusters to create
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with risk labels for each customer
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Standardize RFM metrics
    rfm_scaled = StandardScaler().fit_transform(rfm)
    
    # Cluster customers using K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Analyze clusters to find the highest risk one
    cluster_analysis = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).reset_index()
    
    # Highest risk = high recency (hasn't transacted recently) + 
    # low frequency (doesn't transact often) + 
    # low monetary value (doesn't spend much)
    high_risk_cluster = cluster_analysis.sort_values(
        by=['Recency', 'Frequency', 'MonetaryValue'], 
        ascending=[False, True, True]
    ).iloc[0]['Cluster']
    
    # Assign binary risk labels
    rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)
    
    return rfm[['is_high_risk', 'Cluster', 'Recency', 'Frequency', 'MonetaryValue']]
