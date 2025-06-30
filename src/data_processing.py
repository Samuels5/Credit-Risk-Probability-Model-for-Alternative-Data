import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes a raw dataframe and returns a preprocessed dataframe.
    """
    # Convert TransactionStartTime to datetime
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])

    # Define numerical and categorical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target and other unnecessary columns from feature lists
    numerical_features.remove('FraudResult')
    categorical_features.remove('TransactionId')
    categorical_features.remove('BatchId')
    categorical_features.remove('AccountId')
    categorical_features.remove('SubscriptionId')
    categorical_features.remove('CustomerId')
    categorical_features.remove('CurrencyCode')
    categorical_features.remove('CountryCode')
    categorical_features.remove('ProviderId')
    categorical_features.remove('ProductId')
    categorical_features.remove('ChannelId')


    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the data
    processed_data = pipeline.fit_transform(df)

    # Get the feature names after one-hot encoding
    ohe_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    # Combine numerical and one-hot encoded feature names
    all_feature_names = numerical_features + list(ohe_feature_names)

    # Create a new dataframe with the processed data
    processed_df = pd.DataFrame(processed_data, columns=all_feature_names, index=df.index)


    # --- Feature Engineering ---

    # 1. Aggregate Features
    agg_features = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std']
    }).reset_index()
    agg_features.columns = ['CustomerId', 'TotalTransactionAmount', 'AvgTransactionAmount', 'TransactionCount', 'StdDevTransactionAmount']
    
    # Merge aggregate features
    processed_df = processed_df.merge(agg_features, on='CustomerId', how='left')


    # 2. Extract Features from TransactionStartTime
    processed_df['TransactionHour'] = df['TransactionStartTime'].dt.hour
    processed_df['TransactionDay'] = df['TransactionStartTime'].dt.day
    processed_df['TransactionMonth'] = df['TransactionStartTime'].dt.month
    processed_df['TransactionYear'] = df['TransactionStartTime'].dt.year

    # --- Task 4: Proxy Target Variable Engineering ---

    # 1. Calculate RFM Metrics
    snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerId').agg({
        'TransactionStartTime': lambda date: (snapshot_date - date.max()).days,
        'TransactionId': 'count',
        'Amount': 'sum'
    })

    rfm.rename(columns={'TransactionStartTime': 'Recency',
                        'TransactionId': 'Frequency',
                        'Amount': 'MonetaryValue'},
               inplace=True)

    # 2. Cluster Customers
    rfm_scaled = StandardScaler().fit_transform(rfm)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # 3. Define and Assign the "High-Risk" Label
    # Analyze clusters to find the one with high recency, low frequency, and low monetary value
    cluster_analysis = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'MonetaryValue': 'mean'
    }).reset_index()

    # Assuming the cluster with the highest recency and lowest frequency/monetary is the highest risk
    high_risk_cluster = cluster_analysis.sort_values(by=['Recency', 'Frequency', 'MonetaryValue'], ascending=[False, True, True]).iloc[0]['Cluster']
    
    rfm['is_high_risk'] = rfm['Cluster'].apply(lambda x: 1 if x == high_risk_cluster else 0)

    # 4. Integrate the Target Variable
    processed_df = processed_df.merge(rfm[['is_high_risk']], on='CustomerId', how='left')

    # Fill any NaNs in the new columns with 0 (or an appropriate value)
    processed_df['is_high_risk'].fillna(0, inplace=True)
    processed_df.fillna(0, inplace=True) # General fillna for any other potential NaNs from merges

    return processed_df

if __name__ == '__main__':
    # Load the data
    df = pd.read_csv('../data/data.csv')
    
    # Preprocess the data
    processed_df = preprocess_data(df.copy())
    
    # Save the processed data
    processed_df.to_csv('../data/processed/processed_data.csv', index=False)
    
    print("Data processing complete. Processed data saved to data/processed/processed_data.csv")