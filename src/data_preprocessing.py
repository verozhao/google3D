# src/data_preprocessing.py
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest

def load_data(dataset_path):
    """Load and perform initial processing of the dataset."""
    df = pd.read_csv(dataset_path)
    return df

def check_missing_values(df):
    """Check and document missing values in the dataset."""
    missing_values = df.isnull().sum()
    missing_percentage = df.isnull().mean() * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    return missing_df[missing_df['Missing Values'] > 0]

def calculate_iqr_outliers(param_df):
    """Calculate outliers using IQR method."""
    outliers_mask = pd.DataFrame(False, index=param_df.index, columns=param_df.columns)
    for column in param_df.columns:
        Q1 = param_df[column].quantile(0.25)
        Q3 = param_df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_mask[column] = (param_df[column] < lower_bound) | (param_df[column] > upper_bound)
    outliers_any_column = outliers_mask.any(axis=1)
    outliers_df = param_df[outliers_any_column]
    return outliers_df

def detect_anomalies(df, numerical_cols):
    """Detect anomalies using multiple methods."""
    # Z-score method
    z_scores = np.abs(stats.zscore(df[numerical_cols]))
    z_anomalies_df = df[(z_scores > 3).any(axis=1)]
    
    # Isolation Forest method
    iso_forest = IsolationForest(contamination=0.1)
    df['if_anomaly'] = iso_forest.fit_predict(df[numerical_cols])
    if_anomalies_df = df[df['if_anomaly'] == -1]
    df = df.drop(['if_anomaly'], axis=1)
    
    return z_anomalies_df, if_anomalies_df

def clean_data(df):
    """Clean and transform the dataset."""
    # Convert datetime columns
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df['trending_date'] = pd.to_datetime(df['trending_date'])
    
    # Add datetime features
    df['publishedAt_year'] = df['publishedAt'].dt.year
    df['publishedAt_month'] = df['publishedAt'].dt.month
    df['publishedAt_date'] = df['publishedAt'].dt.day
    df['publishedAt_time'] = df['publishedAt'].dt.time

    df['trendingDate_year'] = df['trending_date'].dt.year
    df['trendingDate_month'] = df['trending_date'].dt.month
    df['trendingDate_date'] = df['trending_date'].dt.day
    
    # Handle missing values
    df.dropna(subset=['description'], inplace=True)
    df = df.reset_index(drop=True)
    
    # Drop duplicates
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    
    # Convert boolean columns
    df['comments_disabled'] = df['comments_disabled'].astype(int)
    df['ratings_disabled'] = df['ratings_disabled'].astype(int)
    
    return df

# Category mapping dictionary
category_mapping = {
    1:'Film & Animation',
    2:'Autos & Vehicles',
    10:'Music',
    15:'Music',
    17:'Sports',
    19:'Travel & Events',
    20:'Gaming',
    22:'Videoblogging',
    23:'Comedy',
    24:'Entertainment',
    25:'News & Politics',
    26:'Howto & Style',
    27:'Education',
    28:'Science & Technology',
    29:'Nonprofits & Activism'
}
