# src/feature_engineering.py
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter

def preprocess_text(text):
    """Preprocess text by converting to lowercase and removing special characters."""
    if pd.isna(text):
        return []
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    stop_words = set(stopwords.words('english'))
    stop_words.update(['https', 'com', 'www', 'http'])
    words = [word for word in words if word not in stop_words]
    return words

def process_text_features(df):
    """Process text features (tags and description)."""
    df['tags_words'] = df['tags'].apply(preprocess_text)
    df['description_words'] = df['description'].apply(preprocess_text)
    
    return df

def calculate_growth_rates(df):
    """Calculate daily and cumulative growth rates."""
    df_sorted = df.sort_values(['video_id', 'days_since_publication'])
    
    # Calculate growth rates
    df_sorted['daily_growth_rate'] = (df_sorted.groupby('video_id')['view_count'].pct_change() / 
                                     df_sorted.groupby('video_id')['days_since_publication'].diff())
    
    df_sorted['cumulative_growth_rate'] = ((df_sorted['view_count'] / 
                                          df_sorted.groupby('video_id')['view_count'].transform('first') - 1) / 
                                         df_sorted['days_since_publication'])
    
    # Handle infinite values
    df_sorted['daily_growth_rate'] = df_sorted['daily_growth_rate'].replace([np.inf, -np.inf], np.nan)
    df_sorted['cumulative_growth_rate'] = df_sorted['cumulative_growth_rate'].replace([np.inf, -np.inf], np.nan)
    
    return df_sorted

def get_word_counts(df):
    """Get word counts for tags and descriptions."""
    all_tags_words = [word for words_list in df['tags_words'] for word in words_list]
    all_description_words = [word for words_list in df['description_words'] for word in words_list]
    
    tags_word_count = Counter(all_tags_words)
    description_word_count = Counter(all_description_words)
    
    return tags_word_count, description_word_count
