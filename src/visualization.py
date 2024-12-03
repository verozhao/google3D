# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def plot_distributions(df):
    """Plot distributions of key metrics."""
    metrics = ['view_count', 'likes', 'dislikes']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[metric], kde=True)
        plt.title(f'Distribution of {metric}')
        plt.show()

def plot_category_analysis(df):
    """Plot category-related visualizations."""
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='category_name', y='view_count', data=df)
    plt.title('View Count by Category')
    plt.xticks(rotation=90)
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.countplot(y='category_name', data=df, 
                 order=df['category_name'].value_counts().index)
    plt.title('Video Categories')
    plt.show()

def plot_engagement_metrics(df):
    """Plot engagement-related metrics."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x='comments_disabled', data=df)
    plt.title('Comments Disabled (0 or 1)')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.countplot(x='ratings_disabled', data=df)
    plt.title('Ratings Disabled (0 or 1)')
    plt.show()

def generate_word_cloud(word_counts):
    """Generate and display word cloud."""
    word_dict = dict(word_counts)
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white').generate_from_frequencies(word_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def plot_growth_rates(df_sorted, video_id):
    """Plot growth rates for a specific video."""
    video_data = df_sorted[df_sorted['video_id'] == video_id]
    plt.figure(figsize=(12, 6))
    plt.plot(video_data['days_since_publication'], 
            video_data['daily_growth_rate'], 
            label='Daily Growth Rate')
    plt.plot(video_data['days_since_publication'], 
            video_data['cumulative_growth_rate'], 
            label='Cumulative Growth Rate')
    plt.title(f"Growth Rates for Video: {video_id}")
    plt.xlabel("Days Since Publication")
    plt.ylabel("Growth Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(df, numerical_cols):
    """Plot correlation matrix."""
    corr_matrix = df[numerical_cols].corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
