import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, LSTM, Bidirectional,
    Dropout, BatchNormalization, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow.keras import backend as K
from IPython.display import SVG

FEATURES = [
    'title', 'publishedAt', 'categoryId', 'trending_date', 'tags',
    'likes', 'comment_count', 'description',
    'publishedAt_year', 'publishedAt_month', 'publishedAt_date', 'publishedAt_time',
    'trendingDate_year', 'trendingDate_month', 'trendingDate_date'
]

class AdvancedFeatureEngineer:
    def __init__(self):
        self.le = LabelEncoder()

    def safe_numeric_convert(self, series):
        """Safely convert series to numeric values"""
        if isinstance(series, (pd.Series, np.ndarray)):
            return pd.to_numeric(series, errors='coerce').fillna(0)
        return series

    def extract_hour(self, time_val):
        """Safely extract hour from various time formats"""
        if pd.isna(time_val):
            return -1
        try:
            if isinstance(time_val, str):
                return pd.to_datetime(time_val).hour
            elif isinstance(time_val, (datetime.time, pd.Timestamp)):
                return time_val.hour
            return -1
        except:
            return -1

    def fit_transform(self, X):
        X_ = X.copy()

        X_['publishedAt_dt'] = pd.to_datetime(X_['publishedAt'])
        X_['trending_dt'] = pd.to_datetime(X_['trending_date'])

        X_['days_between'] = (X_['trending_dt'] - X_['publishedAt_dt']).dt.total_seconds() / 86400
        X_['publish_hour'] = X_['publishedAt_time'].apply(self.extract_hour)

        X_['title_length'] = X_['title'].astype(str).str.len()
        X_['tags_count'] = X_['tags'].astype(str).str.count('\|') + 1
        X_['description_length'] = X_['description'].astype(str).str.len()

        X_['title_word_count'] = X_['title'].astype(str).str.split().str.len()
        X_['description_word_count'] = X_['description'].astype(str).str.split().str.len()

        numeric_cols = ['likes', 'comment_count']
        for col in numeric_cols:
            X_[col] = self.safe_numeric_convert(X_[col])

        X_['likes_per_day'] = X_['likes'] / (X_['days_between'] + 1)
        X_['comments_per_day'] = X_['comment_count'] / (X_['days_between'] + 1)
        X_['engagement_ratio'] = (X_['likes'] + X_['comment_count']) / (X_['days_between'] + 1)
        X_['engagement_per_word'] = X_['engagement_ratio'] / (X_['title_word_count'] + 1)
        X_['likes_per_tag'] = X_['likes'] / (X_['tags_count'] + 1)
        X_['comments_per_tag'] = X_['comment_count'] / (X_['tags_count'] + 1)

        X_['categoryId'] = self.le.fit_transform(X_['categoryId'].astype(str))

        X_['publish_is_weekend'] = X_['publishedAt_dt'].dt.dayofweek.isin([5, 6]).astype(int)
        X_['trending_is_weekend'] = pd.to_datetime(X_['trending_date']).dt.dayofweek.isin([5, 6]).astype(int)

        final_features = [
            'categoryId', 'likes', 'comment_count',
            'publishedAt_year', 'publishedAt_month',
            'trendingDate_year', 'trendingDate_month',
            'days_between', 'title_length', 'tags_count',
            'description_length', 'likes_per_day', 'comments_per_day',
            'engagement_ratio', 'engagement_per_word', 'likes_per_tag',
            'comments_per_tag', 'publish_hour', 'publish_is_weekend',
            'trending_is_weekend', 'title_word_count', 'description_word_count'
        ]

        for col in final_features:
            if col not in X_.columns:
                X_[col] = 0
            else:
                X_[col] = self.safe_numeric_convert(X_[col])

        return X_

    def transform(self, X):
        return self.fit_transform(X)

def prepare_text_data(texts, tokenizer=None, max_words=10000, max_len=50, fit=False):
    processed_texts = [str(text) if pd.notnull(text) else '' for text in texts]

    if fit:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(processed_texts)

    sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_len)

    return (padded_sequences, tokenizer) if fit else padded_sequences

def create_tunable_model(hp, num_features):
    num_input = Input(shape=(num_features,))

    x_num = Dense(
        units=hp.Int('num_dense_units', min_value=128, max_value=512, step=64),
        activation='relu'
    )(num_input)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(hp.Float('num_dropout', min_value=0.1, max_value=0.5, step=0.1))(x_num)

    text_input = Input(shape=(50,))

    embedding_dim = hp.Int('embedding_dim', min_value=64, max_value=256, step=32)
    x_text = Embedding(10000, embedding_dim)(text_input)

    lstm_units = hp.Int('lstm_units', min_value=32, max_value=128, step=32)
    x_text = Bidirectional(LSTM(lstm_units))(x_text)
    x_text = Dropout(hp.Float('text_dropout', min_value=0.1, max_value=0.5, step=0.1))(x_text)

    x = Concatenate()([x_num, x_text])

    for i in range(hp.Int('num_dense_layers', 1, 3)):
        x = Dense(
            units=hp.Int(f'dense_{i}_units', min_value=64, max_value=256, step=32),
            activation='relu'
        )(x)
        x = BatchNormalization()(x)
        x = Dropout(hp.Float(f'dense_{i}_dropout', min_value=0.1, max_value=0.4, step=0.1))(x)

    output = Dense(1)(x)

    model = Model(inputs=[num_input, text_input], outputs=output)

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse'
    )

    return model

def train_model_with_tuning(X_train, X_test, y_train, y_test):
    print("Initial shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    fe = AdvancedFeatureEngineer()
    X_train_fe = fe.fit_transform(X_train)
    X_test_fe = fe.transform(X_test)

    numeric_features = [col for col in X_train_fe.columns if col not in
                       ['title', 'tags', 'description', 'publishedAt', 'trending_date',
                        'publishedAt_dt', 'trending_dt', 'publishedAt_time']]

    print(f"Number of numeric features: {len(numeric_features)}")

    train_texts = [f"{title} {tags}" for title, tags
                  in zip(X_train_fe['title'].fillna(''), X_train_fe['tags'].fillna(''))]
    test_texts = [f"{title} {tags}" for title, tags
                 in zip(X_test_fe['title'].fillna(''), X_test_fe['tags'].fillna(''))]

    X_train_text, tokenizer = prepare_text_data(train_texts, fit=True)
    X_test_text = prepare_text_data(test_texts, tokenizer=tokenizer)

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_fe[numeric_features])
    X_test_num = scaler.transform(X_test_fe[numeric_features])

    print("\nProcessed shapes:")
    print(f"X_train_num shape: {X_train_num.shape}")
    print(f"X_train_text shape: {X_train_text.shape}")

    def build_model(hp):
        return create_tunable_model(hp, num_features=X_train_num.shape[1])

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=20,
        factor=3,
        directory='hyperband_tuning',
        project_name='youtube_views_prediction',
        overwrite=True
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    tuner.search(
        [X_train_num, X_train_text],
        y_train,
        validation_split=0.2,
        callbacks=[early_stopping],
        batch_size=256
    )

    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("\nBest Hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        [X_train_num, X_train_text],
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        callbacks=[
            early_stopping,
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
        ],
        verbose=1
    )

    y_pred = model.predict([X_test_num, X_test_text], batch_size=256)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, history, mse, r2, y_pred, best_hps

def evaluate_and_visualize(y_test, y_pred, y_test_original, y_pred_original, history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(y_test, y_pred, alpha=0.5, s=20)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual View Count (log scale)')
    plt.ylabel('Predicted View Count (log scale)')
    plt.title(f'Log Scale Predictions\nR² = {r2_score(y_test, y_pred):.4f}')

    plt.subplot(1, 3, 3)
    y_test_m = y_test_original / 1e6
    y_pred_m = y_pred_original / 1e6
    plt.scatter(y_test_m, y_pred_m, alpha=0.5, s=20)
    plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], 'r--', lw=2)
    plt.xlabel('Actual Views (millions)')
    plt.ylabel('Predicted Views (millions)')
    plt.title('Original Scale Predictions')

    plt.tight_layout()
    plt.show()

    print("\nPerformance Metrics:")
    print(f"Log Scale R²: {r2_score(y_test, y_pred):.4f}")

    ranges = [
        (0, 1e6, '<1M'),
        (1e6, 10e6, '1M-10M'),
        (10e6, 100e6, '10M-100M'),
        (100e6, float('inf'), '>100M')
    ]

if __name__ == "__main__":
    X = data[FEATURES]
    y = np.log1p(data['view_count'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, history, mse, r2, y_pred, best_hps = train_model_with_tuning(X_train, X_test, y_train, y_test)

    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)

    evaluate_and_visualize(y_test, y_pred, y_test_original, y_pred_original, history)

    model.summary()

    SVG(tf.keras.utils.model_to_dot(model, dpi=70).create(prog='dot', format='svg'))

    
    print("\nFinal Training Metrics:")
    print(f"Final training loss: {history.history['loss'][-1]:.4f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Number of epochs completed: {len(history.history['loss'])}")

    print("\nTest Set Performance:")
    print(f"Mean Squared Error (log scale): {mse:.4f}")
    print(f"R² Score (log scale): {r2:.4f}")

    print("\nBest Hyperparameters:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")

    mse_log = mean_squared_error(y_test, y_pred)
    print(f"\nMSE (log scale): {mse_log:.4f}")

    y_test_original = np.expm1(y_test)
    y_pred_original = np.expm1(y_pred)
    mse_original = mean_squared_error(y_test_original, y_pred_original)
    print(f"MSE (original scale): {mse_original:.2f}")

    rmse_log = np.sqrt(mse_log)
    rmse_original = np.sqrt(mse_original)
    print(f"RMSE (log scale): {rmse_log:.4f}")
    print(f"RMSE (original scale): {rmse_original:.2f}")

    print("Shapes:")
    print(f"y_test shape: {y_test.shape}")
    print(f"y_pred shape: {y_pred.shape}")

    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred).flatten()

    mae_log = np.mean(np.abs(y_test_array - y_pred_array))
    y_test_original = np.expm1(y_test_array)
    y_pred_original = np.expm1(y_pred_array)
    mae_original = np.mean(np.abs(y_test_original - y_pred_original))

    print(f"\nMAE (log scale): {mae_log:.4f}")
    print(f"MAE (original scale): {mae_original:.2f}")

    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100
    print(f"\nMAPE: {mape:.2f}%")

    y_test_array = np.array(y_test).flatten()
    y_pred_array = np.array(y_pred).flatten()

    residuals = y_test_array - y_pred_array

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_pred_array, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values (log scale)')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    plt.subplot(1, 3, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')

    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=50, density=True, alpha=0.7)
    mu, sigma = np.mean(residuals), np.std(residuals)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2,
             label=f'Normal Dist.\nμ={mu:.2f}, σ={sigma:.2f}')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Error Distribution')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("\nResidual Statistics:")
    print(f"Mean of residuals: {np.mean(residuals):.4f}")
    print(f"Standard deviation of residuals: {np.std(residuals):.4f}")
    print(f"Skewness: {stats.skew(residuals):.4f}")
    print(f"Kurtosis: {stats.kurtosis(residuals):.4f}")

    y_test_original = np.expm1(y_test_array)
    y_pred_original = np.expm1(y_pred_array)

    mae_views = np.mean(np.abs(y_test_original - y_pred_original))

    print(f"MAE: {mae_views:,.0f} views")

    errors = np.abs(y_test_original - y_pred_original)
    print("\nError Distribution (views):")
    print(f"25th percentile: {np.percentile(errors, 25):,.0f}")
    print(f"Median error: {np.percentile(errors, 50):,.0f}")
    print(f"75th percentile: {np.percentile(errors, 75):,.0f}")
    print(f"90th percentile: {np.percentile(errors, 90):,.0f}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(y_test_original, bins=50, alpha=0.7)
    plt.title('View Count Distribution')
    plt.xlabel('Views (millions)')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(y_test_original, bins=50, alpha=0.7, log=True)
    plt.title('View Count Distribution (Log Y-axis)')
    plt.xlabel('Views (millions)')
    plt.ylabel('Frequency (log scale)')

    plt.subplot(1, 3, 3)
    plt.hist(np.log1p(y_test_original), bins=50, alpha=0.7)
    plt.title('Log-Transformed View Distribution')
    plt.xlabel('Log(Views in millions + 1)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

    print("\nView Count Statistics:")
    print(f"Mean views: {np.mean(y_test_original):,.0f}")
    print(f"Median views: {np.median(y_test_original):,.0f}")
    print(f"Std deviation: {np.std(y_test_original):,.0f}")
    print("\nPercentiles:")
    percentiles = [5, 25, 50, 75, 95, 99]
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(y_test_original, p):,.0f} views")


