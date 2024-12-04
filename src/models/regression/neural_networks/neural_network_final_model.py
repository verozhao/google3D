import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

features = ['publishedAt', 'trending_date', 'title', 'tags', 'description', 'categoryId',
           'comment_count', 'likes']
target = 'view_count'

X = data[features]
y = np.log1p(data[target])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fe = FeatureEngineer()
X_train_fe = fe.fit_transform(X_train)
X_test_fe = fe.transform(X_test)

numeric_features = ['days_since_published', 'comment_count', 'likes', 'title_length', 'tags_count',
                   'description_length', 'publishedAt_dayofweek', 'publishedAt_hour', 'categoryId',
                   'days_tags_interaction', 'likes_growth_rate', 'comment_count_growth_rate',
                   'publishedAt_year', 'publishedAt_month', 'trendingDate_year', 'trendingDate_month']

categorical_features = ['categoryId']
text_features = ['title', 'tags', 'description']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ('to_string', FunctionTransformer(lambda x: x.astype(str))),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

X_train_processed = preprocessor.fit_transform(X_train_fe)
X_test_processed = preprocessor.transform(X_test_fe)

max_words = 512 # try larger max_words?
max_len = 100

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train_fe['title'] + ' ' + X_train_fe['tags'])  # description?

X_train_text = tokenizer.texts_to_sequences(X_train_fe['title'] + ' ' + X_train_fe['tags'])
X_test_text = tokenizer.texts_to_sequences(X_test_fe['title'] + ' ' + X_test_fe['tags'])

X_train_text = tf.keras.preprocessing.sequence.pad_sequences(X_train_text, maxlen=max_len)
X_test_text = tf.keras.preprocessing.sequence.pad_sequences(X_test_text, maxlen=max_len)



def create_model(input_dim, text_input_dim, embed_dim=128, num_heads=2):
    input_features = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(input_features)
    x = Dropout(0.3)(x)

    text_input = Input(shape=(max_len,))
    embedding = Embedding(input_dim=max_words, output_dim=embed_dim)(text_input)
    x_text = Flatten()(embedding)

    combined = Concatenate()([x, x_text])

    x = Dense(128, activation='relu')(combined)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    output = Dense(1)(x)

    model = Model(inputs=[input_features, text_input], outputs=output)
    return model


input_dim = X_train_processed.shape[1]
model = create_model(input_dim, max_len)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')  # reduced learning rate

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)  # increased patience
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-6)

history = model.fit(
    [X_train_processed, X_train_text], y_train,
    validation_split=0.2,
    epochs=50,  # increased epochs
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

y_pred = model.predict([X_test_processed, X_test_text])
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")


# on the original scale
y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)
final_mse = mean_squared_error(y_test_original, y_pred_original)
final_r2 = r2_score(y_test_original, y_pred_original)

print("\nFinal Model Performance (on original scale):")
print(f"MSE: {final_mse:.2f}")
print(f"R2: {final_r2:.4f}")


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual View Count')
plt.ylabel('Predicted View Count')
plt.title('Predicted vs Actual View Count')
plt.tight_layout()
plt.show()

print("\nFinal Model Performance (on log-transformed scale):")
print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")

print("\nFinal Model Performance (on original scale):")
print(f"MSE: {final_mse:.2f}")
print(f"R2: {final_r2:.4f}")
