data=df

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['publishedAt'] = pd.to_datetime(X_['publishedAt'])
        X_['trending_date'] = pd.to_datetime(X_['trending_date'])

        X_['days_since_published'] = (X_['trending_date'] - X_['publishedAt']).dt.total_seconds() / 86400
        X_['title_length'] = X_['title'].str.len()
        X_['tags_count'] = X_['tags'].str.count('\|') + 1
        X_['description_length'] = X_['description'].str.len()

        X_['publishedAt_dayofweek'] = X_['publishedAt'].dt.dayofweek
        X_['publishedAt_hour'] = X_['publishedAt'].dt.hour
        X_['publishedAt_year'] = X_['publishedAt'].dt.year
        X_['publishedAt_month'] = X_['publishedAt'].dt.month

        X_['trendingDate_year'] = X_['trending_date'].dt.year
        X_['trendingDate_month'] = X_['trending_date'].dt.month

        X_['days_tags_interaction'] = X_['days_since_published'] * X_['tags_count']

        X_['likes_growth_rate'] = X_['likes'] / (X_['days_since_published'] + 1e-6)
        X_['comment_count_growth_rate'] = X_['comment_count'] / (X_['days_since_published'] + 1e-6)

        return X_


features = ['publishedAt', 'trending_date', 'title', 'tags', 'description', 'categoryId',
           'comment_count', 'likes']
target = 'view_count'

plt.figure(figsize=(10, 6))
sns.histplot(data[target], kde=True)
plt.title('Distribution of View Count')
plt.xlabel('View Count')
plt.show()

data['log_view_count'] = np.log1p(data[target])

X = data[features]
y = data['log_view_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['days_since_published', 'comment_count', 'likes', 'title_length', 'tags_count',
                   'description_length', 'publishedAt_dayofweek', 'publishedAt_hour', 'categoryId',
                   'days_tags_interaction', 'likes_growth_rate', 'comment_count_growth_rate',
                   'publishedAt_year', 'publishedAt_month', 'trendingDate_year', 'trendingDate_month']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Huber Regression': HuberRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_regression, k='all')),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return pipeline, mse, r2

results = {}

for name, model in models.items():
    pipeline, mse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'mse': mse, 'r2': r2, 'pipeline': pipeline}
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
    print("="*50)

best_model = max(results, key=lambda x: results[x]['r2'])
print(f"\nBest model: {best_model}")
print(f"MSE: {results[best_model]['mse']:.4f}")
print(f"R2: {results[best_model]['r2']:.4f}")

best_pipeline = results[best_model]['pipeline']


# GridSearchCV
param_grids = {
    'Linear Regression': {},
    'Huber Regression': {
        'regressor__epsilon': [1.1, 1.35, 1.5],
        'regressor__alpha': [0.0001, 0.001, 0.01]
    },
    'Random Forest': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.3]
    }
}

# GridSearchCV on best model
param_grid = param_grids[best_model]

if param_grid:
    grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print("\nBest parameters:", grid_search.best_params_)
    print("Best MSE:", -grid_search.best_score_)

    y_pred = grid_search.predict(X_test)
else:
    y_pred = best_pipeline.predict(X_test)

y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)

final_mse = mean_squared_error(y_test_original, y_pred_original)
final_r2 = r2_score(y_test_original, y_pred_original)

print("\nFinal Model Performance (on original scale):")
print(f"MSE: {final_mse:.2f}")
print(f"R2: {final_r2:.4f}")

if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    importances = best_pipeline.named_steps['regressor'].feature_importances_
    feature_names = best_pipeline.named_steps['feature_selection'].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
elif hasattr(best_pipeline.named_steps['regressor'], 'coef_'):
    coefficients = best_pipeline.named_steps['regressor'].coef_
    feature_names = best_pipeline.named_steps['feature_selection'].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': np.abs(coefficients)})
    print("\nFeature Importance (based on absolute coefficients):")
    print(feature_importance.sort_values('coefficient', ascending=False))

plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual View Count')
plt.ylabel('Predicted View Count')
plt.title('Predicted vs Actual View Count')
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['publishedAt'] = pd.to_datetime(X_['publishedAt'])
        X_['trending_date'] = pd.to_datetime(X_['trending_date'])

        X_['days_since_published'] = (X_['trending_date'] - X_['publishedAt']).dt.total_seconds() / 86400
        X_['title_length'] = X_['title'].str.len()
        X_['tags_count'] = X_['tags'].str.count('\|') + 1
        X_['description_length'] = X_['description'].str.len()

        X_['publishedAt_dayofweek'] = X_['publishedAt'].dt.dayofweek
        X_['publishedAt_hour'] = X_['publishedAt'].dt.hour
        X_['publishedAt_year'] = X_['publishedAt'].dt.year
        X_['publishedAt_month'] = X_['publishedAt'].dt.month

        X_['trendingDate_year'] = X_['trending_date'].dt.year
        X_['trendingDate_month'] = X_['trending_date'].dt.month

        X_['days_tags_interaction'] = X_['days_since_published'] * X_['tags_count']

        X_['likes_growth_rate'] = X_['likes'] / (X_['days_since_published'] + 1e-6)
        X_['comment_count_growth_rate'] = X_['comment_count'] / (X_['days_since_published'] + 1e-6)

        return X_


features = ['publishedAt', 'trending_date', 'title', 'tags', 'description', 'categoryId',
           'comment_count', 'likes']
target = 'view_count'

plt.figure(figsize=(10, 6))
sns.histplot(data[target], kde=True)
plt.title('Distribution of View Count')
plt.xlabel('View Count')
plt.show()

data['log_view_count'] = np.log1p(data[target])

X = data[features]
y = data['log_view_count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['days_since_published', 'comment_count', 'likes', 'title_length', 'tags_count',
                   'description_length', 'publishedAt_dayofweek', 'publishedAt_hour', 'categoryId',
                   'days_tags_interaction', 'likes_growth_rate', 'comment_count_growth_rate',
                   'publishedAt_year', 'publishedAt_month', 'trendingDate_year', 'trendingDate_month']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Huber Regression': HuberRegressor(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_regression, k='all')),
        ('regressor', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return pipeline, mse, r2

results = {}

for name, model in models.items():
    pipeline, mse, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'mse': mse, 'r2': r2, 'pipeline': pipeline}
    print(f"{name} - MSE: {mse:.4f}, R2: {r2:.4f}")
    print("="*50)

best_model = max(results, key=lambda x: results[x]['r2'])
print(f"\nBest model: {best_model}")
print(f"MSE: {results[best_model]['mse']:.4f}")
print(f"R2: {results[best_model]['r2']:.4f}")

best_pipeline = results[best_model]['pipeline']


# GridSearchCV
param_grids = {
    'Linear Regression': {},
    'Huber Regression': {
        'regressor__epsilon': [1.1, 1.35, 1.5],
        'regressor__alpha': [0.0001, 0.001, 0.01]
    },
    'Random Forest': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [3, 5, 7],
        'regressor__learning_rate': [0.01, 0.1, 0.3]
    }
}

# GridSearchCV on best model
param_grid = param_grids[best_model]

if param_grid:
    grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    print("\nBest parameters:", grid_search.best_params_)
    print("Best MSE:", -grid_search.best_score_)

    y_pred = grid_search.predict(X_test)
else:
    y_pred = best_pipeline.predict(X_test)

y_pred_original = np.expm1(y_pred)
y_test_original = np.expm1(y_test)

final_mse = mean_squared_error(y_test_original, y_pred_original)
final_r2 = r2_score(y_test_original, y_pred_original)

print("\nFinal Model Performance (on original scale):")
print(f"MSE: {final_mse:.2f}")
print(f"R2: {final_r2:.4f}")

if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):
    importances = best_pipeline.named_steps['regressor'].feature_importances_
    feature_names = best_pipeline.named_steps['feature_selection'].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
elif hasattr(best_pipeline.named_steps['regressor'], 'coef_'):
    coefficients = best_pipeline.named_steps['regressor'].coef_
    feature_names = best_pipeline.named_steps['feature_selection'].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': np.abs(coefficients)})
    print("\nFeature Importance (based on absolute coefficients):")
    print(feature_importance.sort_values('coefficient', ascending=False))

plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_original, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.xlabel('Actual View Count')
plt.ylabel('Predicted View Count')
plt.title('Predicted vs Actual View Count')
plt.show()
