data=df

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['publishedAt'] = pd.to_datetime(X_['publishedAt'])
        X_['trending_date'] = pd.to_datetime(X_['trending_date'])

        X_['days_to_trend'] = (X_['trending_date'] - X_['publishedAt']).dt.total_seconds() / 86400
        X_['title_length'] = X_['title'].str.len()
        X_['tags_count'] = X_['tags'].str.count('\|') + 1
        X_['description_length'] = X_['description'].str.len()

        X_['publishedAt_dayofweek'] = X_['publishedAt'].dt.dayofweek
        X_['publishedAt_hour'] = X_['publishedAt'].dt.hour

        return X_

features = ['days_to_trend', 'title_length', 'tags_count', 'description_length',
            'publishedAt_dayofweek', 'publishedAt_hour', 'categoryId']

virality_threshold = data['view_count'].quantile(0.9)

data['is_viral'] = (data['view_count'] > virality_threshold).astype(int)

X = data[['publishedAt', 'trending_date', 'title', 'tags', 'description', 'categoryId']]
y = data['is_viral']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['days_to_trend', 'title_length', 'tags_count', 'description_length',
                    'publishedAt_dayofweek', 'publishedAt_hour', 'categoryId']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('feature_selection', SelectKBest(f_classif, k='all')),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

    return pipeline, accuracy, auc

results = {}

for name, model in models.items():
    pipeline, accuracy, auc = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'accuracy': accuracy, 'auc': auc, 'pipeline': pipeline}
    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    print(classification_report(y_test, pipeline.predict(X_test)))
    print("="*50)

best_model = max(results, key=lambda x: results[x]['auc'])
print(f"\nBest model: {best_model}")
print(f"Accuracy: {results[best_model]['accuracy']:.4f}")
print(f"AUC: {results[best_model]['auc']:.4f}")

best_pipeline = results[best_model]['pipeline']

param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga']
    },
    'Random Forest': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1, 0.3]
    }
}

param_grid = param_grids[best_model]

grid_search = GridSearchCV(best_pipeline, param_grid, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)

print("\nBest parameters:", grid_search.best_params_)
print("Best AUC:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
final_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test)[:, 1])

print("\nFinal Model Performance:")
print(f"Accuracy: {final_accuracy:.4f}")
print(f"AUC: {final_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

if hasattr(grid_search.best_estimator_.named_steps['classifier'], 'feature_importances_'):
    importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
    feature_names = grid_search.best_estimator_.named_steps['feature_selection'].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': importances})
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
elif hasattr(grid_search.best_estimator_.named_steps['classifier'], 'coef_'):
    coefficients = grid_search.best_estimator_.named_steps['classifier'].coef_[0]
    feature_names = grid_search.best_estimator_.named_steps['feature_selection'].get_feature_names_out()
    feature_importance = pd.DataFrame({'feature': feature_names, 'coefficient': np.abs(coefficients)})
    print("\nFeature Importance (based on absolute coefficients):")
    print(feature_importance.sort_values('coefficient', ascending=False))
