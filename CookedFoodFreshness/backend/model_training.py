"""Train a scikit-learn model on the generated dataset and save artifacts."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CSV = os.path.join(PROJECT_ROOT, '..', 'database', 'data', 'food_data.csv')
MODEL_PATH = os.path.join(PROJECT_ROOT, '..', 'models', 'freshness_model.pkl')
REPORT_PATH = os.path.join(PROJECT_ROOT, '..', 'reports', 'metrics.txt')


def load_data(path=DATA_CSV):
    return pd.read_csv(path)


def train_and_save():
    df = load_data()
    # Use only the remaining features (ambient_temp and humidity_level removed)
    features = ['storage_time', 'time_since_cooking',
                'storage_condition', 'container_type', 'food_type', 'moisture_type',
                'cooking_method', 'smell']
    X = df[features]
    y = df['freshness_level']

    numeric_features = ['storage_time', 'time_since_cooking']
    categorical_features = ['storage_condition', 'container_type', 'food_type', 'moisture_type', 'cooking_method', 'smell']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    clf = Pipeline(steps=[('pre', preprocessor), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        f.write(report)

    print('Model saved to', MODEL_PATH)
    print('Report saved to', REPORT_PATH)


if __name__ == '__main__':
    train_and_save()
