"""Load saved model and provide prediction helper for single samples."""
import os
import joblib
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, '..', 'models', 'freshness_model.pkl')


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError('Model not found. Run model_training.py first.')
    return joblib.load(MODEL_PATH)


def predict_sample(model, sample: dict):
    # sample should be a dict with the same features used in training (texture removed)
    df = pd.DataFrame([sample])
    pred = model.predict(df)[0]
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(df)[0].tolist()
    return pred, proba


if __name__ == '__main__':
    print('This module is intended to be imported by the frontend or other scripts.')
