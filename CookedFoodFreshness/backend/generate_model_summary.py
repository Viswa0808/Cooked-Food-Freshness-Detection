"""Generate a brief model summary with top feature importances.

Usage: python generate_model_summary.py
"""
import os
import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(PROJECT_ROOT, '..', 'models', 'freshness_model.pkl')
REPORT_PATH = os.path.join(PROJECT_ROOT, '..', 'reports', 'model_summary.txt')


def get_feature_names_from_column_transformer(ct, numeric_features, categorical_features):
    """Reconstruct feature names for the pipeline's ColumnTransformer.
    Works for a transformer that uses 'passthrough' for numeric and OneHotEncoder for categoricals.
    """
    feature_names = []
    # numeric passthrough
    feature_names.extend(numeric_features)

    # categorical: find the OneHotEncoder and get categories_
    # ct is a ColumnTransformer
    for name, trans, cols in ct.transformers:
        if name == 'cat':
            # assume trans is OneHotEncoder or Pipeline with OneHotEncoder
            ohe = trans
            try:
                categories = ohe.categories_
            except Exception:
                # maybe it's a Pipeline
                try:
                    ohe = trans.named_steps['onehot']
                    categories = ohe.categories_
                except Exception:
                    categories = None
            if categories is not None:
                for col, cats in zip(cols, categories):
                    for cat in cats:
                        feature_names.append(f"{col}={cat}")
    return feature_names


def main():
    if not os.path.exists(MODEL_PATH):
        print('Model file not found at', MODEL_PATH)
        return

    pipeline = joblib.load(MODEL_PATH)

    # Assume pipeline.steps = [('pre', ColumnTransformer), ('clf', RandomForestClassifier)]
    pre = pipeline.named_steps.get('pre')
    clf = pipeline.named_steps.get('clf')

    numeric_features = ['ambient_temp', 'humidity', 'storage_time', 'time_since_cooking']
    categorical_features = ['storage_condition', 'container_type', 'food_type', 'moisture_type', 'cooking_method']

    try:
        feature_names = get_feature_names_from_column_transformer(pre, numeric_features, categorical_features)
    except Exception:
        # fallback: create simple names (numeric + categorical raw)
        feature_names = numeric_features + categorical_features

    importances = None
    try:
        importances = clf.feature_importances_
    except Exception:
        # If clf is a wrapped estimator, try to access attribute
        try:
            importances = clf.named_steps['clf'].feature_importances_
        except Exception:
            pass

    if importances is None:
        print('Could not extract feature importances from the model')
        return

    # Ensure lengths match; if not, align by truncation or padding
    if len(importances) != len(feature_names):
        # if more importances, keep starting ones; if fewer, pad names
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]

    idx = np.argsort(importances)[::-1]
    lines = []
    lines.append('Top 10 model features by importance:\n')
    top_n = min(10, len(feature_names))
    for i in range(top_n):
        fi = idx[i]
        lines.append(f"{i+1}. {feature_names[fi]}: {importances[fi]:.4f}\n")

    # Brief interpretation heuristics
    lines.append('\nInterpretation:\n')
    lines.append('Features with higher importance (shown above) are the ones the RandomForest uses most when\n')
    lines.append('deciding between Fresh/Medium/Spoiled. Expect environmental factors (ambient_temp, humidity),\n')
    lines.append('time-related factors (time_since_cooking, storage_time), and strong categorical indicators\n')
    lines.append('(e.g., storage_condition=refrigerated, smell descriptors or wet moisture) to dominate.\n')
    lines.append('\nSuggestions:\n')
    lines.append('- If temperature/humidity features are top-ranked, consider improving measurement precision\n')
    lines.append('- If categorical placeholders dominate, consider richer numeric interactions (temp*humidity)\n')

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    print('Model summary written to', REPORT_PATH)


if __name__ == '__main__':
    main()
