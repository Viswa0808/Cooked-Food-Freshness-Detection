Project: Cooked Food Freshness Prediction — AI agent guidance

Purpose
-------
This file provides concise, actionable guidance for an AI coding agent working on the
CookedFoodFreshness project. It focuses on the repository layout, key files, concrete
developer workflows (run/train/predict), input surface (GUI fields) and conventions
observed in the codebase so the agent can be immediately productive.

Where to look first
-------------------
- `CookedFoodFreshness/frontend/app.py` — Tkinter GUI. Primary place to find the
  interactive inputs and example defaults used by tests.
- `CookedFoodFreshness/backend/data_generation.py` — synthetic data generator and
  authoritative source for categorical choices and `CITY_CLIMATE` presets.
- `CookedFoodFreshness/backend/model_training.py` — shows which features are used
  in training, the preprocessing (OneHotEncoder + passthrough numeric) and
  where the model/report files are written.
- `CookedFoodFreshness/backend/prediction.py` — how the saved model is loaded and
  how a single-sample dict is converted to a DataFrame for prediction.
- `database/data/food_data.csv` — generated dataset that training uses; useful
  for unit tests or example rows.

Big-picture architecture
------------------------
- Single repository, offline: data generation → training → prediction → GUI.
- Data and artifacts:
  - `database/data/food_data.csv` — synthetic dataset
  - `models/freshness_model.pkl` — saved scikit-learn Pipeline
  - `reports/metrics.txt`, `reports/model_summary.txt` — textual reports
- Components:
  - Backend: data generation, model training and helpers under `backend/`.
  - Frontend: single-file Tkinter GUI at `frontend/app.py` that imports `backend.prediction`.
  - Tests: small headless GUI invocation in `frontend/test_frontend_predict.py`.

Key developer workflows (copyable PowerShell examples)
----------------------------------------------------
Note: run these from the workspace root (folder that contains `CookedFoodFreshness`).

1) Create a virtualenv and install deps (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "CookedFoodFreshness/requirements.txt"
```

2) Generate synthetic data (creates `database/data/food_data.csv`):

```powershell
python "CookedFoodFreshness/backend/data_generation.py"
```

3) Train model (writes `models/freshness_model.pkl` and `reports/metrics.txt`):

```powershell
python "CookedFoodFreshness/backend/model_training.py"
```

4) Run GUI (optional):

```powershell
# Optional: set PYTHONPATH to the workspace root in your session if imports fail
python "CookedFoodFreshness/frontend/app.py"
```

5) Headless GUI test (example usage exercised by CI/dev):

```powershell
python "CookedFoodFreshness/frontend/test_frontend_predict.py"
```

Project-specific conventions and patterns
----------------------------------------
- Feature selection mismatch: training uses these features (see `model_training.py`):
  `['storage_time','time_since_cooking','storage_condition','container_type','food_type','moisture_type','cooking_method','texture','smell']`.
  The GUI constructs a sample dict with the same keys (numeric fields are cast to float before prediction).
- OneHotEncoder with `handle_unknown='ignore'` is used; new categorical values will not crash training/prediction but may be ignored in encoding.
- Numeric values are passed through `passthrough` in the ColumnTransformer — keep numeric names exactly as in training.
- CITY_CLIMATE constants in `backend/data_generation.py` are the source of truth for city presets and typical temp/humidity ranges.

GUI fields (exact names, types, defaults & examples)
--------------------------------------------------
The GUI in `frontend/app.py` exposes the following inputs. Use these exact keys when creating test samples or unit tests.

- `city` (preset combobox; derived from `CITY_CLIMATE`): selects a city which maps to a `region` and typical temp/humidity ranges. Example: `"Delhi"`.
- `storage_time` (numeric string input; hours; default: `"2.0"`). Example numerical values: `0.5`, `3.0`, `12.0`.
- `time_since_cooking` (numeric string input; hours; default: `"1.0"`). Example: `0.25`, `2.0`, `24.0`.
- `storage_condition` (categorical combobox) — allowed: `['refrigerated','outside']`. Example: `'refrigerated'`.
- `container_type` (categorical combobox) — allowed: `['open','closed','metal','plastic']`.
- `food_type` (categorical combobox) — allowed: `['Vegetarian','Non-Vegetarian','Seafood','Dairy','Bakery']`.
- `moisture_type` (categorical combobox) — allowed: `['dry','semi-wet','wet']`.
- `cooking_method` (categorical combobox) — allowed: `['fried','boiled','steamed','baked']`.
- `texture` (categorical combobox) — allowed: `['soft','firm','crispy','soggy','dry','moist']`.
- `smell` (categorical combobox) — allowed: `['neutral','slight','strong','sour','fermented']`.

Notes about GUI->Model mapping
-----------------------------
- The GUI casts `storage_time` and `time_since_cooking` to float before prediction — maintain numeric formats when creating programmatic inputs.
- `city` is only used to show region and typical temperature/humidity ranges; the trained model does NOT require `city`, `region`, `ambient_temp` or `humidity` (these were removed from training features). Do not add unknown keys when creating sample dicts unless you also update `model_training.py`.
- Prediction result formatting: the GUI tries to display `model.classes_` and `predict_proba` if present. If `classes_` is not available, it falls back to enumerated indices.

Examples to copy into tests or agent edits
-----------------------------------------
- Minimal valid sample dict (same shape as GUI constructs):

```python
sample = {
  'storage_time': 3.0,
  'time_since_cooking': 1.0,
  'storage_condition': 'refrigerated',
  'container_type': 'closed',
  'food_type': 'Vegetarian',
  'moisture_type': 'dry',
  'cooking_method': 'fried',
  'texture': 'dry',
  'smell': 'neutral'
}
```

- Where to run prediction programmatically:

```python
from backend.prediction import load_model, predict_sample
model = load_model()
pred, proba = predict_sample(model, sample)
```

Testing and CI hints
--------------------
- `frontend/test_frontend_predict.py` shows a small headless pattern: instantiate `app.App()` and set widget vars directly to simulate user input — use this pattern to create more unit tests.
- Avoid relying on GUI rendering in CI; interact with the App object's variables and call `on_predict()`.

Integration and external dependencies
------------------------------------
- Offline scikit-learn pipeline. No external APIs.
- Dependencies live in `CookedFoodFreshness/requirements.txt`. Install before running.

If you change features
----------------------
- If you add or remove model features, update all of these in lockstep:
  1. `backend/model_training.py` (features list + ColumnTransformer)
  2. `frontend/app.py` (GUI keys and sample dict construction)
  3. `backend/generate_model_summary.py` heuristics that reconstruct feature names
  4. tests in `frontend/test_frontend_predict.py`

What not to change without tests
-------------------------------
- Don't rename numeric feature keys used by the pipeline (`storage_time`, `time_since_cooking`) without updating the pipeline and tests.
- Don't rely on `ambient_temp` or `humidity` being present — they were intentionally removed from the model's training features.

If anything is unclear or you'd like a different level of detail (examples for new tests, CI config snippets, or PR checklist rules), tell me which section to expand.
