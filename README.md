# Cooked Food Freshness Prediction — Offline Project

Short overview
---------------
This is a fully offline, beginner-friendly local project that:
- synthesizes Indian-region food-storage data (temperature & humidity variations),
- trains a scikit-learn model to predict cooked-food freshness (Fresh / Medium / Spoiled),
- provides a simple Tkinter GUI to enter sample features and get predictions.


This project generates a synthetic Indian-region dataset, trains a RandomForest
classifier to predict food freshness (Fresh / Medium / Spoiled), and provides a
small Tkinter GUI and helper scripts — all runnable offline.

Quick setup (Windows PowerShell)

1. Create and activate a Python 3.8+ virtual environment (optional but recommended):

```powershell
python -m venv .venv
; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r "CookedFoodFreshness/requirements.txt"
```

3. Generate synthetic data (creates `database/data/food_data.csv`):

```powershell
python "CookedFoodFreshness/backend/data_generation.py"
```

4. Train the model (saves `models/freshness_model.pkl` and `reports/metrics.txt`):

```powershell
python "CookedFoodFreshness/backend/model_training.py"
```

5. Generate model summary (creates `reports/model_summary.txt`):

```powershell
python "CookedFoodFreshness/backend/generate_model_summary.py"
```

6. Run the GUI (optional):

```powershell
# Set PYTHONPATH so frontend can import backend modules (run from workspace root)
Notes
- All scripts are offline and use local files only.
- If imports fail, ensure you installed packages from `requirements.txt` and that
  your PowerShell session has the project root on `PYTHONPATH` (see step 6).

License: MIT-style placeholder (for local/educational use)
# Cooked Food Freshness Prediction System

Folder structure
----------------
CookedFoodFreshness/
- backend/
	- `data_generation.py` — generates the synthetic dataset and saves CSV to `database/data/food_data.csv`.
	- `model_training.py` — trains the model using scikit-learn and writes the trained model to `models/freshness_model.pkl` and metrics to `reports/metrics.txt`.
	- `prediction.py` — helper to load the saved model and predict on a single sample dict.
	- `utils.py` — small helper utilities (pickle, dirs).
- frontend/ 
	- `app.py` — Tkinter GUI that accepts all features via numeric inputs and dropdowns; calls `backend.prediction` to predict.
	- `test_frontend_predict.py` — headless test that programmatically triggers the GUI prediction logic (useful for automated checks).
- database/
	- data/`food_data.csv` — generated synthetic dataset (saved by `data_generation.py`).
	- `local_db.sqlite` — optional placeholder local DB file.
- models/
	- `freshness_model.pkl` — trained scikit-learn model (created by `model_training.py`).
- reports/
	- `metrics.txt` — model evaluation report (classification metrics).
- `requirements.txt` — Python dependencies
- `README.md` — this file

Quick setup (PowerShell)
-------------------------
Open PowerShell in your workspace root (the folder that contains `CookedFoodFreshness`).

Optionally create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install --upgrade pip
pip install -r "CookedFoodFreshness\requirements.txt"
```

Make Python find the `backend`/`frontend` packages easily (one-time in this PowerShell session):

```powershell
$env:PYTHONPATH = "D:/just files/Food Safety Subject/Project/CookedFoodFreshness"
```

Notes
-----
- Everything runs locally. No external APIs or internet access are required—data generation, training, and prediction use only local files and installed Python packages.
- If you see import errors like `ModuleNotFoundError: No module named 'backend'`, make sure you set `PYTHONPATH` as shown above in the same PowerShell session.

How the freshness labeling logic works (short)
-------------------------------------------
The synthetic labels in `food_data.csv` are created by a heuristic scoring function that combines risk factors:
- Storage time: longer storage increases spoilage score.
- Ambient temperature: higher temperatures increase spoilage risk; low temperatures reduce it.
- Humidity: very high humidity raises spoilage risk.
- Storage condition: refrigeration reduces the score; outside storage raises it.
- Container: closed or metal mildly reduces risk; open/plastic slightly raises it.
- Time since cooking before storing: quick storage reduces risk; long delays increase it.
- Smell & texture descriptors: sour/fermented/strong smells and soggy/moist textures add to spoilage score.

The function sums these contributions into a numeric score and thresholds it into three classes:
- low score → `Fresh`
- medium score → `Medium`
- high score → `Spoiled`

This labeling is synthetic and intended for demonstration. For production, collect real labeled data and refine the labeling logic.

Troubleshooting
---------------
- If the GUI doesn't start, ensure dependencies are installed and `PYTHONPATH` points to the `CookedFoodFreshness` folder.
- To re-run steps, regenerate data, retrain, and restart the GUI.

Contact / Next steps
--------------------
If you'd like, I can:
- Enhance the GUI to show named probability breakdowns (e.g., {'Fresh':0.89,...}).
- Add input validation and presets for Indian cities to auto-fill realistic temp/humidity.
- Run hyperparameter tuning or save a small example CSV for testing.

Enjoy exploring the Cooked Food Freshness Prediction System — it runs entirely offline and is beginner-friendly.
