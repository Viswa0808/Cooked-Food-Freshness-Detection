"""Helper utilities for CookedFoodFreshness project."""
import os
import pickle
import pandas as pd


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_pickle(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
