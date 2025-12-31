"""Tkinter GUI for Cooked Food Freshness Prediction with city presets and labeled probabilities.
"""
import os
import sys

# Ensure the package folder (CookedFoodFreshness) is on sys.path so
# `from backend import ...` works when running the script from the project root.
HERE = os.path.dirname(__file__)
PKG_ROOT = os.path.abspath(os.path.join(HERE, '..'))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

import tkinter as tk
from tkinter import ttk, messagebox
from backend import prediction
from backend.data_generation import CITY_CLIMATE
import random


# Categorical options should match those used in data generation/training
FOOD_TYPES = ['Vegetarian', 'Non-Vegetarian', 'Seafood', 'Dairy', 'Bakery']
TEXTURE_DESCRIPTORS = ['soft', 'firm', 'crispy', 'soggy', 'dry', 'moist']
SMELL_DESCRIPTORS = ['neutral', 'slight', 'strong', 'sour', 'fermented']
STORAGE_CONDITIONS = ['refrigerated', 'outside']
MOISTURE_TYPES = ['dry', 'semi-wet', 'wet']
COOKING_METHODS = ['fried', 'boiled', 'steamed', 'baked']
CONTAINER_TYPES = ['open', 'closed', 'metal', 'plastic']


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Cooked Food Freshness Predictor')
        self.geometry('700x560')
        self.resizable(False, False)

        # Load model
        try:
            self.model = prediction.load_model()
        except Exception:
            self.model = None

        self.city_to_region = {}
        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill='both', expand=True)

        # Prepare city list from CITY_CLIMATE
        all_cities = []
        for region, info in CITY_CLIMATE.items():
            for c in info.get('cities', []):
                all_cities.append(c)
                self.city_to_region[c] = region

        row = 0

        ttk.Label(frm, text='City (preset)').grid(row=row, column=0, sticky='w', pady=6, padx=6)
        self.city_var = tk.StringVar(value=all_cities[0])
        city_cb = ttk.Combobox(frm, textvariable=self.city_var, values=all_cities, state='readonly', width=24)
        city_cb.grid(row=row, column=1, sticky='w', pady=6)
        city_cb.bind('<<ComboboxSelected>>', self.on_city_selected)
        row += 1

        ttk.Label(frm, text='Region').grid(row=row, column=0, sticky='w', padx=6)
        self.region_label = ttk.Label(frm, text='')
        self.region_label.grid(row=row, column=1, sticky='w')
        row += 1

        ttk.Label(frm, text='Typical ranges').grid(row=row, column=0, sticky='w', padx=6)
        self.range_label = ttk.Label(frm, text='')
        self.range_label.grid(row=row, column=1, sticky='w')
        row += 1

        # Numeric inputs
        ttk.Label(frm, text='storage_time (hrs)').grid(row=row, column=0, sticky='w', pady=6, padx=6)
        self.storage_time_var = tk.StringVar(value='2.0')
        ttk.Entry(frm, textvariable=self.storage_time_var, width=20).grid(row=row, column=1, sticky='w')
        row += 1

    # Removed ambient_temp and humidity_level fields

        ttk.Label(frm, text='time_since_cooking (hrs)').grid(row=row, column=0, sticky='w', pady=6, padx=6)
        self.time_since_cooking_var = tk.StringVar(value='1.0')
        ttk.Entry(frm, textvariable=self.time_since_cooking_var, width=20).grid(row=row, column=1, sticky='w')
        row += 1

        # Categorical inputs
        self.cat_vars = {}
        def add_combo(label, var_name, options):
            nonlocal row
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky='w', pady=6, padx=6)
            v = tk.StringVar(value=options[0])
            cb = ttk.Combobox(frm, textvariable=v, values=options, state='readonly', width=22)
            cb.grid(row=row, column=1, sticky='w')
            self.cat_vars[var_name] = v
            row += 1

        add_combo('food_type', 'food_type', FOOD_TYPES)
        add_combo('smell', 'smell', SMELL_DESCRIPTORS)
        add_combo('storage_condition', 'storage_condition', STORAGE_CONDITIONS)
        add_combo('moisture_type', 'moisture_type', MOISTURE_TYPES)
        add_combo('cooking_method', 'cooking_method', COOKING_METHODS)
        add_combo('container_type', 'container_type', CONTAINER_TYPES)

        # Predict button and result
        predict_btn = ttk.Button(frm, text='Predict Freshness', command=self.on_predict)
        predict_btn.grid(row=row, column=0, columnspan=2, pady=12)
        row += 1

        self.result_label = ttk.Label(frm, text='Model not loaded' if self.model is None else '', font=('Segoe UI', 11, 'bold'))
        self.result_label.grid(row=row, column=0, columnspan=2, pady=6)

        # Initialize city values
        self.fill_city_climate(self.city_var.get())

    def on_city_selected(self, event):
        self.fill_city_climate(self.city_var.get())

    def fill_city_climate(self, city):
        region = self.city_to_region.get(city)
        if not region:
            return
        info = CITY_CLIMATE.get(region, {})
        tmin, tmax = info.get('temp_range', (20, 35))
        hmin, hmax = info.get('hum_range', (30, 80))
        self.region_label.config(text=region)
        self.range_label.config(text=f'Temp: {tmin}–{tmax} °C   Humidity: {hmin}–{hmax} %')

    def on_predict(self):
        if self.model is None:
            messagebox.showerror('Model missing', 'Model not trained or not found. Run backend/model_training.py first.')
            return

        try:
            sample = {
                'storage_time': float(self.storage_time_var.get()),
                'time_since_cooking': float(self.time_since_cooking_var.get()),
                'storage_condition': self.cat_vars['storage_condition'].get(),
                'container_type': self.cat_vars['container_type'].get(),
                'food_type': self.cat_vars['food_type'].get(),
                'moisture_type': self.cat_vars['moisture_type'].get(),
                'cooking_method': self.cat_vars['cooking_method'].get(),
                'smell': self.cat_vars['smell'].get(),
            }

            pred, proba = prediction.predict_sample(self.model, sample)
            prob_text = ''
            if proba is not None:
                try:
                    labels = list(self.model.classes_)
                except Exception:
                    labels = []
                if labels and len(labels) == len(proba):
                    prob_map = {str(lbl): round(float(p), 3) for lbl, p in zip(labels, proba)}
                else:
                    prob_map = {str(i): round(float(p), 3) for i, p in enumerate(proba)}
                prob_text = f'  Probabilities: {prob_map}'

            # Add final suggestion based on prediction
            suggestion_map = {
                'Fresh': 'Perfect to eat',
                'Medium': 'Good to eat',
                'Spoiled': "Not recommended, don't eat"
            }
            suggestion = suggestion_map.get(str(pred), '')
            self.result_label.config(text=f'Predicted: {pred}{prob_text}\nFinal suggestion: {suggestion}')

        except Exception as e:
            messagebox.showerror('Prediction error', str(e))

        def on_city_selected(self, event):
            # Called when city combo changes
            city = self.city_var.get()
            self.fill_city_climate(city)

        def fill_city_climate(self, city):
            # Find the region for city and sample temp & humidity within that region's ranges
            region = self.city_to_region.get(city)
            if not region:
                return
            info = CITY_CLIMATE.get(region, {})
            tmin, tmax = info.get('temp_range', (20, 35))
            hmin, hmax = info.get('hum_range', (30, 80))
            # Add small randomness to mimic local variation
            temp = round(random.uniform(tmin, tmax) + random.gauss(0, 1.5), 1)
            hum = round(min(max(random.uniform(hmin, hmax) + random.gauss(0, 4), 0), 100), 1)
            self.ambient_temp_var.set(str(temp))
            self.humidity_var.set(str(hum))
            # Update region and typical ranges display
            self.region_label.config(text=region)
            self.range_label.config(text=f'Temp: {tmin}–{tmax} °C   Humidity: {hmin}–{hmax} %')


def main():
    app = App()
    app.mainloop()


if __name__ == '__main__':
    main()
