"""
Generates a synthetic dataset for Cooked Food Freshness Prediction.
Creates >6000 rows of samples, simulates Indian city temperature/humidity ranges,
randomly generates food-related features and a target freshness_level.
Saves CSV to ../database/data/food_data.csv (relative to project root).

Usage: run this script directly (python data_generation.py)
"""
import os
import random
import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, '..', 'database', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(DATA_DIR, 'food_data.csv')


# Define Indian cities grouped by region with realistic temp/humidity ranges
CITY_CLIMATE = {
    'North': {
        'cities': ['Delhi', 'Chandigarh', 'Lucknow', 'Jaipur', 'Srinagar'],
        'temp_range': (10, 25),  # width 15
        'hum_range': (20, 80)
    },
    'South': {
        'cities': ['Chennai', 'Kochi', 'Hyderabad', 'Bengaluru', 'Pune', 'Madurai'],
        'temp_range': (24, 34),  # width 10
        'hum_range': (50, 90)
    },
    'West': {
        'cities': ['Mumbai', 'Goa', 'Ahmedabad', 'Surat'],
        'temp_range': (23, 35),  # width 12
        'hum_range': (50, 90)
    },
    'East': {
        'cities': ['Kolkata', 'Bhubaneswar', 'Guwahati', 'Patna'],
        'temp_range': (22, 32),  # width 10
        'hum_range': (50, 95)
    },
    'Central': {
        'cities': ['Bhopal', 'Nagpur', 'Indore', 'Raipur'],
        'temp_range': (20, 32),  # width 12
        'hum_range': (30, 85)
    },
    'NorthEast': {
        'cities': ['Guwahati', 'Imphal', 'Shillong'],
        'temp_range': (15, 27),  # width 12
        'hum_range': (60, 98)
    }
}


FOOD_TYPES = ['Vegetarian', 'Non-Vegetarian', 'Seafood', 'Dairy', 'Bakery']
TEXTURE_DESCRIPTORS = ['soft', 'firm', 'crispy', 'soggy', 'dry', 'moist']
SMELL_DESCRIPTORS = ['neutral', 'slight', 'strong', 'sour', 'fermented']
STORAGE_CONDITIONS = ['refrigerated', 'outside']
MOISTURE_TYPES = ['dry', 'semi-wet', 'wet']
COOKING_METHODS = ['fried', 'boiled', 'steamed', 'baked']
CONTAINER_TYPES = ['open', 'closed', 'metal', 'plastic']


def sample_city():
    # pick a region weighted equally
    region = random.choice(list(CITY_CLIMATE.keys()))
    city = random.choice(CITY_CLIMATE[region]['cities'])
    temp_min, temp_max = CITY_CLIMATE[region]['temp_range']
    hum_min, hum_max = CITY_CLIMATE[region]['hum_range']
    # Add local daily variation
    temperature = round(random.uniform(temp_min, temp_max) + random.gauss(0, 2), 1)
    humidity = round(min(max(random.uniform(hum_min, hum_max) + random.gauss(0, 5), 0), 100), 1)
    return city, region, temperature, humidity


def freshness_label(row):
    """
    Improved heuristic: compute a region-aware risk score that combines
    temperature, humidity, time_since_cooking, storage_time and other
    categorical indicators. Also apply combined-threshold rules which
    increase risk when multiple adverse conditions co-occur (e.g. high
    temp + high humidity + delayed storing).

    The returned label is one of 'Fresh', 'Medium', or 'Spoiled'.
    """
    score = 0.0

    # region baseline factor (some regions with high humidity get a small boost)
    region_factors = {
        'North': 0.0,
        'South': 0.25,
        'West': 0.15,
        'East': 0.35,
        'Central': 0.15,
        'NorthEast': 0.4
    }
    region = row.get('region')
    score += region_factors.get(region, 0.0)

    # Temperature and humidity removed; heuristic now uses only available features
    # You may want to adjust region_factors or other weights to compensate

    # Time since cooking before storing (hours) - food left out before refrigeration
    tsc = row['time_since_cooking']
    if tsc <= 0.5:
        score -= 1.5
    elif tsc <= 2:
        score -= 0.4
    elif tsc <= 6:
        score += 0.6
    elif tsc <= 24:
        score += 1.2
    else:
        score += 2.0

    # Storage time (how long it's kept stored) also matters
    st = row['storage_time']
    if st <= 2:
        score -= 1.2
    elif st <= 8:
        score -= 0.4
    elif st <= 24:
        score += 0.6
    else:
        score += 1.5

    # Storage condition
    if row['storage_condition'] == 'refrigerated':
        score -= 2.3
    else:
        score += 1.0

    # Container type
    if row['container_type'] in ['closed', 'metal']:
        score -= 0.6
    else:
        score += 0.6

    # Smell descriptors (strong indicators)
    smell = row['smell']
    if smell in ['sour', 'fermented']:
        score += 2.5
    elif smell == 'strong':
        score += 1.2

    # Texture and moisture combined
    if row['texture'] in ['soggy', 'moist'] and row['moisture_type'] == 'wet':
        score += 1.0
    if row['moisture_type'] == 'wet':
        score += 0.9
    elif row['moisture_type'] == 'semi-wet':
        score += 0.4

    # Cooking method nuance
    if row['cooking_method'] == 'fried':
        score -= 0.5
    elif row['cooking_method'] in ['boiled', 'steamed']:
        score += 0.3

    # Combined-rule boosts: removed (no temp/humidity available)

    # Final thresholds to map score -> label
    # lower score -> fresher; higher score -> more spoiled
    if score <= -0.8:
        return 'Fresh'
    elif score <= 1.8:
        return 'Medium'
    else:
        return 'Spoiled'


def generate_row():
    city, region, temp, hum = sample_city()
    storage_time = round(max(0.0, random.gauss(12, 10)), 1)  # hours
    food_type = random.choice(FOOD_TYPES)
    texture = random.choice(TEXTURE_DESCRIPTORS)
    smell = random.choice(SMELL_DESCRIPTORS)
    storage_condition = random.choice(STORAGE_CONDITIONS)
    moisture_type = random.choice(MOISTURE_TYPES)
    cooking_method = random.choice(COOKING_METHODS)
    container_type = random.choice(CONTAINER_TYPES)
    time_since_cooking = round(abs(random.gauss(2, 3)), 2)

    row = {
        'city': city,
        'region': region,
        'storage_time': storage_time,
        'food_type': food_type,
        'texture': texture,
        'smell': smell,
        'storage_condition': storage_condition,
        'moisture_type': moisture_type,
        'cooking_method': cooking_method,
        'container_type': container_type,
        'time_since_cooking': time_since_cooking,
    }
    row['freshness_level'] = freshness_label(row)
    return row


def generate_dataset(n=6000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for _ in range(n):
        rows.append(generate_row())
    df = pd.DataFrame(rows)
    # Small cleanup and ordering
    cols = ['city', 'region', 'storage_time', 'time_since_cooking', 'storage_condition', 'container_type',
            'food_type', 'moisture_type', 'cooking_method', 'texture', 'smell', 'freshness_level']
    df = df[cols]
    return df


def main():
    print('Generating synthetic dataset...')
    df = generate_dataset(n=6500)
    print('Sample rows:', len(df))
    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'Dataset saved to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
