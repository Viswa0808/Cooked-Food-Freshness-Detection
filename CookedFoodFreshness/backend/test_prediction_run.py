from backend import prediction


def main():
    model = prediction.load_model()
    print('Model loaded:', hasattr(model, 'predict'))

    sample1 = {
        'ambient_temp': 8.0,
        'humidity': 45.0,
        'storage_time': 1.0,
        'time_since_cooking': 0.3,
        'storage_condition': 'refrigerated',
        'container_type': 'closed',
        'food_type': 'Vegetarian',
        'moisture_type': 'dry',
        'cooking_method': 'fried'
    }

    sample2 = {
        'ambient_temp': 34.0,
        'humidity': 85.0,
        'storage_time': 20.0,
        'time_since_cooking': 6.0,
        'storage_condition': 'outside',
        'container_type': 'open',
        'food_type': 'Seafood',
        'moisture_type': 'wet',
        'cooking_method': 'steamed'
    }

    for i, s in enumerate([sample1, sample2], 1):
        pred, proba = prediction.predict_sample(model, s)
        print(f"\nSample {i} input:")
        print(s)
        print('Predicted freshness:', pred)
        if proba is not None:
            print('Predicted probabilities:', proba)


if __name__ == '__main__':
    main()
