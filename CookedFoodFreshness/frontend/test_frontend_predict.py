from frontend import app
from backend import prediction


def run_headless_prediction():
    # Create a minimal App instance but do not call mainloop; use its on_predict logic by filling vars
    A = app.App()
    # Select a city preset to auto-fill ambient_temp & humidity
    # pick a city from available list
    city = list(A.city_to_region.keys())[3]
    A.city_var.set(city)
    A.fill_city_climate(city)
    A.storage_time_var.set('3.0')
    A.time_since_cooking_var.set('1.0')
    A.cat_vars['storage_condition'].set('refrigerated')
    A.cat_vars['container_type'].set('closed')
    A.cat_vars['food_type'].set('Vegetarian')
    A.cat_vars['moisture_type'].set('dry')
    A.cat_vars['cooking_method'].set('fried')
    A.cat_vars['smell'].set('neutral')

    # Call prediction logic
    A.on_predict()
    print('Result label text:', A.result_label['text'])


if __name__ == '__main__':
    run_headless_prediction()
