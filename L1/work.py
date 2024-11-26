import numpy as np
import joblib

theta = joblib.load('weights.pkl')

def predict(num_cars):

    X_new = np.array([[1, num_cars]])
    # X_new = [1 3]

    return X_new.dot(theta)
    # [1 3] x [0 1] = 3

num_cars = float(input("Автомобили: "))
print(f'Прибыль с {num_cars} автомобилей: {predict(num_cars)[0]}')