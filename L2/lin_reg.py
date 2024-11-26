import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from L1.gradientDescent import gradientDescent

# Создайте с-му, предсказывающую стоимость б/у тракторов,
# основываясь на количестве передач и скорости оборота двигателя.
# Задачу решить:
# а) методом градиентного спуска, при этом подобрать наилучшую скорость обучения.
# б) используя аналитическое решение.
# Сравнить полученные результаты

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

# Загрузка данных
data = pd.read_csv("ex1data2.txt", header=None, names=["Обороты", "Передачи", "Цена"])

# Подготовка данных
X = data[["Обороты", "Передачи"]].values
y = data["Цена"].values
X_normalized, mean, std = normalize(X)

# График данных до нормализации
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y, label="Обороты", alpha=0.7)
plt.scatter(X[:, 1], y, label="Передачи", alpha=0.7, color="red")
plt.xlabel("Признак")
plt.ylabel("Цена")
plt.title("До нормализации")
plt.legend()
plt.grid(True)

# График данных после нормализации
plt.subplot(1, 2, 2)
plt.scatter(X_normalized[:, 0], y, label="Нормализованная скорость оборотов", alpha=0.7)
plt.scatter(X_normalized[:, 1], y, label="Нормализованное количество передач", alpha=0.7, color="red")
plt.xlabel("Признака")
plt.ylabel("Цена")
plt.title("После нормализации")
plt.legend()
plt.grid(True)
plt.show()

m = len(y)

X_normalized = np.c_[np.ones((m, 1)), X_normalized]

alphas = [0.001, 0.01, 0.1, 0.15, 0.2, 0.3,  0.5, 1, 1.2, 1.3, 1.5]
iter = 400

theta_init = np.zeros(X_normalized.shape[1])
# [0 0 0]

results = {}

plt.figure(figsize=(12, 8))
for alpha in alphas:
    theta, costs = gradientDescent(X_normalized, y, theta_init.copy(), alpha, iter)
    results[alpha] = costs
    plt.plot(range(len(costs)), costs, label=f"alpha = {alpha}")

plt.xlabel("Итерации")
plt.ylabel("Функция стоимости")
plt.title("Снижение функции стоимости при различных alpha")
plt.legend()
plt.grid(True)
plt.show()

# Анализ результата
alpha_optimal = min(results, key=lambda a: results[a][-1])
print(alpha_optimal)
theta_optimal, J_history_best = gradientDescent(X_normalized, y, theta_init, alpha_optimal, iter)

# Аналитическое решение
theta_analytical = np.linalg.inv(X_normalized.T.dot(X_normalized)).dot(X_normalized.T).dot(y)

# Сравнение результатов
print("Методом градиентного спуска:", theta_optimal)
print("Аналитического метода:", theta_analytical)
print("Отклонение параметров: ", np.abs(theta_optimal - theta_analytical))

plt.figure(figsize=(8, 6))
plt.plot(range(len(J_history_best)), J_history_best, label=("Функция стоимости при " + str(alpha_optimal)))
plt.xlabel("Итерации")
plt.ylabel("Функция стоимости")
plt.title("Снижение функции стоимости")
plt.legend()
plt.grid(True)
plt.show()

# Сравнение предсказаний
prediction_gradient = X_normalized.dot(theta_optimal)
prediction_analytical = X_normalized.dot(theta_analytical)

plt.figure(figsize=(8, 6))
plt.scatter(range(m), y, label="Фактическая цена", color="red", alpha=0.6)
plt.plot(range(m), prediction_gradient, label="Предсказание (градиентный спуск)", color="orange")
plt.plot(range(m), prediction_analytical, label="Предсказание (аналитический метод)", color="green", linestyle="dashed")
plt.xlabel("Индекс данных")
plt.ylabel("Цена")
plt.title("Сравнение предсказаний")
plt.legend()
plt.grid(True)
plt.show()

