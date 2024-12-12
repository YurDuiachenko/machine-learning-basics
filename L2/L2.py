import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import argmin

from L1.gradientDescent import gradientDescent
from normalization import normalize

# Создайте с-му, предсказывающую стоимость б/у тракторов,
# основываясь на количестве передач и скорости оборота двигателя.
# Задачу решить:
# а) методом градиентного спуска, при этом подобрать наилучшую скорость обучения.
# б) используя аналитическое решение.
# Сравнить полученные результаты

data = pd.read_csv("ex1data2.txt", header=None, names=["Обороты", "Передачи", "Цена"])

X = data[["Обороты", "Передачи"]].values
#    [[2104 3]
# X = [1600 3]
#     [2400 3]]

y = data["Цена"].values
#    [399900
# y = 329900
#     369000]

X_normalized, mean, std = normalize(X)

# До нормализации
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], y, label="Обороты", alpha=0.7)
plt.scatter(X[:, 1], y, label="Передачи", alpha=0.7, color="red")
plt.xlabel("Признак")
plt.ylabel("Цена")
plt.title("До нормализации")
plt.legend()
plt.grid(True)

# После нормализации
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
#               [[1 2104 3]
# X_normalized = [1 1600 3]
#                [1 2400 3]]

alphas = [i/100 for i in range(0, 120, 1)]
# alphas = [0.00 0.01 ... 0.99 1.00 1.01 ... 1.19 1.20]

iter = 500

theta_init = np.zeros(X_normalized.shape[1])
# [0 0 0]

results = []
all_iterations = []
all_costs = []
all_learning_rates = []

for alpha in alphas:
    theta_init = np.random.randn(3)
    theta, costs = gradientDescent(X_normalized, y, theta_init.copy(), alpha, iter)
    results.append(costs[-1])

    for i in range(iter):
        all_iterations.append(i)
        all_costs.append(costs[i])
        all_learning_rates.append(alpha)

all_iterations = np.array(all_iterations)
all_costs = np.array(all_costs)
all_learning_rates = np.array(all_learning_rates)

fig = plt.figure(figsize=(20, 14))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(all_learning_rates, all_iterations, all_costs , c=all_costs) # cmap='coolwarm'

ax.set_xlabel('Альфа')
ax.set_ylabel('Итерации')
ax.set_zlabel('Стоимость')


# Анализ результата
alpha_opt = alphas[argmin(results)]
print(alpha_opt, min(results))
theta_opt, costs_best = gradientDescent(X_normalized, y, theta_init, alpha_opt, iter)


# θ = (Xᵀ⋅X)⁻¹⋅Xᵀ⋅y
theta_analytical = np.linalg.inv(X_normalized.T.dot(X_normalized)).dot(X_normalized.T).dot(y)


# Сравнение результатов
print("Методом градиентного спуска:", theta_opt)
print("Аналитического метода:", theta_analytical)
print("Отклонение параметров: ", np.abs(theta_opt - theta_analytical))

plt.figure(figsize=(8, 6))
plt.plot(range(len(costs_best)), costs_best, label=("Функция стоимости при " + str(alpha_opt)))
plt.xlabel("Итерации")
plt.ylabel("Функция стоимости")
plt.title("Снижение функции стоимости")
plt.legend()
plt.grid(True)
plt.show()

# Сравнение предсказаний
prediction_gradient = X_normalized.dot(theta_opt)
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



def predict(X):
    X = np.array(X)

    X_normalized = (X - mean) / std

    X_normalized = np.concatenate([[1], X_normalized])

    price_gd = X_normalized.dot(theta_opt)
    price_analytical = X_normalized.dot(theta_analytical)

    print()
    print(f"Cтоимость б/у тракторов (градиентный спуск): {price_gd}")
    print(f"Cтоимость б/у тракторов (аналитический метод): {price_analytical}")

# 1890, 3, 329999
predict([1890, 3])