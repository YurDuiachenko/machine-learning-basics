import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from L1.gradientDescent import gradientDescent

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std

# Функция для рисования графиков
def plot_graph(x, y, xlabel, ylabel, title, legends=None, colors=None, styles=None, scatter=False, grid=True):
    plt.figure(figsize=(8, 6))
    if not isinstance(x, list):  # Если `x` и `y` одномерные, преобразуем в список для универсальности
        x, y = [x], [y]
    colors = colors or ['blue'] * len(y)
    styles = styles or ['-'] * len(y)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if scatter:
            plt.scatter(xi, yi, label=legends[i] if legends else None, color=colors[i], alpha=0.7)
        else:
            plt.plot(xi, yi, label=legends[i] if legends else None, color=colors[i], linestyle=styles[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legends:
        plt.legend()
    if grid:
        plt.grid(True)
    plt.show()

# Загрузка данных
data = pd.read_csv("ex1data2.txt", header=None, names=["Скорость оборотов", "Количество передач", "Цена"])

# Подготовка данных
X = data[["Скорость оборотов", "Количество передач"]].values
y = data["Цена"].values
X_norm, mean, std = normalize(X)

# График данных до нормализации
plot_graph(
    [X[:, 0], X[:, 1]], [y, y],
    xlabel="Значение признака",
    ylabel="Цена",
    title="Данные до нормализации",
    legends=["Скорость оборотов", "Количество передач"],
    scatter=True
)

# График данных после нормализации
plot_graph(
    [X_norm[:, 0], X_norm[:, 1]], [y, y],
    xlabel="Нормализованное значение признака",
    ylabel="Цена",
    title="Данные после нормализации",
    legends=["Нормализованная скорость оборотов", "Нормализованное количество передач"],
    scatter=True,
    colors=["blue", "orange"]
)

m = len(y)

X_norm = np.c_[np.ones((m, 1)), X_norm]

# Подбор наилучшего alpha
alphas = [0.001, 0.01, 0.1, 0.3, 0.5, 1]
num_iters = 400

theta_init = np.zeros(X_norm.shape[1])
results = {}

for alpha in alphas:
    theta, J_history = gradientDescent(X_norm, y, theta_init.copy(), alpha, num_iters)
    results[alpha] = J_history

# График снижения функции стоимости при различных alpha
plot_graph(
    [range(len(J_history)) for J_history in results.values()],
    list(results.values()),
    xlabel="Итерации",
    ylabel="Функция стоимости",
    title="Снижение функции стоимости при различных alpha",
    legends=[f"alpha = {alpha}" for alpha in alphas]
)

# Анализ результата
min_cost_alpha = min(results, key=lambda a: results[a][-1])
theta_gd_best, J_history_best = gradientDescent(X_norm, y, theta_init, min_cost_alpha, num_iters)

# Аналитическое решение
theta_analytical = np.linalg.inv(X_norm.T.dot(X_norm)).dot(X_norm.T).dot(y)

# Сравнение результатов
print("Параметры из градиентного спуска:", theta_gd_best)
print("Параметры аналитического метода:", theta_analytical)
print(f"\nОтклонение параметров: {np.abs(theta_gd_best - theta_analytical)}")

# График снижения функции стоимости (оптимальное alpha)
plot_graph(
    range(len(J_history_best)), J_history_best,
    xlabel="Итерации",
    ylabel="Функция стоимости",
    title="Снижение функции стоимости",
    legends=["Градиентный спуск (оптимальное alpha)"]
)

# Сравнение предсказаний
y_pred_gd = X_norm.dot(theta_gd_best)
y_pred_analytical = X_norm.dot(theta_analytical)

# График сравнения предсказаний
plot_graph(
    [range(m), range(m), range(m)],
    [y, y_pred_gd, y_pred_analytical],
    xlabel="Индекс данных",
    ylabel="Цена",
    title="Сравнение предсказаний",
    legends=["Фактическая цена", "Предсказание (градиентный спуск)", "Предсказание (аналитический метод)"],
    colors=["blue", "red", "green"],
    styles=["-", "-", "--"]
)
