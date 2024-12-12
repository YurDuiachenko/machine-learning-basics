
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Выполните нормировку признаков тремя способами.
# Визуализировать исходные признаки и нормированные)
# Среднее знач и СКО вычислять явно по определению и используя стандартные ф-и питона
#
# Загрузка данных
data = pd.read_csv('ex1data2.txt', header=None, names=['x', 'y', 'z'])
X = data['z'].values
y = data['x'].values


def m(X):
    m_np = np.mean(X)
    return sum(X) / len(X)


def o(X):
    o_np = np.std(X, ddof=0)
    _m = m(X)
    D = sum((x - _m) ** 2 for x in X) / len(X)
    return np.sqrt(D)


def normalize_1(X):
    max_x = max(X)
    return [x / max_x for x in X]


def normalize_2a(X):
    min_x = min(X)
    max_x = max(X)
    _m = m(X)
    return [(x - _m) / (max_x - min_x) for x in X]


def normalize_2b(X):
    _m = m(X)
    _o = o(X)
    return [(x - _m) / _o for x in X]

def normalize(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std
    return X_norm, mean, std


# Нормализация
x_1 = normalize_1(X)
x_2a = normalize_2a(X)
x_2b = normalize_2b(X)
y_1 = normalize_1(y)
y_2a = normalize_2a(y)
y_2b = normalize_2b(y)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.scatter(X, y)
plt.title('Исходные данные')
plt.xlabel('X')
plt.ylabel('y')

# 1
plt.subplot(2, 3, 2)
plt.scatter(x_1, y_1)
plt.title('Нормализация делением на максимум')
plt.xlabel('X_1')
plt.ylabel('y')

# 2a
plt.subplot(2, 3, 3)
plt.scatter(x_2a, y_2a)
plt.title('Нормализация "центрируем и делим на диапазон"')
plt.xlabel('X_2a')
plt.ylabel('y')

# 2b
plt.subplot(2, 3, 4)
plt.scatter(x_2b, y_2b)
plt.title('Нормализация "центрируем и делим на среднеквадратичное отклонение"')
plt.xlabel('X_2b')
plt.ylabel('y')

plt.tight_layout()
plt.show()
