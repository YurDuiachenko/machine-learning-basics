import numpy as np
import matplotlib.pyplot as plt

def plotData(X, y, theta=None):
    plt.scatter(X, y, marker='x', color='r', label='Обучающие данные')
    plt.xlabel('Количество автомобилей')
    plt.ylabel('Прибыль')
    plt.title('Данные о прибыли СТО')

    if theta is not None:
        X_line = np.array([[1, x] for x in np.linspace(X.min(), X.max(), 100)])
        y_line = X_line.dot(theta)

        plt.plot(X_line[:, 1], y_line, color='blue', label='Линия регрессии')

    plt.legend()
    plt.show()


def plotCost(X, y):
    plt.scatter(X, y, marker='x', color='r', label='Ошибка')
    plt.xlabel('Итерации')
    plt.title('Градиент')

    plt.legend()
    plt.show()
