import joblib
import numpy as np

from computeCost import computeCost
from gradientDescent import gradientDescent
from plotData import plotCost
from plotData import plotData


# Загрузка данных и подготовка
data = np.loadtxt('ex1data1.txt', delimiter=',')

X = data[:, 0].reshape(-1, 1)
#    [[1]
# X = [2]
#     [3]]

y = data[:, 1]
# y = [1 2 3]

y_len = len(y)
it = 2000

X_b = np.c_[np.ones((y_len, 1)), X]
#      [[1 1]
# X_b = [1 2]
#       [1 3]]

theta_0 = np.random.rand(2)
# theta = [0 1]

plotData(X, y, None)

# Вычисляем начальную стоимость
cost_0 = computeCost(theta_0, X_b, y)
print(f'Начальная ошибка: {cost_0}')

# Градиентный спуск
theta_final, costs = gradientDescent(X_b, y, theta_0, alpha=0.01, iters=it)

# Стоимость после оптимизации
final_cost = computeCost(theta_final, X_b, y)
print(f'Ошибка после оптимизации: {final_cost}')

# Строим графики
plotData(X, y, theta_final)
plotCost(range(it), costs)

print(theta_final)

# Сохраняем веса модели
joblib.dump(theta_final, 'weights.pkl')
