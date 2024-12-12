import csv

import matplotlib.pyplot as plt
import numpy as np
import random
from tabulate import tabulate

from write import write_csv
from plot import plotData


# z(x, w) = w₀ + w₁x₁ + w₂x₂
def z(x, w):
    return np.dot(x, w)


def y(x, w):
    if z(x, w) >= 0:
        return 1
    return 0


Xs = [
    [0.2, -1.25],
    [0.4, -0.55],
    [0.8, 0.4],
    [1.15, -0.9],
    [1.4, -0.3]
]

Os = [
    [-1.4, 0.3],
    [-1, 1],
    [-0.6, -0.9],
    [-0.15, 0.15],
    [0.2, 0.85]
]

xslw_row = []

plotData(Xs, Os)
plt.show()

Xs_np = np.array(Xs)
Os_np = np.array(Os)
S = np.vstack((Os_np, Xs_np))

ones = np.ones((S.shape[0], 1))
S = np.c_[ones, S]

w = np.array([random.uniform(-1, 1) for _ in range(3)])
fields = ['класс', 'x0', 'x1', 'x2', 'w0', 'w1', 'w2', 'z', 'y', 'верно?']

epoch = 0
while epoch < 100:
    epoch += 1
    is_weight_changed = False
    rows = []
    delimiter = "-" * 30

    print(delimiter, "Эпоха", str(epoch), delimiter)
    for i in range(len(S)):
        x = S[i]

        flag = 1
        if (i < len(Os)):
            flag = 0

        pred_y = y(x, w)
        if pred_y != flag:
            w += (flag - pred_y) * x
            is_weight_changed = True

        row = ['+' if flag else '-', x[0], x[1], x[2], w[0], w[1], w[2], z(x, w), pred_y, '+' if (pred_y == flag) else '-']
        xslw_row.append(";".join([str(i) for i in row]))
        rows.append(row)

    print(tabulate(rows, headers=fields, floatfmt=".2f"))

    if not is_weight_changed:
        break

print(f"Веса: {w}")
print(f"Потребовалось эпох: {epoch}")

a = -w[1] / w[2]
b = -w[0] / w[2]

x1 = np.linspace(-2, 2, 100)
x2 = a * x1 + b

plotData(Xs, Os)
plt.plot(x1, x2)
plt.show()

write_csv("result.csv", ";".join(fields), xslw_row)