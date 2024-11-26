import numpy as np

def warmUpExercise(n):

    # создание единичной матрицы с помощью соответсвующей функции numpy
    E_eye = np.eye(n)

    # создание единичной матрицы с помощью numpy и цикла
    E_for = np.zeros((n, n))
    for i in range(n):
        E_for[i][i] = 1

    return E_eye, E_for

print(warmUpExercise(3))
