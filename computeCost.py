import numpy as np


def computeCost(theta, X, y):
    y_len = len(y)

    predictions = X.dot(theta)
    #               [[1 1]            [1
    # predictions =  [1 2]  x [0 1] =  2  = [1 2 3]
    #                [1 3]]            3]

    err = (predictions - y)
    err = np.dot(err.T, err)
    # err = ([1 2 3] - [3 2 1])² = [(1-3)² (2-2) (3-1)²] = [4 0 4]

    return (1 / (2 * y_len)) * err


def computeCostElements(theta, X, y):
    y_len = len(y)
    total_error = 0

    for i in range(y_len):
        prediction = 0
        for j in range(len(theta)):
            prediction += X[i, j] * theta[j]
            # [1 1] * [1 2] =  3
            #  ...            ... +=
            # [1 3] * [1 2] =  7

        error = prediction - y[i]
        total_error += error ** 2

    cost = (1 / (2 * y_len)) * total_error
    return cost


def computeCostSum(theta, X, y):
    y_len = len(y)
    total_error = 0

    for i in range(y_len):
        prediction = np.sum(X[i] * theta)

        error = prediction - y[i]
        total_error += error ** 2

    cost = (1 / (2 * y_len)) * total_error
    return cost