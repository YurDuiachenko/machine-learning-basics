import numpy as np


def gradient_descent(X, y, theta, alpha, iters):
    y_len = len(y)
    costs = np.zeros(iters)

    for i in range(iters):
        h = X.dot(theta)
        error = h - y
        theta -= (alpha / y_len) * (X.T.dot(error))

        costs[i] = compute_cost(theta, X, y)
    return theta


def compute_cost(theta, X, y):
    y_len = len(y)

    predictions = X.dot(theta)

    err = (predictions - y)
    err = np.dot(err.T, err)

    return (1 / (2 * y_len)) * err
