import numpy as np

from L1.computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iters):
    y_len = len(y)
    costs = np.zeros(iters)

    for i in range(iters):
        h = X.dot(theta)
        error = h - y
        theta -= (alpha / y_len) * (X.T.dot(error))

        costs[i] = computeCost(theta, X, y)
    return theta, costs
