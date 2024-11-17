import numpy as np

from computeCost import computeCost


def gradientDescent(X, y, theta, alpha, iters):
    y_len = len(y)
    costs = np.zeros(iters)

    for i in range(iters):
        grad = np.zeros(len(theta))

        for j in range(len(theta)):
            gradient_sum = 0
            for k in range(y_len):
                prediction = 0
                for m in range(len(theta)):
                    prediction += X[k, m] * theta[m]
                error = prediction - y[k]
                gradient_sum += X[k, j] * error
            grad[j] = (1 / y_len) * gradient_sum

        for j in range(len(theta)):
            theta[j] -= alpha * grad[j]

        costs[i] = computeCost(theta, X, y)

    return theta, costs