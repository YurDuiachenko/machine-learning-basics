import numpy as np


def MSE(y, h):
    return np.sum((y - h) ** 2) / len(y)


def R2(y, h):
    y_mean = np.mean(y)
    delitel = np.sum((y - y_mean) ** 2)
    znam = np.sum((y - h) ** 2)
    return 1 - (znam / delitel)
