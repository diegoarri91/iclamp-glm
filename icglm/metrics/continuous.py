import numpy as np


def rmse(y_true, y):
    return np.sqrt(np.mean((y_true - y)**2))

def explained_variance(y_true, y):
    return 1 - np.var(y_true - y) / np.var(y_true)
