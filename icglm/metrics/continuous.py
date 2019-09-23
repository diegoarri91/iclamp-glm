import numpy as np


def rmse(y_true, y):
    np.sqrt(np.mean((y_true - y)**2))
