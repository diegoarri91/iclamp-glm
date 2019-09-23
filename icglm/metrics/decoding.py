import numpy as np
from scipy.linalg import cho_solve_banded, solve_triangular

from ..utils.linalg import unband_matrix


def log_det_from_banded_cholesky(banded_cholesky):
    diag_cholesky = banded_cholesky[0, :]
    return 2 * np.sum(np.log(diag_cholesky))


def sum_elements_from_inv_banded_cholesky(inv_banded_cholesky):
    n_dim = inv_banded_cholesky.shape[1]
    return np.sum(cho_solve_banded((inv_banded_cholesky, True), np.ones(n_dim)))


def trace_from_inv_banded_cholesky(inv_banded_cholesky):
    inv_cholesky = unband_matrix(inv_banded_cholesky, symmetric=False, lower=True)
    cholesky = solve_triangular(inv_cholesky, np.eye(inv_cholesky.shape[1]), lower=True)
    return np.sum(cholesky ** 2)
