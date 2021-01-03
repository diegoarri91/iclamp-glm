import numpy as np


def diag_indices(n, k=0):
    rows, cols = np.diag_indices(n)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def unband_matrix(banded_matrix, symmetric=True, lower=True):
    """
        Assumes banded_matrix.shape=(n_diags, lent). banded_matrix=[diag0, diag1, diag2, ....]. See scipy format
        :param banded_matrix:
        :return:
        """
    N = banded_matrix.shape[1]
    unbanded_matrix = np.zeros((N, N))
    for diag in range(banded_matrix.shape[0]):
        indices = diag_indices(N, k=diag)
        unbanded_matrix[indices] = banded_matrix[diag, :N - diag]
    if symmetric:
        indices = np.tril_indices(N)
        unbanded_matrix[indices] = unbanded_matrix.T[indices]
    if not (symmetric) and lower:
        unbanded_matrix = unbanded_matrix.T
    return unbanded_matrix


def band_matrix(unbanded_matrix, max_band=None, fill_with_nan=False):
    N = unbanded_matrix.shape[1]
    max_band = max_band if max_band is not None else N
    banded_matrix = np.zeros((max_band, N))
    if fill_with_nan:
        banded_matrix = banded_matrix * np.nan
    for diag in range(max_band):
        indices = diag_indices(N, k=diag)
        banded_matrix[diag, :N - diag] = unbanded_matrix[indices]
    return banded_matrix