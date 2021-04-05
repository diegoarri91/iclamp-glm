import numpy as np

from .masks import shift_mask

def diag_indices(n, k=0):
    rows, cols = np.diag_indices(n)
    if k < 0:
        return rows[-k:], cols[:k]
    elif k > 0:
        return rows[:-k], cols[k:]
    else:
        return rows, cols


def mask_extrema(signal, mask_data=None, order=30, left_comparator=np.greater, right_comparator=np.greater_equal):
    """
    Given an nd-array signal and a mask of this array mask_data determines gives back a mask of signal of the points in
    mask_data that are also local extrema with respect to order number of points to both sides and axis=0.
    :param signal:
    :param mask_data:
    :param order:
    :param left_comparator:
    :param right_comparator:
    :return:
    """
    if mask_data is None:
        mask_data = np.ones(signal.shape, dtype=bool)
        mask_data[:order] = False
        mask_data[-order:] = False

    mask_res = np.copy(mask_data)
    mask_res[:order] = False
    mask_res[-order:] = False

    for shift in range(1, order + 1):
        arg_extrema = np.where(mask_res)
        plus  = signal[(arg_extrema[0] + shift,) + arg_extrema[1:]]
        minus = signal[(arg_extrema[0] - shift,) + arg_extrema[1:]]
        mask_extrema_plus = right_comparator(signal[mask_res], plus)
        mask_extrema_minus = left_comparator(signal[mask_res], minus)
        mask_res[mask_res] &= (mask_extrema_plus & mask_extrema_minus)

    return mask_res


def threshold_crossings(signal, threshold, upcrossing=True):
    """
    Given a signal gives the up/down crossings of signal of threshold signal = [t,shape], threshold = [shape] or threshold = float
    :param signal:
    :param threshold:
    :param upcrossing:
    :return:
    """

    mask_lower = signal <= threshold
    mask_greater = signal > threshold

    if upcrossing:
        mask_greater = shift_mask(mask_greater, -1, fill_value=False)
    else:
        mask_lower = shift_mask(mask_lower, -1, fill_value=False)

    return mask_lower & mask_greater

def unband_matrix(banded_matrix, symmetric=True):
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
    return unbanded_matrix


def mask_args_away_from_maskinp(mask_inp, argl, argr, **kwargs):
    
    # Created on 05/03/2018. Given a maskinp ndarray with isolated true values returns a mask that has False values argl and argr around these True values in axis 0.
    
    arg0 = kwargs.get('arg0', 0)
    argf = kwargs.get('argf', mask_inp.shape[0])
        
    mask0 = np.zeros(mask_inp.shape, dtype = bool)
    maskf = np.zeros(mask_inp.shape, dtype = bool)
    mask0[:arg0,...] = True
    maskf[argf:,...] = True
    
    maskl = np.copy(mask_inp)
    maskr = np.copy(mask_inp)
        
    for shift in range(1, argl + 1):
        zeros_ = np.zeros( (shift,) + mask_inp.shape[1:], dtype=bool)
        maskl |= np.concatenate( (mask_inp[shift:,...],  zeros_) )
            
    for shift in range(1, argr + 1):
        zeros_ =np.zeros( (shift,) + mask_inp.shape[1:], dtype=bool)
        maskr |= np.concatenate( (zeros_, mask_inp[:-shift,...]) )
        
    return ~(mask0 | maskl | mask_inp | maskr | maskf)
