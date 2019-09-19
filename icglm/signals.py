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

def get_dt(t):
    argf = 20 if len(t) >= 20 else len(t)
    dt = np.mean(np.diff(t[:argf]))
    return dt

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

    mask_res = np.copy( mask_data )
    mask_res[:order] = False # added 12/10/2018 if the candidate is closer than order to a border it is false because I can't compare
    mask_res[-order:] = False # Prevents error when evaluating shifted array

    for shift in range(1, order + 1):
        arg_extrema = np.where(mask_res)
        plus  = signal[(arg_extrema[0] + shift,) + arg_extrema[1:]]
        minus = signal[(arg_extrema[0] - shift,) + arg_extrema[1:]]
        mask_extrema_plus  = right_comparator(signal[mask_res], plus)
        mask_extrema_minus = left_comparator(signal[mask_res], minus)
        mask_res[mask_res] &= (mask_extrema_plus & mask_extrema_minus)

    return mask_res

def searchsorted(t, s):    
    '''
    Uses np.searchsorted but handles numerical round error with care
    such that returned index satisfies
    t[i-1] < s <= t[i]
    np.searchsorted(side='right') doesn't properly handle the equality sign
    on the right side
    '''
    s = np.atleast_1d(s)
    arg = np.searchsorted(t, s, side='right')
    
    if len(t) > 1:
        dt = get_dt(t)
        s_ = (s - t[0]) / dt
        round_s = np.round(s_, 0)
        #print(s_, round_s, t[0], dt, len(t))
        mask_round = np.isclose(s_, np.round(s_, 0)) & (round_s >= 0) & (round_s < len(t))
        arg[mask_round] = np.array(round_s[mask_round], dtype=int)
    else:
        s_ = s - t[0]
        #round_s = np.round(s_, 0)
        #print(s_, round_s, t[0], dt, len(t))
        mask = np.isclose(s - t[0], 0.)# & (round_s >= 0) & (round_s < len(t))
        arg[mask] = np.array(s_[mask], dtype=int)
        

    if len(arg) == 1:
        arg = arg[0]
        
    return arg

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