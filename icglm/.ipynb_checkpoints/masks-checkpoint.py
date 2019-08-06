import numpy as np


def coincident_trues(mask1, mask2, max_shift):
    """
    Given two n-dimensional masks returns an (n-1)-dimensional array  with the number of coincident Trues arg apart on
    axis 0. Used to calculate distances between Spike Trains
    Parameters
    ----------
    mask1: array_like
    mask2: array_like
    max_shift: int
    Returns
    -------
    coincidences : ndarray
        Number of coincident Trues on axis 0 for every dimension
    """

    coincidences = mask1 & mask2
    coincidences = np.array(coincidences, dtype=int)

    for shift in range(1, max_shift + 1):
        mask2_shifted = shift_mask(mask2, -shift, fill_value=False)
        coincidences += (mask1 & mask2_shifted)

    for shift in range(1, max_shift + 1):
        mask2_shifted = shift_mask(mask2, shift, fill_value=False)
        coincidences += (mask1 & mask2_shifted)

    coincidences = np.sum(coincidences, 0)

    return coincidences

def extend_trues(mask, argl, argr):
    """
    Given input mask returns a mask that has True values argl elements to the left and argr elements
    to the right of the original True values
    :param mask:
    :param argl:
    :param argr:
    :return:
    """

    maskl = np.copy(mask)
    maskr = np.copy(mask)

    for shift in range(1, argl + 1):
        falses = np.zeros((shift,) + mask.shape[1:], dtype=bool)
        maskl |= np.concatenate((mask[shift:, ...], falses), axis=0)

    for shift in range(1, argr + 1):
        falses = np.zeros((shift,) + mask.shape[1:], dtype=bool)
        maskr |= np.concatenate((falses, mask[:-shift, ...]), axis=0)

    return maskl | maskr

def shift_mask(arr, shift, fill_value=False):
    """
    Moves shift places along axis 0 an array filling the shifted values with fill_value
    Positive shift is to the right, negative to the left
    """

    result = np.empty_like(arr)
    if shift > 0:
        result[:shift, ...] = fill_value
        result[shift:, ...] = arr[:-shift, ...]
    elif shift < 0:
        result[shift:, ...] = fill_value
        result[:shift, ...] = arr[-shift:, ...]
    else:
        result = arr
    return result
