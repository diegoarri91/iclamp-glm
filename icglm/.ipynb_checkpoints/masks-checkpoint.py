import numpy as np


def coincident_trues(mask1, mask2, max_shift):
    """
    Given two n-dimensional masks returns an (n-1)-dimensional array 
    with the number of coincident Trues arg apart on axis 0
    Used to calculate distances between Spike Trains
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
    
    coincidences = np.zeros(mask1.shape[1:]) * np.nan
    
    coincidences = mask1 & mask2
    coincidences = np.array(coincidences, dtype = int)
    
    for shift in range(1, max_shift + 1):
        
        mask2_shifted = shift_array(mask2, -shift, fill_value = False)
        coincidences += (mask1 & mask2_shifted)
        
    for shift in range(1, max_shift + 1):
        
        mask2_shifted = shift_array(mask2, shift, fill_value = False)
        coincidences += (mask1 & mask2_shifted)
        
    coincidences = np.sum(coincidences, 0)
        
    return coincidences

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

def shift_mask(arr, shift, fill_value=np.nan):
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