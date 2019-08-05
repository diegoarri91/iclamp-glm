import numpy as np


def get_arg(t, dt, t0=0):
    """
    Given t float, list or np.array returns argument of corresponding t values using dt
    """
    
    if 'float' in str(type(t))  or type(t) is int:
        return int(np.round((t - t0) / dt , 0))
    
    elif type(t) is list:
        return np.array(np.round((np.array(t) - t0) / dt ,0), dtype=int)
    
    elif type(t) is np.ndarray:
        return np.array(np.round((t - t0) / dt ,0), dtype=int)
    
    else:
        raise
        
def get_dt(t):
    argf = 20 if len(t) >= 20 else len(t)
    dt = np.mean(np.diff(t[:argf]))
    return dt

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