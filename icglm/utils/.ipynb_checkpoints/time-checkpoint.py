import numpy as np


def get_arg(t, dt, t0=0, func='round'):
    """
    Given t float, list or np.array returns argument of corresponding t values using dt
    """
    if isinstance(t, float) or isinstance(t, int) or isinstance(t, np.int64) or isinstance(t, np.int32):
        float_or_int = True
    elif isinstance(t, list) or isinstance(t, np.ndarray):
        float_or_int = False

    arg = (np.array(t) - t0) / dt

    if func == 'round':
        arg = np.round(arg, 0)
    elif func == 'ceil':
        arg = np.ceil(arg)
    elif func == 'floor':
        arg = np.floor(arg)
    arg = np.array(arg, dtype=int)

    if float_or_int:
        arg = arg[()]

    return arg


def get_dt(t):
    argf = 20 if len(t) >= 20 else len(t)
    dt = np.mean(np.diff(t[:argf]))
    return dt


def searchsorted(t, s, side='left'):
    '''
    Uses np.searchsorted but handles numerical round error with care
    such that returned index satisfies
    t[i-1] < s <= t[i]
    np.searchsorted(side='right') doesn't properly handle the equality sign
    on the right side
    '''
    s = np.atleast_1d(s)
    arg = np.searchsorted(t, s, side=side)

    if len(t) > 1:
        dt = get_dt(t)
        s_ = (s - t[0]) / dt
        round_s = np.round(s_, 0)
        mask_round = np.isclose(s_, np.round(s_, 0)) & (round_s >= 0) & (round_s < len(t))
        if side == 'left':
            arg[mask_round] = np.array(round_s[mask_round], dtype=int)
        elif side == 'right':
            arg[mask_round] = np.array(round_s[mask_round], dtype=int) + 1
    else:
        s_ = s - t[0]
        mask = np.isclose(s - t[0], 0.)# & (round_s >= 0) & (round_s < len(t))
        arg[mask] = np.array(s_[mask], dtype=int)

    if len(arg) == 1:
        arg = arg[0]

    return arg
