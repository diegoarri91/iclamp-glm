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
