from abc import abstractmethod
import numpy as np
from scipy.fftpack.helper import next_fast_len
from scipy.signal import fftconvolve

from ..masks import shift_mask
from ..utils.linalg import diag_indices
from ..utils.time import get_dt, searchsorted


class KernelFun2d:
    pass