import numpy as np
from scipy.fftpack.helper import next_fast_len
from scipy.signal import fftconvolve

from .rect import KernelRect
from .rect2d import KernelRect2d


class KernelRect1d_2d:

    def __init__(self, tbins, coefs1d=None, coefs2d=None):
        self.tbins = tbins
        self.ker1d = KernelRect(tbins=tbins, coefs=coefs1d)
        self.ker2d = KernelRect2d(tbins_x=tbins, tbins_y=tbins, coefs=coefs2d)
        self.support_x = np.array([tbins[0], tbins[-1]])

    def convolve_continuous(self, t, I):
        conv1d = self.ker1d.convolve_continuous(t, I)
        conv2d = self.ker2d.convolve_continuous(t, I)
        conv = conv1d + conv2d
        return conv

    def deconvolve_continuous(self, t, I, v):

        mask = np.ones(I.shape, dtype=bool)
        X1d = self.ker1d.convolve_basis_continuous(t, I)
        X2d = self.ker2d.convolve_basis_continuous(t, I)
        X = np.concatenate((X1d, X2d), axis=-1)
        X = X[mask, :]
        v = v[mask]

        coefs = np.linalg.lstsq(X, v, rcond=None)[0]
        self.ker1d.coefs = coefs[:len(self.tbins) - 1]
        shape = (len(self.tbins) - 1, len(self.tbins) - 1)
        self.ker2d.coefs = np.zeros(shape)
        self.ker2d.coefs[np.triu_indices(len(self.tbins) - 1)] = coefs[len(self.tbins) - 1:]
        self.ker2d.coefs[np.tril_indices(len(self.tbins) - 1)] = self.ker2d.coefs.T[np.tril_indices(len(self.tbins) - 1)]