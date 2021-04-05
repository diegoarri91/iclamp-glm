import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from .base import Kernel
from ..signals import diag_indices
from ..utils.time import get_dt, searchsorted


class KernelRect(Kernel):

    def __init__(self, tbins, coefs=None, prior=None, prior_pars=None):
        self.nbasis = len(tbins) - 1
        self.tbins = np.array(tbins)
        self.support = np.array([tbins[0], tbins[-1]])
        self.coefs = np.array(coefs)
        self.prior = prior
        self.prior_pars = np.array(prior_pars)

    def interpolate(self, t, sorted_t=True):

        if sorted_t:
            res = self.interpolate_sorted(t)
        else:
            arg_bins = searchsorted(self.tbins, t, side='right') - 1

            idx = np.array(np.arange(len(arg_bins)), dtype=int)
            idx = idx[(arg_bins >= 0) & (arg_bins < len(self.tbins) - 1)]

            res = np.zeros(len(t))
            res[idx] = self.coefs[arg_bins[idx]]

        return res

    def interpolate_sorted(self, t):

        t = np.atleast_1d(t)
        res = np.zeros(len(t))

        arg_bins = searchsorted(t, self.tbins, side='left')

        for ii, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
            res[arg0:argf] = self.coefs[ii]

        return res

    def area(self, dt=None):
        return np.sum(self.coefs * np.diff(self.tbins))

    def plot_basis(self, t, ax=None, coefs=False):

        if ax is None:
            fig, ax = plt.subplots()

        arg_bins = searchsorted(t, self.tbins)

        for k, (arg0, argf) in enumerate(zip( arg_bins[:-1][::-1], arg_bins[1:][::-1] )):
            vals = np.zeros( (len(t)) )
            vals[arg0:argf] = 1.
            if coefs:
                vals[arg0:argf] = self.coefs[::-1][k + 1]
            ax.plot(t, vals, linewidth=1, color='C' + str(9 - k))

        return ax

    def copy(self):
        kernel = KernelRect(self.tbins.copy(), coefs=self.coefs.copy(), prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

    @classmethod
    def kistler_kernels(cls, delta, dt):
        kernel1 = cls(np.array([-delta, delta + dt]), [1.])
        kernel2 = cls(np.array([0, dt]), [1. / dt])
        return kernel1, kernel2

    @classmethod
    def exponential(cls, tf=None, dt=None, tau=None, A=None, prior=None, prior_pars=None):
        tbins = np.arange(0, tf, dt)
        return cls(tbins, coefs=A * np.exp(-tbins[:-1] / tau), prior=prior, prior_pars=prior_pars)

    def convolve_basis_continuous(self, t, I):
        """# Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values
        Assumes sorted t"""

        dt = get_dt(t)

        arg_bins = searchsorted(t, self.tbins)
        X = np.zeros(I.shape + (self.nbasis, ))

        basis_shape = tuple([len(t)] + [1 for ii in range(I.ndim - 1)] + [self.nbasis])
        basis = np.zeros(basis_shape)

        for k, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
            basis[arg0:argf, ..., k] = 1.

        X = fftconvolve(basis, I[..., None], axes=0)
        X = X[:len(t), ...] * dt

        return X

    def convolve_basis_discrete(self, t, s, shape=None):

        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg_bins = searchsorted(t, self.tbins)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)

        for ii, arg in enumerate(arg_s):
            for k, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
                indices = tuple([slice(arg +arg0, arg +argf)] + [s[dim][ii] for dim in range(1, len(s))] + [k])
                X[indices] += 1.

        return X

    def gh_log_prior(self, coefs):

        if self.prior == 'exponential':
            lam, mu = self.prior_pars[0], np.exp(-self.prior_pars[1] * np.diff(self.tbins[:-1]))

            log_prior = -lam * np.sum((coefs[1:] - mu * coefs[:-1]) ** 2)

            g_log_prior = np.zeros(len(coefs))
            # TODO. somethingg odd with g_log_prior[0]. FIX
            g_log_prior[1] = -2 * lam * mu[0] * (coefs[1] - mu[0] * coefs[0])
            g_log_prior[2:-1] = 2 * lam * (-mu[:-1] * coefs[:-2] + (1 + mu[1:] ** 2) * coefs[1:-1] - mu[1:] * coefs[2:])
            g_log_prior[-1] = 2 * lam * (coefs[-1] - mu[-1] * coefs[-2])
            g_log_prior = -g_log_prior

            h_log_prior = np.zeros((len(coefs), len(coefs)))

            h_log_prior[1, 1], h_log_prior[1, 2] = mu[0] ** 2, -mu[0]
            h_log_prior[2:-1, 2:-1][diag_indices(len(coefs) - 2, k=0)] = 1 + mu[1:] ** 2
            h_log_prior[2:-1, 2:-1][diag_indices(len(coefs) - 2, k=1)] = -mu[1:-1]
            h_log_prior[-1, -1] = 1
            h_log_prior = -2 * lam * h_log_prior

            h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
                np.tril_indices_from(h_log_prior, k=-1)]

        elif self.prior == 'smooth_2nd_derivative':

            lam = self.prior_pars[0]

            log_prior = -lam * np.sum((coefs[:-2] + coefs[2:] - 2 * coefs[1:-1]) ** 2)

            g_log_prior = np.zeros(len(coefs))
            g_log_prior[0] = -2 * lam * (coefs[0] - 2 * coefs[1] + coefs[2])
            g_log_prior[1] = -2 * lam * (-2 * coefs[0] + 5 * coefs[1] - 4 * coefs[2] + coefs[3])
            g_log_prior[2:-2] = -2 * lam * \
                        (coefs[:-4] - 4 * coefs[1:-3] + 6 * coefs[2:-2] - 4 * coefs[3:-1] + coefs[4:])
            g_log_prior[-2] = -2 * lam * (coefs[-4] - 4 * coefs[-3] + 5 * coefs[-2] - 2 * coefs[-1])
            g_log_prior[-1] = -2 * lam * (coefs[-3] - 2 * coefs[-2] + coefs[-1])

            h_log_prior = np.zeros((len(coefs), len(coefs)))
            h_log_prior[0, 0], h_log_prior[0, 1], h_log_prior[0, 2] = 1, -2, 1
            h_log_prior[1, 1], h_log_prior[1, 2], h_log_prior[1, 3] = 5, -4, 1
            h_log_prior[2:-2, 2:-2][diag_indices(len(coefs) - 4, k=0)] = 6
            h_log_prior[2:-2, 2:-2][diag_indices(len(coefs) - 4, k=1)] = -4
            h_log_prior[2:-2, 2:-2][diag_indices(len(coefs) - 4, k=2)] = 1
            h_log_prior[-2, -2], h_log_prior[-2, -1] = 5, -2
            h_log_prior[-1, -1] = 1
            h_log_prior = - 2 * lam * h_log_prior
            h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
                np.tril_indices_from(h_log_prior, k=-1)]

        return log_prior, g_log_prior, h_log_prior