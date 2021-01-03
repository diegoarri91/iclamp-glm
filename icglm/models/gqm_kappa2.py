import pickle

import numpy as np
from scipy.signal import fftconvolve

from .base import BayesianSpikingModel
# from ..decoding import BayesianDecoder
# from ..kernels import KernelValues
from ..masks import shift_mask
from ..utils.time import get_dt


class GQMKappa2(BayesianSpikingModel):

    def __init__(self, u0, kappa, eta, quad_kappa):
        self.u0 = u0
        self.kappa = kappa
        self.eta = eta
        self.quad_kappa = quad_kappa

    def copy(self):
        pass
        # return self.__class__(u0=self.u0, kappa=self.kappa.copy(), eta=self.eta.copy())

    def save(self, path):
        params = dict(u0=self.u0, kappa=self.kappa, eta=self.eta, quad_kappa=self.quad_kappa)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        gqm = cls(u0=params['u0'], kappa=params['kappa'], eta=params['eta'], quad_kappa=params['quad_kappa'])
        return gqm

    def sample(self, t, stim, stim_h=0, full=False):

        u0, kappa = self.u0, self.kappa
        dt = get_dt(t)

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        else:
            shape = stim.shape

        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        mask_spk = np.zeros(shape, dtype=bool)

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        quad_kappa_conv = self.quad_kappa.convolve_continuous(t, stim)

        j = 0
        while j < len(t):

            r[j, ...] = np.exp(kappa_conv[j, ...] + quad_kappa_conv[j, ...] - eta_conv[j, ...] - u0)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spk[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1

        v = kappa_conv + quad_kappa_conv - eta_conv - u0
        if full:
            return kappa_conv, quad_kappa_conv, eta_conv, v, r, mask_spk
        else:
            return v, r, mask_spk

    def simulate_subthreshold(self, t, stim, mask_spk, stim_h=0, full=False):

        if stim.ndim == 1:
            # shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        if mask_spk.ndim == 1:
            # shape = (len(t), 1)
            mask_spk = mask_spk.reshape(len(t), 1)
        # else:
        #     shape = stim.shape

        shape = mask_spk.shape
        dt = get_dt(t)
        arg_spikes = np.where(shift_mask(mask_spk, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])
        spikes = np.zeros(mask_spk.shape)
        spikes[arg_spikes] = 1 / dt

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        quad_kappa_conv = self.quad_kappa.convolve_continuous(t, stim)

        if len(t_spikes[0]) > 0:
            eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        else:
            eta_conv = np.zeros(shape)

        v = kappa_conv + quad_kappa_conv - eta_conv - self.u0
        r = np.exp(v)

        if full:
            return kappa_conv, quad_kappa_conv, eta_conv, v, r
        else:
            return v, r

    def use_prior_kernels(self):
        return self.kappa.prior is not None or self.eta.prior is not None or self.quad_kappa.prior is not None

    def gh_log_prior_kernels(self, theta):

        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_quad_kappa = self.quad_kappa.n_coefs
        log_prior = 0
        g_log_prior = np.zeros(len(theta))
        h_log_prior = np.zeros((len(theta), len(theta)))

        if self.kappa.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.kappa.gh_log_prior(theta[1:n_kappa + 1])
            log_prior += _log_prior
            g_log_prior[1:n_kappa + 1] = _g_log_prior
            h_log_prior[1:n_kappa + 1, 1:n_kappa + 1] = _h_log_prior

        if self.eta.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.eta.gh_log_prior(theta[1 + n_kappa:1 + n_kappa + n_eta])
            log_prior += _log_prior
            g_log_prior[1 + n_kappa:1 + n_kappa + n_eta] = _g_log_prior
            h_log_prior[1 + n_kappa:1 + n_kappa + n_eta, 1 + n_kappa:1 + n_kappa + n_eta] = _h_log_prior

        if self.quad_kappa.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.quad_kappa.gh_log_prior(theta[1 + n_kappa + n_eta:1 + n_kappa + n_eta + n_quad_kappa])
            log_prior += _log_prior
            g_log_prior[1 + n_kappa + n_eta:1 + n_kappa + n_eta + n_quad_kappa] = _g_log_prior
            h_log_prior[1 + n_kappa + n_eta:1 + n_kappa + n_eta + n_quad_kappa, 1 + n_kappa + n_eta:1 + n_kappa + n_eta + n_quad_kappa] = _h_log_prior

        return log_prior, g_log_prior, h_log_prior

    def gh_log_likelihood_kernels(self, theta, dt, X=None, mask_spikes=None):

        v = X @ theta
        r = np.exp(v)

        log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)
        g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.matmul(X.T, r)
        h_log_likelihood = - dt * np.matmul(X.T * r, X)

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_quad_kappa = self.quad_kappa.n_coefs
        theta = np.zeros((1 + n_kappa + n_eta + n_quad_kappa))
        theta[0] = self.u0
        theta[1:1 + n_kappa] = self.kappa.coefs
        theta[1 + n_kappa:1 + n_kappa + n_eta] = self.eta.coefs
        theta[1 + n_kappa + n_eta:] = self.quad_kappa.coefs[np.triu_indices_from(self.quad_kappa.coefs)]
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0):

        dt = get_dt(t)
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_quad_kappa = self.quad_kappa.n_coefs
        X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta + n_quad_kappa,))

        X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)
        X_quad_kappa = self.quad_kappa.convolve_basis_continuous(t, stim)

        args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_spk = (t[args[0]],) + args[1:]
        spikes = np.zeros(mask_spikes.shape)
        spikes[args] = 1 / dt

        X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)

        X[:, :, 0] = -1.
        X[:, :, 1:1 + n_kappa] = X_kappa + np.diff(self.kappa.tbins)[None, None, :] * stim_h
        X[:, :, 1 + n_kappa:1 + n_kappa + n_eta] = -X_eta
        X[:, :, 1 + n_kappa + n_eta:1 + n_kappa + n_eta + n_quad_kappa] = X_quad_kappa

        X = X.reshape(-1, 1 + n_kappa + n_eta + n_quad_kappa)
        mask_spikes = mask_spikes.reshape(-1)

        likelihood_kwargs = dict(dt=dt, X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def set_params(self, theta):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        self.u0 = theta[0]
        self.kappa.coefs = theta[1:n_kappa + 1]
        self.eta.coefs = theta[n_kappa + 1:n_kappa + 1 + n_eta]

        self.quad_kappa.coefs = np.zeros((self.quad_kappa.n, self.quad_kappa.n))
        self.quad_kappa.coefs[np.triu_indices(self.quad_kappa.n)] = theta[1 + n_kappa + n_eta:]
        self.quad_kappa.coefs[np.tril_indices(self.quad_kappa.n)] = self.quad_kappa.coefs.T[np.tril_indices(self.quad_kappa.n)]
        return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):
        return super().fit(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)
