import pickle

import numpy as np

from .base import BayesianSpikingModel
from ..decoding import BayesianDecoder
from ..kernels import KernelValues
from ..masks import shift_mask
from ..utils.time import get_dt


class GLM(BayesianSpikingModel, BayesianDecoder):

    def __init__(self, u0, kappa, eta):
        self.u0 = u0
        self.kappa = kappa
        self.eta = eta
        
    @property
    def r0(self):
        return np.exp(-self.u0)

    def copy(self):
        return self.__class__(u0=self.u0, kappa=self.kappa.copy(), eta=self.eta.copy())

    def save(self, path):
        params = dict(u0=self.u0, kappa=self.kappa, eta=self.eta)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):

        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        glm = cls(u0=params['u0'], kappa=params['kappa'], eta=params['eta'])
        return glm

    def sample(self, t, stim, stim_h=0, full=False):

        # np.seterr(over='ignore')  # Ignore overflow warning when calculating r[j+1] which can be very big

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

        j = 0
        while j < len(t):

            r[j, ...] = np.exp(kappa_conv[j, ...] - eta_conv[j, ...] - u0)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if self.eta is not None and np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spk[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        v = kappa_conv - eta_conv - u0
        if full:
            return kappa_conv, eta_conv, v, r, mask_spk
        else:
            return v, r, mask_spk

    def simulate_subthreshold(self, t, stim, mask_spk, stim_h=0., full=False):

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        else:
            shape = stim.shape

        dt = get_dt(t)
        arg_spikes = np.where(shift_mask(mask_spk, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)

        if self.eta is not None and len(t_spikes[0]) > 0:
            eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        else:
            eta_conv = np.zeros(shape)

        v = kappa_conv - eta_conv - self.u0
        r = np.exp(v)

        if full:
            return kappa_conv, eta_conv, v, r
        else:
            return v, r

    # def log_likelihood(self, Y_spikes, Y, mask_spikes, dt):
    #
    #     Yspk_theta = np.dot(Y_spikes, theta)
    #     Y_theta = np.dot(Y, theta)
    #     exp_Y_theta = np.exp(Y_theta)
    #
    #     Log Likelihood
        # logL = np.sum(Yspk_theta) - dt * np.sum(exp_Y_theta) + np.sum(mask_spikes) * np.log(dt)

        # return logL

    def use_prior_kernels(self):
        return self.kappa.prior is not None or self.eta.prior is not None

    def gh_log_prior_kernels(self, theta):

        n_kappa = self.kappa.nbasis
        log_prior = 0
        g_log_prior = np.zeros(len(theta))
        h_log_prior = np.zeros((len(theta), len(theta)))

        if self.kappa.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.kappa.gh_log_prior(theta[1:n_kappa + 1])
            log_prior += _log_prior
            g_log_prior[1:n_kappa + 1] = _g_log_prior
            h_log_prior[1:n_kappa + 1, 1:n_kappa + 1] = _h_log_prior

        if self.eta.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.eta.gh_log_prior(theta[n_kappa + 1:])
            log_prior += _log_prior
            g_log_prior[n_kappa + 1:] = _g_log_prior
            h_log_prior[n_kappa + 1:, n_kappa + 1:] = _h_log_prior

        return log_prior, g_log_prior, h_log_prior

    def gh_log_likelihood_kernels(self, theta, dt, X=None, X_spikes=None):

        Xspk_theta = np.dot(X_spikes, theta)
        X_theta = np.dot(X, theta)
        exp_X_theta = np.exp(X_theta)

        log_likelihood = np.sum(Xspk_theta) - dt * np.sum(exp_X_theta)
        g_log_likelihood = np.sum(X_spikes, axis=0) - dt * np.matmul(X.T, exp_X_theta)
        h_log_likelihood = - dt * np.dot(X.T * exp_X_theta, X)

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        theta = np.zeros((1 + n_kappa + n_eta))
        theta[0] = self.u0
        theta[1:1 + n_kappa] = self.kappa.coefs
        theta[1 + n_kappa:] = self.eta.coefs
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0):

        n_kappa = self.kappa.nbasis
        X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)

        if self.eta is not None:
            args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]], ) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
            X[:, :, n_kappa + 1:] = -X_eta
        else:
            X = np.zeros(mask_spikes.shape + (1 + n_kappa, ))

        X[:, :, 0] = -1.
        X[:, :, 1:n_kappa + 1] = X_kappa + np.diff(self.kappa.tbins)[None, None, :] * stim_h

        X_spikes, X = X[mask_spikes, :], X[np.ones(mask_spikes.shape, dtype=bool), :]

        likelihood_kwargs = dict(dt=get_dt(t), X=X, X_spikes=X_spikes)

        return likelihood_kwargs

    def set_params(self, theta):
        n_kappa = self.kappa.nbasis
        self.u0 = theta[0]
        self.kappa.coefs = theta[1:n_kappa + 1]
        self.eta.coefs = theta[n_kappa + 1:]
        return self

    # def set_params(self, u0=None, kappa_coefs=None, eta_coefs=None):
    #     self.u0 = u0
    #     self.kappa.coefs = kappa_coefs
    #     self.eta.coefs = eta_coefs
    #     return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):
        return super().fit(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)

    def convolution_kappa_t_spikes(self, t, mask_spikes, sd_stim=1):

        inverted_t = -t[::-1]
        arg_spikes = np.where(mask_spikes)
        minus_t_spikes = (-t[arg_spikes[0]], arg_spikes[1])
        convolution_kappa_t_spikes = sd_stim * self.kappa.convolve_discrete(inverted_t, minus_t_spikes,
                                                                         shape=(mask_spikes.shape[1],))

        convolution_kappa_t_spikes = convolution_kappa_t_spikes[::-1, :]

        return convolution_kappa_t_spikes

    def build_K(self, dt):

        K = []
        t_support = self.kappa.support
        kappa_vals = self.kappa.interpolate(np.arange(0, t_support[1], dt))
        arg_support = int(t_support[1] / dt)
        for v in range(arg_support):
            K.append(KernelValues(values=kappa_vals[v:] * kappa_vals[:len(kappa_vals) - v], support=[v * dt, t_support[1]]))

        return K

    def gh_log_likelihood_stim(self, stim, t, mask_spikes, sum_convolution_kappa_t_spikes, K, max_band, mu_stim=0,
                               sd_stim=1, stim_h=0):

        dt = get_dt(t)

        I = sd_stim * stim + mu_stim
        v, r = self.simulate_subthreshold(t, I, mask_spikes, stim_h=stim_h)

        log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)

        g_log_likelihood = dt * sum_convolution_kappa_t_spikes - \
                           dt * sd_stim * self.kappa.correlate_continuous(t, np.sum(r, 1))

        t_support = self.kappa.support
        arg_support = int(t_support[1] / dt)
        h_log_likelihood = np.zeros((max_band, len(t)))
        for v in range(arg_support):
            h_log_likelihood[v, :] += -dt**2 * sd_stim**2 * K[v].correlate_continuous(t, np.sum(r, 1))

        return log_likelihood, g_log_likelihood, h_log_likelihood
