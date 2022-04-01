import pickle

import torch

from .base import BayesianSpikingModel
from ..decoding import BayesianDecoder
from ..kernels import KernelValues
from ..masks import shift_mask
from ..utils.time import get_dt


class GLM(BayesianSpikingModel, BayesianDecoder):

    def __init__(self, b, kappa, eta, noise='poisson'):
        self.b = b
        self.k = kappa
        self.h = eta
        self.noise = noise

    @property
    def r0(self):
        return torch.exp(self.b)

    def sample(self, t, stim, stim_h=0, full=False):

        b, kappa = self.b, self.k
        dt = get_dt(t)

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        else:
            shape = stim.shape

        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        mask_spk = np.zeros(shape, dtype=bool)

        # TODO. think wheteher I hav to change the convolution so r[j] depends on stim[j-1] and not stim[j]

        kappa_conv = self.k.convolve_continuous(t, stim - stim_h) + stim_h * self.k.area(dt=dt)

        j = 0
        while j < len(t):

            r[j, ...] = np.exp(kappa_conv[j, ...] - eta_conv[j, ...] - b)
            # r[j, ...] = np.log(1 + np.exp(kappa_conv[j, ...] - eta_conv[j, ...] - b))
            # r[j, ...] = np.max(np.concatenate((np.zeros(shape[1:])[None, :], (kappa_conv[j, ...] - eta_conv[j, ...] - b)[None, :]), 1), 1)
            # r[j, ...] = np.exp((kappa_conv[j, ...] - eta_conv[j, ...] - b) ** 3)
            # r[j, ...] = 0

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if self.h is not None and np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spk[j, ...]] += self.h.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        v = kappa_conv - eta_conv - b
        if full:
            return kappa_conv, eta_conv, v, r, mask_spk
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

        kappa_conv = self.k.convolve_continuous(t, stim - stim_h) + stim_h * self.k.area(dt=dt)

        if self.h is not None and len(t_spikes[0]) > 0:
            eta_conv = self.h.convolve_discrete(t, t_spikes, shape=shape[1:])
        else:
            eta_conv = np.zeros(shape)

        v = kappa_conv - eta_conv - self.b
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
        return self.k.prior is not None or self.h.prior is not None

    def gh_log_prior_kernels(self, theta):

        n_kappa = self.k.nbasis
        log_prior = 0
        g_log_prior = np.zeros(len(theta))
        h_log_prior = np.zeros((len(theta), len(theta)))

        if self.k.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.k.gh_log_prior(theta[1:n_kappa + 1])
            log_prior += _log_prior
            g_log_prior[1:n_kappa + 1] = _g_log_prior
            h_log_prior[1:n_kappa + 1, 1:n_kappa + 1] = _h_log_prior

        if self.h.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.h.gh_log_prior(theta[n_kappa + 1:])
            log_prior += _log_prior
            g_log_prior[n_kappa + 1:] = _g_log_prior
            h_log_prior[n_kappa + 1:, n_kappa + 1:] = _h_log_prior

        return log_prior, g_log_prior, h_log_prior

    def gh_log_likelihood_kernels(self, theta, dt, X=None, mask_spikes=None):

        v = X @ theta
        r = np.exp(v)

        if self.noise == 'poisson':
            log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)
            g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.matmul(X.T, r)
            h_log_likelihood = - dt * np.matmul(X.T * r, X)
        elif self.noise == 'bernoulli':
            r_s = r[mask_spikes]
            X_s = X[mask_spikes, :]
            exp_r_s = np.exp(r_s)
            r_ns = r[~mask_spikes]
            X_ns = X[~mask_spikes, :]
            log_likelihood = np.sum(np.log(1 - np.exp(-r_s * dt))) - dt * np.sum(r_ns)
            g_log_likelihood = dt * np.matmul(X_s.T, r_s / (exp_r_s - 1)) - dt * np.matmul(X_ns.T, r_ns)
            h_log_likelihood = dt * np.matmul(X_s.T * r_s * (exp_r_s * (1 - r_s * dt) - 1) / (exp_r_s - 1)**2, X_s) - \
                               dt * np.matmul(X_ns.T * r_ns, X_ns)

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        n_kappa = self.k.nbasis
        n_eta = self.h.nbasis
        theta = np.zeros((1 + n_kappa + n_eta))
        theta[0] = self.b
        theta[1:1 + n_kappa] = self.k.coefs
        theta[1 + n_kappa:] = self.h.coefs
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0):

        n_kappa = self.k.nbasis
        X_kappa = self.k.convolve_basis_continuous(t, stim - stim_h)

        if self.h is not None:
            args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.h.nbasis
            X_eta = self.h.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
            X[:, :, n_kappa + 1:] = -X_eta
        else:
            X = np.zeros(mask_spikes.shape + (1 + n_kappa,))

        X[:, :, 0] = -1.
        X[:, :, 1:n_kappa + 1] = X_kappa + np.diff(self.k.tbins)[None, None, :] * stim_h

        X = X.reshape(-1, 1 + n_kappa + n_eta)
        mask_spikes = mask_spikes.reshape(-1)

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def get_likelihood_kwargs2(self, t, stim, mask_spikes, stim_h=0):

        n_kappa = self.k.nbasis
        X_kappa = self.k.convolve_basis_continuous(t, stim - stim_h)

        if self.h is not None:
            args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.h.nbasis
            X_eta = self.h.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
            X[:, :, n_kappa + 1:] = -X_eta
        else:
            X = np.zeros(mask_spikes.shape + (1 + n_kappa,))

        X[:, :, 0] = -1.
        X[:, :, 1:n_kappa + 1] = X_kappa + np.diff(self.k.tbins)[None, None, :] * stim_h

        X_spikes, X = X[mask_spikes, :], X[np.ones(mask_spikes.shape, dtype=bool), :]

        likelihood_kwargs = dict(dt=get_dt(t), X=X, X_spikes=X_spikes)

        return likelihood_kwargs

    def set_params(self, theta):
        n_kappa = self.k.nbasis
        self.b = theta[0]
        self.k.coefs = theta[1:n_kappa + 1]
        self.h.coefs = theta[n_kappa + 1:]
        return self

    # def set_params(self, b=None, kappa_coefs=None, eta_coefs=None):
    #     self.b = b
    #     self.k.coefs = kappa_coefs
    #     self.h.coefs = eta_coefs
    #     return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):
        return super().fit(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)

    def sum_convolution_kappa_t_spikes(self, t, mask_spikes, sd_stim=1):

        inverted_t = -t[::-1]
        arg_spikes = np.where(mask_spikes)
        minus_t_spikes = (-t[arg_spikes[0]], arg_spikes[1])
        sum_conv_kappa_t_spikes = sd_stim * self.k.convolve_discrete(inverted_t, minus_t_spikes,
                                                                            shape=(mask_spikes.shape[1],))

        sum_conv_kappa_t_spikes = np.sum(sum_conv_kappa_t_spikes[::-1, :], 1)

        return sum_conv_kappa_t_spikes

    def build_K(self, dt):

        K = []
        t_support = self.k.support
        kappa_vals = self.k.interpolate(np.arange(0, t_support[1], dt))
        arg_support = int(t_support[1] / dt)
        for v in range(arg_support):
            K.append(KernelValues(values=kappa_vals[v:] * kappa_vals[:len(kappa_vals) - v],
                                  support=[v * dt, t_support[1]]))

        return K

    def get_max_band(self, dt):
        return int(self.k.support[1] / dt)

    def gh_log_likelihood_stim(self, stim, t, mask_spikes, sum_convolution_kappa_t_spikes, K, max_band, mu_stim=0,
                               sd_stim=1, stim_h=0):

        dt = get_dt(t)

        I = sd_stim * stim + mu_stim
        v, r = self.simulate_subthreshold(t, I, mask_spikes, stim_h=stim_h)

        log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)

        g_log_likelihood = dt * sum_convolution_kappa_t_spikes - \
                           dt * sd_stim * self.k.correlate_continuous(t, np.sum(r, 1))

        t_support = self.k.support
        arg_support = int(t_support[1] / dt)
        h_log_likelihood = np.zeros((max_band, len(t)))
        for v in range(arg_support):
            h_log_likelihood[v, :] += -dt ** 2 * sd_stim ** 2 * K[v].correlate_continuous(t, np.sum(r, 1))

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def decode(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        dt = get_dt(t)
        self.k.set_values(dt)
        self.k.fix_values = True
        stim_dec, optimizer = super().decode(t, mask_spikes, stim0=stim0, mu_stim=mu_stim, sd_stim=sd_stim,
                                             stim_h=stim_h, prior=prior, newton_kwargs=newton_kwargs, verbose=verbose)
        self.k.fix_values = False
        return stim_dec, optimizer
