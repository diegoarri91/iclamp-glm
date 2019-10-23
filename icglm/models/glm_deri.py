import pickle

import numpy as np

from .base import BayesianSpikingModel
from ..decoding import BayesianDecoder
from ..kernels import KernelValues
from ..masks import shift_mask
from ..utils.time import get_dt


class GLMDeri(BayesianSpikingModel, BayesianDecoder):

    def __init__(self, u0, kappa, eta, beta, nonlinearity=np.exp, nonlinearity_str='exp', nonlinearity_pars=None,
                 noise='poisson'):
        self.u0 = u0
        self.kappa = kappa
        self.eta = eta
        self.beta = beta
        self.nonlinearity = nonlinearity
        self.nonlinearity_str = nonlinearity_str
        self.nonlinearity_pars = nonlinearity_pars
        self.noise = noise

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

        # TODO. think wheteher I hav to change the convolution so r[j] depends on stim[j-1] and not stim[j]

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        # deri = np.diff(stim, axis=0) / dt
        # deri = np.concatenate((deri, deri[-1:]), 0)
        # beta_conv = self.beta.convolve_continuous(t, deri)
        beta_conv = self.beta.convolve_continuous(t, (stim - stim_h) ** 2)

        j = 0
        while j < len(t):

            r[j, ...] = self.nonlinearity(kappa_conv[j, ...] - eta_conv[j, ...] + beta_conv[j, ...] - u0)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if self.eta is not None and np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spk[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        v = kappa_conv + beta_conv - eta_conv - u0
        if full:
            return kappa_conv, eta_conv, beta_conv, v, r, mask_spk
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

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        # deri = np.diff(stim, axis=0) / dt
        # deri = np.concatenate((deri, deri[-1:]), 0)
        # beta_conv = self.beta.convolve_continuous(t, deri)
        beta_conv = self.beta.convolve_continuous(t, (stim - stim_h) ** 2)

        if self.eta is not None and len(t_spikes[0]) > 0:
            eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        else:
            eta_conv = np.zeros(shape)

        v = kappa_conv + beta_conv - eta_conv - self.u0
        r = np.exp(v)

        if full:
            return kappa_conv, eta_conv, beta_conv, v, r
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

    def gh_log_likelihood_kernels(self, theta, dt, X=None, mask_spikes=None):

        v = X @ theta
        r = self.nonlinearity(v)

        if self.noise == 'poisson':
            if self.nonlinearity_str == 'exp':
                log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)
                g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.matmul(X.T, r)
                h_log_likelihood = - dt * np.matmul(X.T * r, X)
        elif self.noise == 'bernoulli':
            if self.nonlinearity_str == 'exp':
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

    def gh_log_likelihood_kernels2(self, theta, dt, X=None, X_spikes=None):

        Xspk_theta = np.dot(X_spikes, theta)
        X_theta = np.dot(X, theta)

        if self.nonlinearity_str == 'exp':
            exp_X_theta = np.exp(X_theta)
            log_likelihood = np.sum(Xspk_theta) - dt * np.sum(exp_X_theta)
            g_log_likelihood = np.sum(X_spikes, axis=0) - dt * np.matmul(X.T, exp_X_theta)
            h_log_likelihood = - dt * np.dot(X.T * exp_X_theta, X)
        elif self.nonlinearity_str == 'softplus':
            exp_Xspk_theta = np.exp(Xspk_theta)
            exp_X_theta = np.exp(X_theta)
            r_spk = np.log(1 + exp_Xspk_theta)
            log_likelihood = np.sum(r_spk) - dt * np.sum(np.log(1 + exp_X_theta))
            g_log_likelihood = np.matmul(X_spikes.T, exp_Xspk_theta / (r_spk * (1 + exp_Xspk_theta))) - \
                               dt * np.matmul(X.T, exp_X_theta / (1 + exp_X_theta))
            aux = exp_Xspk_theta * (r_spk - exp_Xspk_theta) / (r_spk ** 2 * (1 + exp_Xspk_theta) ** 2)
            h_log_likelihood = np.dot(X_spikes.T * aux, X_spikes) - \
                               dt * np.dot(X.T * exp_X_theta / (1 + exp_X_theta) ** 2, X)
            # print('\n', theta[1], np.min(h_log_likelihood), np.max(h_log_likelihood))
        elif self.nonlinearity_str == 'exp_alpha':
            alpha = self.nonlinearity_pars['alpha']
            exp_X_theta_alpha = np.exp(X_theta ** alpha)
            log_likelihood = alpha * np.sum(Xspk_theta) - dt * np.sum(exp_X_theta_alpha)
            g_log_likelihood = alpha * np.sum(X_spikes, axis=0) - \
                               dt * np.dot(X.T, alpha * X_theta ** (alpha - 1) * exp_X_theta_alpha)
            print(np.min(exp_X_theta_alpha), np.max(exp_X_theta_alpha))
            h_log_likelihood = - dt * np.dot(X.T * alpha * X_theta ** (alpha - 2) * exp_X_theta_alpha *\
                                             (alpha - 1 + X_theta ** alpha), X)
            print(np.min(g_log_likelihood), np.max(g_log_likelihood))
            # print(alpha, np.min(h_log_likelihood), np.max(h_log_likelihood))
            # print(np.max(alpha * X_theta ** (alpha - 2) * exp_X_theta_alpha))
            print(np.min(exp_X_theta_alpha), np.max(exp_X_theta_alpha))
        elif self.nonlinearity_str == 'threshold_poly':
            alpha = self.nonlinearity_pars['alpha']
            X = X[X_theta > 0, ...]
            X_spikes = X_spikes[Xspk_theta > 0, ...]
            X_theta = X_theta[X_theta > 0]
            Xspk_theta = Xspk_theta[Xspk_theta > 0]
            log_likelihood = alpha * np.sum(np.log(Xspk_theta)) - dt * np.sum(X_theta ** alpha)
            g_log_likelihood = alpha * np.sum(1 / Xspk_theta[:, None] * X_spikes, axis=0) - dt * alpha * np.matmul(X.T, X_theta ** (alpha - 1))
            h_log_likelihood = -alpha * np.dot(X_spikes.T / Xspk_theta ** 2, X_spikes) - \
                               dt * alpha * (alpha - 1) * np.dot(X.T * X_theta ** (alpha - 2), X)
            # print(np.min(g_log_likelihood), np.max(g_log_likelihood), np.min(h_log_likelihood),
            #       np.max(h_log_likelihood))
        elif self.nonlinearity_str == 'threshold_poly2':
            alpha = self.nonlinearity_pars['alpha']
            eps = self.nonlinearity_pars['eps']
            mask = X_theta < 0
            X_theta[mask] = eps
            X_theta[~mask] += eps
            mask = Xspk_theta < 0
            Xspk_theta[mask] = eps
            Xspk_theta[~mask] += eps
            log_likelihood = alpha * np.sum(np.log(Xspk_theta)) - dt * np.sum(X_theta ** alpha)
            g_log_likelihood = alpha * np.sum(1 / Xspk_theta[:, None] * X_spikes, axis=0) - dt * alpha * np.matmul(X.T, X_theta ** (alpha - 1))
            print(alpha * np.sum(np.log(Xspk_theta)), dt * np.sum(X_theta ** alpha), np.min(g_log_likelihood), np.max(g_log_likelihood))
            h_log_likelihood = -alpha * np.dot(X_spikes.T / Xspk_theta ** 2, X_spikes) - \
                               dt * alpha * (alpha - 1) * np.dot(X.T * X_theta ** (alpha - 2), X)

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_beta = self.beta.nbasis
        theta = np.zeros((1 + n_kappa + n_eta + n_beta))
        theta[0] = self.u0
        theta[1:1 + n_kappa] = self.kappa.coefs
        theta[1 + n_kappa:1 + n_kappa + n_eta] = self.eta.coefs
        theta[1 + n_kappa + n_eta:] = self.beta.coefs
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0):

        dt = get_dt(t)
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_beta = self.beta.nbasis
        X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)
        # deri = np.diff(stim, axis=0) / dt
        # deri = np.concatenate((deri, deri[-1:]), 0)
        # X_beta = self.beta.convolve_basis_continuous(t, deri)
        X_beta = self.beta.convolve_basis_continuous(t, (stim - stim_h) ** 2)

        if self.eta is not None:
            args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta + n_beta,))
            X[:, :, n_kappa + 1:n_kappa + 1 + n_eta] = -X_eta
        else:
            X = np.zeros(mask_spikes.shape + (1 + n_kappa,))

        X[:, :, 0] = -1.
        X[:, :, 1:n_kappa + 1] = X_kappa + np.diff(self.kappa.tbins)[None, None, :] * stim_h
        X[:, :, 1 + n_kappa + n_eta:] = X_beta

        X = X.reshape(-1, 1 + n_kappa + n_eta + n_beta)
        mask_spikes = mask_spikes.reshape(-1)

        likelihood_kwargs = dict(dt=dt, X=X, mask_spikes=mask_spikes)

        return likelihood_kwargs

    def get_likelihood_kwargs2(self, t, stim, mask_spikes, stim_h=0):

        n_kappa = self.kappa.nbasis
        X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)

        if self.eta is not None:
            args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
            X[:, :, n_kappa + 1:] = -X_eta
        else:
            X = np.zeros(mask_spikes.shape + (1 + n_kappa,))

        X[:, :, 0] = -1.
        X[:, :, 1:n_kappa + 1] = X_kappa + np.diff(self.kappa.tbins)[None, None, :] * stim_h

        X_spikes, X = X[mask_spikes, :], X[np.ones(mask_spikes.shape, dtype=bool), :]

        likelihood_kwargs = dict(dt=get_dt(t), X=X, X_spikes=X_spikes)

        return likelihood_kwargs

    def set_params(self, theta):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        self.u0 = theta[0]
        self.kappa.coefs = theta[1:n_kappa + 1]
        self.eta.coefs = theta[n_kappa + 1:n_kappa + 1 + n_eta]
        self.beta.coefs = theta[n_kappa + 1 + n_eta:]
        return self

    # def set_params(self, u0=None, kappa_coefs=None, eta_coefs=None):
    #     self.u0 = u0
    #     self.kappa.coefs = kappa_coefs
    #     self.eta.coefs = eta_coefs
    #     return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):
        return super().fit(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)

    def sum_convolution_kappa_t_spikes(self, t, mask_spikes, sd_stim=1):

        inverted_t = -t[::-1]
        arg_spikes = np.where(mask_spikes)
        minus_t_spikes = (-t[arg_spikes[0]], arg_spikes[1])
        sum_conv_kappa_t_spikes = sd_stim * self.kappa.convolve_discrete(inverted_t, minus_t_spikes,
                                                                            shape=(mask_spikes.shape[1],))

        sum_conv_kappa_t_spikes = np.sum(sum_conv_kappa_t_spikes[::-1, :], 1)

        return sum_conv_kappa_t_spikes

    def build_K(self, dt):

        K = []
        t_support = self.kappa.support
        kappa_vals = self.kappa.interpolate(np.arange(0, t_support[1], dt))
        arg_support = int(t_support[1] / dt)
        for v in range(arg_support):
            K.append(KernelValues(values=kappa_vals[v:] * kappa_vals[:len(kappa_vals) - v],
                                  support=[v * dt, t_support[1]]))

        return K

    def get_max_band(self, dt):
        return int(self.kappa.support[1] / dt)

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
            h_log_likelihood[v, :] += -dt ** 2 * sd_stim ** 2 * K[v].correlate_continuous(t, np.sum(r, 1))

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def decode(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        dt = get_dt(t)
        self.kappa.set_values(dt)
        self.kappa.fix_values = True
        stim_dec, optimizer = super().decode(t, mask_spikes, stim0=stim0, mu_stim=mu_stim, sd_stim=sd_stim,
                                             stim_h=stim_h, prior=prior, newton_kwargs=newton_kwargs, verbose=verbose)
        self.kappa.fix_values = False
        return stim_dec, optimizer
