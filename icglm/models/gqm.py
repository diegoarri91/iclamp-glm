import pickle

import numpy as np

from .base import BayesianSpikingModel
from ..decoding import BayesianDecoder
from ..kernels import KernelValues
from ..masks import shift_mask
from ..utils.time import get_dt


class GQM(BayesianSpikingModel):

    def __init__(self, u0, kappa, eta, Kappa):
        self.u0 = u0
        self.kappa = kappa
        self.eta = eta
        self.Kappa = Kappa

    def copy(self):
        pass
        # return self.__class__(u0=self.u0, kappa=self.kappa.copy(), eta=self.eta.copy())

    def save(self, path):
        pass
        # params = dict(u0=self.u0, kappa=self.kappa, eta=self.eta)
        # with open(path, "wb") as fit_file:
        #     pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):
        pass
        # with open(path, "rb") as fit_file:
        #     params = pickle.load(fit_file)
        # glm = cls(u0=params['u0'], kappa=params['kappa'], eta=params['eta'])
        # return glm

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
        Kappa_conv = 0

        j = 0
        while j < len(t):

            r[j, ...] = np.exp(kappa_conv[j, ...] + Kappa_conv[j, ...] - eta_conv[j, ...] - u0)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if self.eta is not None and np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spk[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        v = kappa_conv - eta_conv - u0
        if full:
            return kappa_conv, Kappa_conv, eta_conv, v, r, mask_spk
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

    def gh_log_likelihood_kernels(self, theta, dt, X=None, mask_spikes=None):

        v = X @ theta
        r = self.nonlinearity(v)

        if self.noise == 'poisson':
            if self.nonlinearity_str == 'exp':
                log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)
                g_log_likelihood = np.sum(X[mask_spikes, :], axis=0) - dt * np.matmul(X.T, r)
                h_log_likelihood = - dt * np.matmul(X.T * r, X)
            elif self.nonlinearity_str == 'double_exp':
                exp_v = np.exp(v)
                X_s = X[mask_spikes, :]
                log_likelihood = np.sum(v[mask_spikes]) - dt * np.sum(r)
                g_log_likelihood = np.matmul(X_s.T, np.exp()) - dt * np.matmul(X.T, r)
                h_log_likelihood = - dt * np.matmul(X.T * r, X)
            elif self.nonlinearity_str == 'exp_linear':
                heavi = lambda x: (np.sign(x) + 1) / 2
                drdv = r * heavi(-v) + 1 * heavi(v)
                d2rdv2 = r * heavi(-v)
                X_s = X[mask_spikes, :]
                r_s = r[mask_spikes]
                log_likelihood = np.sum(np.log(r_s)) - dt * np.sum(r)
                g_log_likelihood = np.matmul(X_s.T, drdv[mask_spikes] / r_s) - dt * np.matmul(X.T, drdv)
                h_log_likelihood = np.matmul(X_s.T * (d2rdv2[mask_spikes] * r_s - drdv[mask_spikes] ** 2) / r_s ** 2, X_s) - \
                                   dt * np.matmul(X.T * d2rdv2, X)
            elif self.nonlinearity_str == 'exp_quadratic':
                heavi = lambda x: (np.sign(x) + 1) / 2
                drdv = r * heavi(-v) + (1 + v) * heavi(v)
                d2rdv2 = r * heavi(-v) + 1 * heavi(v)
                X_s = X[mask_spikes, :]
                r_s = r[mask_spikes]
                log_likelihood = np.sum(np.log(r_s)) - dt * np.sum(r)
                g_log_likelihood = np.matmul(X_s.T, drdv[mask_spikes] / r_s) - dt * np.matmul(X.T, drdv)
                h_log_likelihood = np.matmul(X_s.T * (d2rdv2[mask_spikes] * r_s - drdv[mask_spikes] ** 2) / r_s ** 2, X_s) - \
                                   dt * np.matmul(X.T * d2rdv2, X)
            elif self.nonlinearity_str == 'threshold_alpha':
                alpha = self.nonlinearity_pars['alpha']
                # eps = self.nonlinearity_pars['alpha']
                heavi = lambda x: (np.sign(x) + 1) / 2
                drdv = alpha * v ** (alpha - 1) * heavi(v)
                d2rdv2 = alpha * (alpha - 1) * v ** (alpha - 2) * heavi(v)
                X_s = X[mask_spikes, :]
                r_s = r[mask_spikes]
                log_likelihood = np.sum(np.log(r_s)) - dt * np.sum(r)
                g_log_likelihood = np.matmul(X_s.T, drdv[mask_spikes] / r_s) - dt * np.matmul(X.T, drdv)
                h_log_likelihood = np.matmul(X_s.T * (d2rdv2[mask_spikes] * r_s - drdv[mask_spikes] ** 2) / r_s ** 2,
                                             X_s) - \
                                   dt * np.matmul(X.T * d2rdv2, X)
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
            t_spk = (t[args[0]],) + args[1:]
            n_eta = self.eta.nbasis
            X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)
            X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
            X[:, :, n_kappa + 1:] = -X_eta
        else:
            X = np.zeros(mask_spikes.shape + (1 + n_kappa,))

        X[:, :, 0] = -1.
        X[:, :, 1:n_kappa + 1] = X_kappa + np.diff(self.kappa.tbins)[None, None, :] * stim_h

        X = X.reshape(-1, 1 + n_kappa + n_eta)
        mask_spikes = mask_spikes.reshape(-1)

        likelihood_kwargs = dict(dt=get_dt(t), X=X, mask_spikes=mask_spikes)

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
        self.u0 = theta[0]
        self.kappa.coefs = theta[1:n_kappa + 1]
        self.eta.coefs = theta[n_kappa + 1:]
        return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):
        return super().fit(t, stim, mask_spikes, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)