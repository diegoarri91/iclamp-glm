import numpy as np

from .base import BayesianSpikingModel
from .srm import SRM
from ..masks import shift_mask
from ..utils.time import get_dt


class VGLM(SRM, BayesianSpikingModel):

    def __init__(self, kappa=None, eta=None, gamma=None, vr=None, vt=None, dv=None, lam=None):
        super().__init__(vr=vr, kappa=kappa, eta=eta, vt=vt, dv=dv, gamma=gamma)
        self.lam = lam

    def copy(self):
        return self.__class__(u0=self.vr, kappa=self.kappa.copy(), eta=self.eta.copy(), gamma=self.gamma.copy(), vt=self.vt, dv=self.dv)

    def use_prior_kernels(self):
        return self.kappa.prior is not None or self.eta.prior is not None or self.gamma.prior

    def gh_log_prior_kernels(self, theta):

        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
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
            h_log_prior[1 + n_kappa:1 + n_kappa + n_eta:, 1 + n_kappa:1 + n_kappa + n_eta] = _h_log_prior

        if self.gamma.prior is not None:
            _log_prior, _g_log_prior, _h_log_prior = self.gamma.gh_log_prior(theta[2 + n_kappa + n_eta: -1])
            log_prior += _log_prior
            g_log_prior[2 + n_kappa + n_eta: -1] = _g_log_prior
            h_log_prior[2 + n_kappa + n_eta: -1, 2 + n_kappa + n_eta: -1] = _h_log_prior

        return log_prior, g_log_prior, h_log_prior

    def gh_log_likelihood_kernels(self, theta, dt, data_sub=None, X_spikes=None, X_sub=None, X=None, Y_spikes=None,
                                  Y=None):

        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_gamma = self.gamma.nbasis
        n_sub = 1 + n_kappa + n_eta
        a = theta[-1]
        # a, lam = 1, 1

        npoints_sub = data_sub.shape[0]
        Xsub_theta_data = X_sub @ theta[:1 + n_kappa + n_eta] - data_sub
        Xspk_theta = np.dot(X_spikes, theta[:1 + n_kappa + n_eta])
        Yspk_theta = np.dot(Y_spikes, theta[1 + n_kappa + n_eta: -1])
        X_theta = np.dot(X, theta[:1 + n_kappa + n_eta])
        Y_theta = np.dot(Y, theta[1 + n_kappa + n_eta: -1])
        exp_X_theta_Y_phi = np.exp(a * X_theta + Y_theta)
        # print(exp_X_theta_Y_phi)

        log_likelihood = np.sum(a * Xspk_theta + Yspk_theta) - dt * np.sum(exp_X_theta_Y_phi) - \
                         self.lam / 2 * np.sum(Xsub_theta_data**2) / npoints_sub

        g_log_likelihood = np.zeros(len(theta))
        g_log_likelihood[:1 + n_kappa + n_eta] = a * np.sum(X_spikes, axis=0) - \
                                                 dt * a * np.matmul(X.T, exp_X_theta_Y_phi) - \
                                                 self.lam * X_sub.T @ Xsub_theta_data / npoints_sub
        g_log_likelihood[1 + n_kappa + n_eta: -1] = np.sum(Y_spikes, axis=0) - \
                                                    dt * np.matmul(Y.T, exp_X_theta_Y_phi)
        g_log_likelihood[-1] = np.sum(Xspk_theta, axis=0) - \
                               dt * np.matmul(X_theta.T, exp_X_theta_Y_phi)

        h_log_likelihood = np.zeros((len(theta), len(theta)))
        h_log_likelihood[:n_sub, :n_sub] = - dt * a**2 * np.matmul(X.T * exp_X_theta_Y_phi, X) - self.lam * X_sub.T @ X_sub / npoints_sub
        h_log_likelihood[:n_sub, n_sub:-1] = - dt * a * np.matmul(X.T * exp_X_theta_Y_phi, Y)
        h_log_likelihood[:n_sub, -1] = np.sum(X_spikes, axis=0) - \
                                                 dt * np.matmul(X.T, exp_X_theta_Y_phi) - \
                                                 dt * a * np.matmul(X.T * exp_X_theta_Y_phi, X_theta)
        h_log_likelihood[n_sub:-1, n_sub:-1] = - dt * np.dot(Y.T * exp_X_theta_Y_phi, Y)
        h_log_likelihood[n_sub:-1, -1] = -dt * np.matmul(Y.T * exp_X_theta_Y_phi, X_theta)
        h_log_likelihood[-1, -1] = -dt * np.matmul(X_theta.T * exp_X_theta_Y_phi, X_theta)
        indices = np.tril_indices(len(theta))
        h_log_likelihood[indices] = h_log_likelihood.T[indices]
        # print(h_log_likelihood)

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_gamma = self.gamma.nbasis
        theta = np.zeros((3 + n_kappa + n_eta + n_gamma))
        theta[0] = self.vr
        theta[1:1 + n_kappa] = self.kappa.coefs
        theta[1 + n_kappa:1 + n_kappa + n_eta] = self.eta.coefs
        theta[1 + n_kappa + n_eta] = self.vt / self.dv
        theta[2 + n_kappa + n_eta: -1] = self.gamma.coefs / self.dv
        theta[-1] = 1 / self.dv
        return theta

    def get_likelihood_kwargs(self, t, stim, mask_spikes, data=None, mask_subthreshold=None, stim_h=0):

        dt = get_dt(t)
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        n_gamma = self.gamma.nbasis

        X = np.zeros(mask_spikes.shape + (1 + n_kappa + n_eta,))
        Y = np.zeros(mask_spikes.shape + (1 + n_gamma,))

        X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)

        args = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_spk = (t[args[0]], ) + args[1:]
        X_eta = self.eta.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)

        Y_gamma = self.gamma.convolve_basis_discrete(t, t_spk, shape=mask_spikes.shape)

        X[:, :, 0] = 1
        X[:, :, 1:1 + n_kappa] = X_kappa + np.diff(self.kappa.tbins)[None, None, :] * stim_h
        X[:, :, 1 + n_kappa:] = -X_eta
        Y[:, :, 0] = -1
        Y[:, :, 1:] = -Y_gamma

        X_spikes = X[mask_spikes, :]
        X_sub = X[mask_subthreshold, :]
        X = X[np.ones(mask_spikes.shape, dtype=bool), :]
        Y_spikes = Y[mask_spikes, :]
        Y = Y[np.ones(mask_spikes.shape, dtype=bool), :]

        Xs = dict(dt=dt, X_spikes=X_spikes, X=X, X_sub=X_sub, data_sub=data[mask_subthreshold], Y_spikes=Y_spikes, Y=Y)

        return Xs

    def fit(self, t, stim, mask_spikes, data=None, mask_subthreshold=None, stim_h=0, newton_kwargs=None, verbose=False):
        return super().fit(t, stim, mask_spikes, data=data, mask_subthreshold=mask_subthreshold, stim_h=stim_h, newton_kwargs=newton_kwargs, verbose=verbose)

