from abc import abstractmethod
from functools import partial

import numpy as np

from ..optimization import NewtonMethod
from ..signals import get_dt


class BayesianSpikingModel:

    @abstractmethod
    def sample(self, t, stim, stim_h=0, full=False):
        return v, r, mask_spikes

    @abstractmethod
    def simulate_subthreshold(self, t, stim, mask_spikes, stim_h=0., full=False):
        return v, r

    @abstractmethod
    def use_prior_kernels(self):
        pass

    @abstractmethod
    def gh_log_prior_kernels(self, theta):
        pass

    @abstractmethod
    def gh_log_likelihood_kernels(self, theta, **kwargs):
        return log_likelihood, g_log_likelihood, h_log_likelihood

    def get_theta(self):
        return theta

    @abstractmethod
    def get_Xmatrix(self, t, stim, mask_spikes, stim_h=0):
        return Xs

    @abstractmethod
    def set_params(self, **kwargs):
        return self

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        dt = get_dt(t)

        theta0 = self.get_theta()
        Xs = self.get_Xmatrix(t, stim, mask_spikes, stim_h=stim_h, **kwargs)

        gh_log_prior = None if not(self.use_prior_kernels()) else self.gh_log_prior_kernels
        gh_log_likelihood = partial(self.gh_log_likelihood_kernels, dt=dt, **Xs)

        optimizer = NewtonMethod(theta0=theta0, gh_log_prior=gh_log_prior, gh_log_likelihood=gh_log_likelihood, banded_h=False,
                                 verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        theta = optimizer.theta_iterations[:, -1]

        self.set_params(theta)

        N_spikes = np.sum(mask_spikes, 0)
        N_spikes = N_spikes[N_spikes > 0]
        if self.use_prior_kernels():
            log_likelihood = optimizer.log_posterior_iterations[-1] - optimizer.log_prior_iterations[-1] + \
                         np.sum(N_spikes) * np.log(dt)
        else:
            log_likelihood = optimizer.log_posterior_iterations[-1] + np.sum(N_spikes) * np.log(dt)

        log_likelihood_poisson = np.sum(N_spikes * (np.log(N_spikes / len(t)) - 1))  # L_poisson = rho0**n_neurons * np.exp(-rho0*T)

        log_likelihood_normed = (log_likelihood - log_likelihood_poisson) / np.log(2) / np.sum(N_spikes)

        return optimizer, log_likelihood_normed

    def get_log_likelihood(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import log_likelihood_normed
        dt = get_dt(t)
        v, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h)
        log_like_normed = log_likelihood_normed(dt, mask_spikes, v, r)
        return log_like_normed

    def time_rescale_transform(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import time_rescale_transform
        dt = get_dt(t)
        _, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h)
        z, ks_stats = time_rescale_transform(dt, mask_spikes, r)
        return z, ks_stats