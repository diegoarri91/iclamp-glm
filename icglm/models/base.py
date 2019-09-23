from abc import abstractmethod
from functools import partial

import numpy as np

from ..optimization import NewtonMethod
from ..utils.time import get_dt


class BayesianSpikingModel:

    @abstractmethod
    def sample(self, t, stim, stim_h=0, full=False):
        if not full:
            v, r, mask_spikes = None, None, None
            return v, r, mask_spikes

    @abstractmethod
    def simulate_subthreshold(self, t, stim, mask_spikes, stim_h=0., full=False):
        if not full:
            v, r = None, None
            return v, r

    @abstractmethod
    def use_prior_kernels(self):
        pass

    @abstractmethod
    def gh_log_prior_kernels(self, theta):
        pass

    @abstractmethod
    def gh_log_likelihood_kernels(self, theta, dt, **kwargs):
        log_likelihood, g_log_likelihood, h_log_likelihood = None, None, None
        return log_likelihood, g_log_likelihood, h_log_likelihood

    @abstractmethod
    def get_theta(self):
        theta = None
        return theta

    @abstractmethod
    def get_likelihood_kwargs(self, t, stim, mask_spikes, stim_h=0):
        pass

    @abstractmethod
    def set_params(self, theta):
        pass

    def fit(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False, **kwargs):

        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        dt = get_dt(t)
        theta0 = self.get_theta()
        likelihood_kwargs = self.get_likelihood_kwargs(t, stim, mask_spikes, stim_h=stim_h)

        gh_log_prior = None if not(self.use_prior_kernels()) else self.gh_log_prior_kernels
        gh_log_likelihood = partial(self.gh_log_likelihood_kernels, dt=dt, **likelihood_kwargs)

        optimizer = NewtonMethod(theta0=theta0, gh_log_prior=gh_log_prior, gh_log_likelihood=gh_log_likelihood,
                                 banded_h=False, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        theta = optimizer.theta_iterations[:, -1]
        self.set_params(theta)

        return optimizer

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
