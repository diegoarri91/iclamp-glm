from abc import abstractmethod
from functools import partial
import numpy as np

from .kernels.base import KernelValues
from .optimization import NewtonMethod, NewtonRaphson
from .utils.time import get_dt


class BayesianDecoder:

    @abstractmethod
    def sum_convolution_kappa_t_spikes(self, t, mask_spikes, sd_stim=1):
        pass

    @abstractmethod
    def build_K(self, dt):
        pass

    @abstractmethod
    def get_max_band(self, dt):
        pass

    @abstractmethod
    def gh_log_likelihood_stim(self, stim, t, mask_spikes, sum_convolution_kappa_t_spikes, K, max_band, mu_stim,
                               sd_stim, stim_h, prior_noise, sd_noise, nbatch_noise):
        pass

    def decode(self, t, mask_spikes, stim0, mu_stim, sd_stim, stim_h=0, prior=None, sd_noise=0, nbatch_noise=1, newton_kwargs=None,
               verbose=False):
        newton_kwargs = newton_kwargs.copy()
        if isinstance(mask_spikes, list):
            newton_kwargs['stop_cond'] = newton_kwargs['stop_cond'] * np.sum(np.concatenate(mask_spikes, 1))
        else:
            newton_kwargs['stop_cond'] = newton_kwargs['stop_cond'] * np.sum(mask_spikes)
        dt = get_dt(t)
        sum_convolution_kappa_t_spikes = self.sum_convolution_kappa_t_spikes(t, mask_spikes, sd_stim=sd_stim)
        K = self.build_K(dt)
        max_band = self.get_max_band(dt)
        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        log_prior_kwargs = prior.get_log_prior_kwargs(t)
        g_log_prior = partial(prior.g_log_prior, **log_prior_kwargs)
        h_log_prior = partial(prior.h_log_prior, **log_prior_kwargs)
        gh_log_likelihood = partial(self.gh_log_likelihood_stim, t=t, mask_spikes=mask_spikes,
                                    sum_convolution_kappa_t_spikes=sum_convolution_kappa_t_spikes, K=K,
                                    max_band=max_band, mu_stim=mu_stim, sd_stim=sd_stim, stim_h=stim_h, prior_noise=prior, 
                                    sd_noise=sd_noise, nbatch_noise=nbatch_noise)

        if not(sd_noise > 0):
            optimizer = NewtonMethod(theta0=stim0, g_log_prior=g_log_prior, h_log_prior=h_log_prior,
                                     gh_log_likelihood=gh_log_likelihood, banded_h=True,
                                     theta_independent_h=True, verbose=verbose, **newton_kwargs)
        else:
            optimizer = NewtonRaphson(theta0=stim0, g_log_prior=g_log_prior, h_log_prior=h_log_prior,
                                     gh_log_likelihood=gh_log_likelihood, banded_h=True,
                                     theta_independent_h=True, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        stim_dec = optimizer.theta_iterations[:, -1]

        return stim_dec, optimizer


class MultiModelDecoder(BayesianDecoder):

    def __init__(self, glms):
        self.glms = glms
        self.n_decoders = len(self.glms)

    def decode(self, t, mask_spikes, stim0=None, mu_stim=None, sd_stim=None, stim_h=None, prior=None, sd_noise=0, 
               nbatch_noise=1, newton_kwargs=None, verbose=False):

        mu_stim = mu_stim if mu_stim is not None else [0] * self.n_decoders
        sd_stim = sd_stim if sd_stim is not None else [1] * self.n_decoders
        stim_h = stim_h if stim_h is not None else [0] * self.n_decoders

        dt = get_dt(t)
        for n in range(self.n_decoders):
            self.glms[n].kappa.set_values(dt)
            self.glms[n].kappa.fix_values = True
        stim_dec, optimizer = super().decode(t, mask_spikes, stim0=stim0, mu_stim=mu_stim, sd_stim=sd_stim,
                                             stim_h=stim_h, prior=prior, sd_noise=sd_noise, newton_kwargs=newton_kwargs, verbose=verbose)
        for n in range(self.n_decoders):
            self.glms[n].kappa.fix_values = False
        return stim_dec, optimizer

    def get_max_band(self, dt):
        return max([int(self.glms[n].kappa.support[1] / dt) for n in range(self.n_decoders)])

    def sum_convolution_kappa_t_spikes(self, t, mask_spikes, sd_stim=None):

        sd_stim = sd_stim if sd_stim is not None else [1] * self.n_decoders

        sum_conv_kappa_t_spikes = []
        for n in range(self.n_decoders):
            sum_conv_kappa_t_spikes += [self.glms[n].sum_convolution_kappa_t_spikes(t, mask_spikes[n],
                                                                                      sd_stim=sd_stim[n])]
        return sum_conv_kappa_t_spikes

    @abstractmethod
    def gh_log_likelihood_stim(self, stim, t, mask_spikes, sum_convolution_kappa_t_spikes, K, max_band, mu_stim,
                               sd_stim, stim_h, prior_noise, sd_noise, nbatch_noise):

        dt = get_dt(t)

        max_band = [self.glms[n].get_max_band(dt) for n in range(self.n_decoders)]
        log_likelihood = 0
        g_log_likelihood = np.zeros((len(t)))
        h_log_likelihood = np.zeros((max(max_band), len(t)))

        for n in range(self.n_decoders):
            likelihood = self.glms[n].gh_log_likelihood_stim(stim, t, mask_spikes[n],
                                                             sum_convolution_kappa_t_spikes[n], K[n], max_band[n],
                                                             mu_stim=mu_stim[n], sd_stim=sd_stim[n], stim_h=stim_h[n], 
                                                             prior_noise=prior_noise, sd_noise=sd_noise, nbatch_noise=nbatch_noise)
            log_likelihood += likelihood[0]
            g_log_likelihood += likelihood[1]
            h_log_likelihood[:max_band[n], :] += likelihood[2]

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def build_K(self, dt):

        K = []
        for n in range(self.n_decoders):
            K += [self.glms[n].build_K(dt)]
        return K
