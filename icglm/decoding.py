from functools import partial
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solveh_banded, cholesky_banded, cho_solve_banded, solve_triangular
import time

from .optimization import NewtonMethod
from .signals import diag_indices
from utils.time import get_dt, searchsorted
from .spiketrain import SpikeTrain


class BayesianDecoder:

    def build_K(self):
        pass

    def decode(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        newton_kwargs = newton_kwargs.copy()
        newton_kwargs['stop_cond'] = newton_kwargs['stop_cond'] * np.sum(mask_spikes)
        dt = get_dt(t)
        self.kappa.set_values(dt)
        self.kappa.fix_values = True
        sum_convolution_kappa_t_spikes = np.sum(self.convolution_kappa_t_spikes(t, mask_spikes, sd_stim=sd_stim), 1)
        K = self.build_K(dt)
        max_band = int(self.kappa.support[1] / dt)
        newton_kwargs = {} if newton_kwargs is None else newton_kwargs

        log_prior_kwargs = prior.get_log_prior_kwargs(t)
        g_log_prior = partial(prior.g_log_prior, **log_prior_kwargs)
        h_log_prior = partial(prior.h_log_prior, **log_prior_kwargs)

        gh_log_likelihood = partial(self.gh_log_likelihood_stim, t=t, mask_spikes=mask_spikes,
                                    sum_convolution_kappa_t_spikes=sum_convolution_kappa_t_spikes, K=K,
                                    max_band=max_band, mu_stim=mu_stim, sd_stim=sd_stim, stim_h=stim_h)

        optimizer = NewtonMethod(theta0=stim0, g_log_prior=g_log_prior, h_log_prior=h_log_prior,
                                 gh_log_likelihood=gh_log_likelihood, banded_h=True,
                                 theta_independent_h=True, verbose=verbose, **newton_kwargs)
        optimizer.optimize()

        self.kappa.fix_values = False

        stim_dec = optimizer.theta_iterations[:, -1]

        return stim_dec, optimizer


class MultiModelDecoder:

    def __init__(self, glms=None, t=None, mask_spk=None, mu_xi=0, sd_xi=1, tau=None, Ih=None, Imu=None, Isd=None, I_true=None,
                 n_subsample=None, neurons=None, files=None, trials=None, fit_files=None):

        self.neurons = neurons  # saved
        self.files = files  # saved
        self.trials = trials  # saved
        self.fit_files = fit_files  # saved
        self.glms = glms  # saved
        self.t = t
        self.mask_spk = mask_spk
        self.n_spikes = np.sum(self.mask_spk)
        self.I_true = I_true
        self.dt = get_dt(t)
        self.n_subsample = n_subsample
        self.n_trials = mask_spk[0].shape[1]
        self.n_neurons = len(self.glms)

        self.I = None
        self.I_dec = None
        self.tau = tau  # saved
        if Ih is None:
            self.Ih = [0] * self.n_neurons  # saved
        else:
            self.Ih = Ih
        self.Imu = Imu  # saved
        self.Isd = Isd  # saved

        self.prior = None
        self.mu_xi = mu_xi
        self.sd_xi = sd_xi

        self.dec_status = None
        self.optimization_kwargs = None

        self.log_prior_iterations = None
        self.log_posterior_iterations = None
        
        self.log_det_cov_posterior = None
        self.var_baseline = None
        self.var_I_dec = None
        self.cov_I_dec = None

        self.r2 = None
        self.rmse = None
        self.corr = None
        self.hessian_det = None

    def save(self, path, extras=None):

        dec_params = {'neurons': self.neurons, 'files': self.files, 'glms': self.glms, 'fit_files': self.fit_files,
                      'trials': self.trials, 'I_true': self.I_true, 'I_dec': self.I_dec,
                      'Ih': self.Ih, 'Imu': self.Imu, 'Isd': self.Isd, 'tau': self.tau,
                      'log_posterior_iterations': self.log_posterior_iterations, 'n_spikes': self.n_spikes,
                      'mu_xi': self.mu_xi, 'sd_xi': self.sd_xi, 'log_det_cov_prior': self.log_det_cov_prior,
                      'var_prior_baseline': self.var_prior_baseline,
                      'log_det_cov_posterior': self.log_det_cov_posterior, 'var_baseline': self.var_baseline, 
                      'n_subsample': self.n_subsample, 'dt': self.dt, 'arg0': get_arg(self.t, self.dt),
                      'shape': self.mask_spk[0].shape, 'arg_spikes': [np.where(self.mask_spk[n]) for n in range(self.n_neurons)],
                      'log_prior_iterations': self.log_prior_iterations, 'optimization_kwargs': self.optimization_kwargs,
                      'dec_status': self.dec_status, 'r2': self.r2, 'corr': self.corr}

        if extras is not None:
            dec_params = {**dec_params, **extras}

        with open(path, "wb") as pickle_file:
            pickle.dump(dec_params, pickle_file)

    @classmethod
    def load(cls, path):

        with open(path, "rb") as dec_file:
            dic = pickle.load(dec_file)

        t = np.arange(0, len(dic['I_true']), 1) * dic['dt']
        arg_spikes = dic['arg_spikes']
        mask_spk = []
        for n in range(len(dic['glms'])):
            _mask_spk_n = np.zeros(dic['shape'], dtype=bool)
            _mask_spk_n[arg_spikes[n]] = True
            mask_spk.append(_mask_spk_n)
        
        glm_dec = cls(glms=dic['glms'], t=t, mask_spk=mask_spk)

        for key, val in dic.items():
            if key not in ['glm', 'dt', 'shape', 'arg_spikes']:
                if val is not None:
                    setattr(glm_dec, key, val)

        return glm_dec

    def convolution_kappa_t_spikes(self):

        inverted_t = -self.t[::-1]

        convolution_kappa_t_spikes = np.zeros((len(self.t), self.n_neurons, self.n_trials))

        for n in range(self.n_neurons):
            arg_spikes = np.where(self.mask_spk[n])
            minus_t_spikes = (-self.t[arg_spikes[0]], arg_spikes[1])
            convolution_kappa_t_spikes[:, n, :] = self.Isd[n] * \
                                                  self.glms[n].kappa.convolve_discrete(inverted_t, minus_t_spikes,
                                                                                      shape=(self.n_trials,))

        convolution_kappa_t_spikes = convolution_kappa_t_spikes[::-1, ...]

        return convolution_kappa_t_spikes

    def r2_score(self):
        sum_square_error = np.sum( (self.I_dec - self.I_true )**2. )
        sum_square_mean = np.sum( (self.I_true - np.mean(self.I_true) )**2. )
        r2 = 1. - sum_square_error/sum_square_mean
        self.r2 = r2
        return self.r2

    def correlation_score(self):
        self.corr = self.I_true.dot(self.I_dec) / (np.linalg.norm(self.I_true) * np.linalg.norm(self.I_dec))
        return self.corr

    def banded_gh_log_likelihood(self, sum_convolution_kappa_t_spikes, K, max_band):

        log_likelihood = 0
        g_log_likelihood = self.dt * sum_convolution_kappa_t_spikes
        h_log_likelihood = np.zeros((max_band, len(self.t)))

        for n in range(self.n_neurons):
            pass
        #     _I = self.Isd[n] * self.I + self.Imu[n]
        #     _r, _u = self.glms[n].simulate_subthr(self.t, _I, self.mask_spk[n], full=True, iterating=True,
        #                                           stim_h=self.Ih[n])
        #     log_likelihood += np.sum(_u[self.mask_spk[n]]) - self.dt * np.sum(_r)
        #
        #     g_log_likelihood += -self.dt * self.Isd[n] * self.glms[n].kappa.correlate_continuous(self.t, np.sum(_r, 1),
        #                                                                                         iterating=True)
        #     t_support = self.glms[n].kappa.support
        #     arg_support = int(t_support[1] / self.dt)
        #     for v in range(arg_support):
        #         # TODO CREO QUE FALTA UN CUADRADO EN self.Isd[n]
        #         # h_log_likelihood[v, :] += -self.dt ** 2 * self.Isd[n] * K[n][v].correlate_continuous(self.t, np.sum(_r, 1))
        #
        # return log_likelihood, g_log_likelihood, h_log_likelihood

    def build_K(self):

        K = []
        for n in range(self.n_neurons):
            K.append([])
            t_support = self.glms[n].kappa.support
            kappa_vals = self.glms[n].kappa.interpolate(np.arange(0, t_support[1], self.dt))
            arg_support = int(t_support[1] / self.dt)
            for v in range(arg_support):
                K[n].append(KernelVals(values=kappa_vals[v:] * kappa_vals[:len(kappa_vals) - v], support=[v * self.dt, t_support[1]]))

        return K

    def set_log_prior_covariance(self):

        banded_h_log_prior = self.banded_ou_h_log_prior(self.prior)
        ch = cholesky_banded(-banded_h_log_prior, lower=True)
        self.log_det_cov_prior = -2 * np.sum(np.log(ch[0, :]))
        self.var_prior_baseline = np.sum(cho_solve_banded((ch, True), np.ones(len(self.t)))) / len(self.t)

        return self
    
    def set_baseline_variance(self):
        
        self.var_baseline = np.sum(cho_solve_banded(
                            (self.banded_cholesky_inv_cov, True), 
                            np.ones(len(self.t)))
                                  ) / len(self.t)
        return self
    
    def set_I_covariance(self):
        
        self.cov_I_dec = solveh_banded(-self.banded_h_log_posterior, np.eye(len(self.t)), lower=True)
        
        return self

    def set_I_variance(self):
        
        def unband(band_h):
            N = band_h.shape[1]
            h = np.zeros((N, N))
            for diag in range(band_h.shape[0]):
                indices = diag_indices(N, k=diag)
                h[indices] = band_h[diag, :N - diag]
            indices = np.tril_indices(N)
            h[indices] = h.T[indices]
            return h
        
        unb_ch = unband(self.banded_cholesky_inv_cov).T
        inv_L = solve_triangular(unb_ch, np.eye(len(self.t)), lower=True)
        
        self.var_I_dec = np.sum(inv_L ** 2, 0)
        
        return self