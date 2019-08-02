import numpy as np
import sys

from scipy.signal import fftconvolve

from GLM import GLM

from fun_signals import get_dt, diag_indices


class GLMDecoder:

    def __init__(self, glm=None, t=None, mask_spk=None, tau=None, Imu=None, Istd=None, I_true=None):

        self.glm = glm
        self.t = t
        self.mask_spk = mask_spk
        self.I_true = I_true
        self.dt = get_dt(t)
        self.trials = mask_spk.shape[2]
        self.n_neurons = len(self.glm)

        self.I = None
        self.tau = tau
        self.Imu = Imu
        self.Istd = Istd

        self.v = None
        self.r = None

    def convolution_kappa_t_spikes(self):

        inverted_t = -self.t[::-1]

        convolution_kappa_t_spikes = np.zeros((len(self.t), self.n_neurons, self.trials))

        for n in range(self.n_neurons):
            arg_spikes = np.where(self.mask_spk[:, n, :])
            minus_t_spikes = (-self.t[arg_spikes[0]], arg_spikes[1])
            convolution_kappa_t_spikes[:, n, :] = self.glm[n].kappa.convolve_discrete(inverted_t, minus_t_spikes, shape=(self.trials, ))

        convolution_kappa_t_spikes = convolution_kappa_t_spikes[::-1, ...]

        return convolution_kappa_t_spikes

    def correlation_kappa_r(self):
        # TODO paralelize
        correlation_kappa_r = np.zeros((len(self.t), self.n_neurons, self.trials))
        
        for n in range(self.n_neurons):
            convolution = self.glm[n].kappa.convolve_continuous(self.t, self.r[::-1, n, :]) / self.dt
            correlation_kappa_r[:, n, :] = convolution[::-1, :]
            
        return correlation_kappa_r

    def R2(self):
        sum_square_error = np.sum( (self.I - self.I_true )**2. )
        sum_square_mean = np.sum( (self.I_true - np.mean(self.I_true) )**2. )
        R2 = 1. - sum_square_error/sum_square_mean
        return R2
    
    def inv_gp_covariance_matrix(self):
        try:
            Sigma = self.Istd**2 * np.exp(-np.abs(self.t[:, None] - self.t[None, :]) / self.tau)
        except MemoryError:
            Sigma = np.zeros((len(self.t), len(self.t)))
            for ii in range(len(self.t)):
                Sigma[ii, :] = self.Istd**2 * np.exp(-np.abs(self.t[ii] - self.t) / self.tau)
        invSigma = np.linalg.inv(Sigma)
        del Sigma
        return invSigma

    def gh_log_likelihood(self, sum_convolution_kappa_t_spikes, K):

        log_likelihood = 0
        g_log_likelihood = self.dt * sum_convolution_kappa_t_spikes
        h_log_likelihood = np.zeros((len(self.t), len(self.t)))

        for n in range(self.n_neurons):
            _r, _v = self.glm[n].simulate_subthr(self.t, self.I, self.mask_spk[:, n, :], full=True)
            log_likelihood += np.sum(_v[self.mask_spk[:, n, :]]) - self.dt * np.sum(_r)

            g_log_likelihood += -self.dt * self.glm[n].kappa.correlate_continuous(self.t, np.sum(_r, 1))
            h_log_likelihood += -self.dt ** 2 * self.glm[n].kappa.correlate_continuous(
                self.t, np.sum(np.stack([_r] * len(self.t), 1) * K[..., n, None], 2))

            # g_log_likelihood += -self.dt * np.sum(self.glm[n].kappa.correlate_continuous(self.t, _r), 1)
#             convolution = self.glm[n].kappa.convolve_continuous(self.t, sum([_r[:, k] * K[..., n] for k in range(_r.shape[1])])[::-1]) / self.dt
#             convolution = self.glm[n].kappa.convolve_continuous(self.t, np.sum(np.stack([_r] * len(self.t), 1) * K[..., n, None], 2)[::-1]) / self.dt
            # correlation = convolution[::-1]

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def gh_log_likelihood2(self, sum_convolution_kappa_t_spikes, K, h_):

            log_likelihood = 0
            g_log_likelihood = self.dt * sum_convolution_kappa_t_spikes
            h_log_likelihood = np.zeros((len(self.t), len(self.t)))

            for n in range(self.n_neurons):
                _r, _v = self.glm[n].simulate_subthr(self.t, self.I, self.mask_spk[:, n, :], full=True)
                log_likelihood += np.sum(_v[self.mask_spk[:, n, :]]) - self.dt * np.sum(_r)

                g_log_likelihood += -self.dt * self.glm[n].kappa.correlate_continuous(self.t, np.sum(_r, 1))
                for v in :
                    h_log_likelihood[:, ] += -self.dt ** 2 * self.glm[n].kappa.correlate_continuous(
                        self.t, np.sum(np.stack([_r] * len(self.t), 1) * K[..., n, None], 2))

            return log_likelihood, g_log_likelihood, h_log_likelihood

    def estimate_MAP_I(self, I0, optimization_kwargs=None):

        if optimization_kwargs is None:
            optimization_kwargs = {}

        max_iterations = optimization_kwargs.get('max_iterations', int(5e3))
        stop_cond = optimization_kwargs.get('stop_cond', 1e-7)
        learning_rate = optimization_kwargs.get('learning_rate', 1e0)

        dt = self.dt
        
        invSigma = self.inv_gp_covariance_matrix()
        h_log_prior = -invSigma
        sum_convolution_kappa_t_spikes = np.sum(self.convolution_kappa_t_spikes(), axis=(1, 2))
        K = np.zeros((len(self.t), len(self.t), self.n_neurons))
        for n in range(self.n_neurons):
            for v in range(len(self.t)):
                K[v:, v, n] = self.glm[n].kappa.interpolate(self.t)[:len(self.t)-v]

        log_prior_iterations = []
        log_posterior_iterations = []

        self.I = I0
        
        for ii in range(max_iterations):

            # prior
            log_prior = -1/2 * np.dot(self.I - self.Imu, np.dot(invSigma, self.I - self.Imu))
            g_log_prior = -np.dot(invSigma, self.I - self.Imu)

            # likelihood
            log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(sum_convolution_kappa_t_spikes, K)

            # posterior
            log_posterior = log_likelihood + log_prior
            g_log_posterior = g_log_likelihood + g_log_prior
            h_log_posterior = h_log_likelihood + h_log_prior

            log_prior_iterations += [log_prior]
            log_posterior_iterations += [log_posterior]

            old_log_posterior = 0 if len(log_posterior_iterations) < 2 else log_posterior_iterations[-2]
            if ii > 0 and np.abs((log_posterior - old_log_posterior) / old_log_posterior) < stop_cond:
                break
#             self.I = self.I - learning_rate * np.dot(np.linalg.inv(h_log_posterior), g_log_posterior)
            self.I = self.I - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
#             self.I = self.I - learning_rate * np.dot(np.linalg.inv(h_log_likelihood), g_log_posterior)

        log_prior_iterations = np.array(log_prior_iterations)
        log_posterior_iterations = np.array(log_posterior_iterations)

        return log_posterior_iterations, log_prior_iterations