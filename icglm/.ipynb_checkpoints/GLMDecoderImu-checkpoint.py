import time
#import pickle
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import solveh_banded

from GLM import GLM
from KernelVals import KernelVals

from fun_signals import get_dt, get_arg


class GLMDecoder:

    def __init__(self, glms=None, t=None, mask_spk=None, tau=None, Ih=None, Imu=None, Isd=None, I_true=None,
                 n_subsample=None, neurons=None, files=None, trials=None, fit_files=None):

        self.neurons = neurons  # saved
        self.files = files  # saved
        self.trials = trials  # saved
        self.fit_files = fit_files  # saved
        self.glms = glms  # saved
        self.t = t
        self.mask_spk = mask_spk
        self.I_true = I_true
        self.dt = get_dt(t)
        self.n_subsample = n_subsample
        self.n_trials = mask_spk[0].shape[1]
        self.n_neurons = len(self.glms)

        self.I = None
        self.I_dec = None
        self.tau = tau  # saved
        self.Ih = Ih  # saved
        self.Imu = Imu  # saved
        self.Isd = Isd  # saved

        self.dec_status = None
        self.optimization_kwargs = None

        self.log_prior_iterations = None
        self.log_posterior_iterations = None

        self.r2 = None
        self.corr = None

    def save(self, path):

        dec_params = {'neurons': self.neurons, 'files': self.files, 'glms': self.glms, 'fit_files': self.fit_files,
                      'trials': self.trials, 'I_true': self.I_true, 'I_dec': self.I_dec,
                      'Ih': self.Ih, 'Imu': self.Imu, 'Isd': self.Isd, 'tau': self.tau,
                      'log_posterior_iterations': self.log_posterior_iterations,
                      'n_subsample': self.n_subsample, 'dt': self.dt, 'arg0': get_arg(self.t, self.dt),
                      'shape': self.mask_spk[0].shape, 'arg_spikes': [np.where(self.mask_spk[n]) for n in range(self.n_neurons)],
                      'log_prior_iterations': self.log_prior_iterations, 'optimization_kwargs': self.optimization_kwargs,
                      'dec_status': self.dec_status, 'r2': self.r2, 'corr': self.corr}

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
    
    def banded_ou_h_log_prior(self):
        
        if False:
            lam = 1. - self.dt / self.tau
            _sigma2 = 2. * self.dt / self.tau * 1 ** 2.
        else:
            lam = np.exp(- self.dt / self.tau)
            _sigma2 = (1 - np.exp(-2 * self.dt / self.tau)) * 1 ** 2.
        
        banded_h_log_prior = np.zeros((2, len(self.t)))
        banded_h_log_prior[0, 0] = lam ** 2. + _sigma2 / 1. ** 2
        banded_h_log_prior[0, -1] = 1.
        banded_h_log_prior[0, 1:-1] = 1. + lam ** 2.
        banded_h_log_prior[1, :] = -lam
        
        banded_h_log_prior = -banded_h_log_prior/_sigma2
        
        return banded_h_log_prior
    
    def banded_gh_log_likelihood(self, sum_convolution_kappa_t_spikes, K, cum_kappa, max_band):
        
        log_likelihood = 0
        g_log_likelihood = np.zeros((len(self.t) + self.n_neurons))
        g_log_likelihood[self.n_neurons:] = self.dt * sum_convolution_kappa_t_spikes
        h_log_likelihood = np.zeros((max_band, len(self.t) + self.n_neurons))
        
        for n in range(self.n_neurons):
            _I = self.I * self.Isd[n] + self.Imu[n]
            _r, _u = self.glms[n].simulate_subthr(self.t, _I, self.mask_spk[n], full=True, iterating=True,
                                                 I0=self.Ih[n])
            log_likelihood += np.sum(_u[self.mask_spk[n]]) - self.dt * np.sum(_r)

            g_log_likelihood[n] = np.sum(cum_kappa[n][self.mask_spk[n]]) - self.dt * np.sum(_r * cum_kappa[n])
            g_log_likelihood[self.n_neurons:] += -self.dt * self.Isd[n] * self.glms[n].kappa.correlate_continuous(self.t, np.sum(_r, 1),
                                                                                                iterating=True)
            t_support = self.glms[n].kappa.support
            arg_support = int(t_support[1] / self.dt)
            h_log_likelihood[0, n] = -self.dt * np.sum(_r * cum_kappa[n]**2)
            h_log_likelihood[self.n_neurons - n:, n] = -self.dt * self.Isd[n] * self.glms[n].kappa.correlate_continuous(self.t, np.sum(_r * cum_kappa[n], 1), iterating=True)[:max_band + n - self.n_neurons]
            for v in range(arg_support):
                h_log_likelihood[v, self.n_neurons:] += -self.dt ** 2 * self.Isd[n] * K[n][v].correlate_continuous(self.t, np.sum(_r, 1))

        return log_likelihood, g_log_likelihood, h_log_likelihood

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
    
    def cum_kappa(self):
        
        cum_kappa = []

        for n in range(self.n_neurons):
            cum_kappa.append(self.glms[n].kappa.convolve_continuous(self.t, np.ones((len(self.t), self.n_trials))))

        return cum_kappa
        

    def estimate_MAP_I(self, I0, optimization_kwargs=None, verbose=False):

        if optimization_kwargs is None:
            optimization_kwargs = {}
        
        max_iterations = optimization_kwargs.get('max_iterations', int(5e3))
        stop_cond = optimization_kwargs.get('stop_cond', 1e-7)
        learning_rate = optimization_kwargs.get('learning_rate', 1e-1)
        initial_learning_rate = optimization_kwargs.get('initial_learning_rate', learning_rate * 1e-1)
        optimization_kwargs = {'max_iterations': max_iterations, 'stop_cond': stop_cond, 'learning_rate': learning_rate,
                               'initial_learning_rate': initial_learning_rate}

        self.optimization_kwargs = optimization_kwargs
        
        if False:
            lam = 1. - self.dt / self.tau
            _mu = 0 * self.dt / self.tau
            _sigma2 = 2. * self.dt / self.tau * 1 ** 2.
        else:
            lam = np.exp(- self.dt / self.tau)
            _mu = 0 * (1 - np.exp(-self.dt / self.tau))
            _sigma2 = (1 - np.exp(-2 * self.dt / self.tau)) * 1 ** 2.

        max_band = [self.glms[n].kappa.support[1] for n in range(self.n_neurons)]
        max_band = int(np.max(max_band)/self.dt)
        
        g_log_prior = np.zeros(len(self.t))
        banded_h_log_prior = self.banded_ou_h_log_prior()

        for n in range(self.n_neurons):
            self.glms[n].kappa.set_values(self.dt, 1)
            
        sum_convolution_kappa_t_spikes = np.sum(self.convolution_kappa_t_spikes(), axis=(1, 2))
        K = self.build_K()
        
        cum_kappa = self.cum_kappa()

        log_prior_iterations = []
        log_posterior_iterations = []

        self.I = I0

        status = ''
        converged = nan_parameters = False

        if verbose:
            print('Starting decoding... \n')

        t0 = time.time()
        for ii in range(max_iterations):

            learning_rate = optimization_kwargs['learning_rate']

            if ii <= 10:
                learning_rate = optimization_kwargs['initial_learning_rate']

            if verbose and ii % 20 == 0:
                print('\r', 'Iteration {} of {}'.format(ii, max_iterations),
                      'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), end='')

            # prior
            log_prior = -1. / (2. * _sigma2) * np.sum((self.I[1:] - lam * self.I[:-1] - _mu) ** 2.) + \
                        -1. / (2. * 1**2.) * (self.I[0] - 0) ** 2.
            g_log_prior[0] = lam / _sigma2 * (self.I[1] - _mu - lam * self.I[0]) - 1. / 1**2. * (self.I[0] - 0)
            g_log_prior[1:-1] = 1. / _sigma2 * ((1 - lam) * _mu - (1 + lam**2.) * self.I[1:-1] + lam * (self.I[2:] + self.I[:-2]) )
            g_log_prior[-1] = 1. / _sigma2 * (_mu - self.I[-1] + lam * self.I[-2] )

            # likelihood
            log_likelihood, g_log_likelihood, banded_h_log_likelihood = self.banded_gh_log_likelihood(sum_convolution_kappa_t_spikes, K, cum_kappa, max_band)

            # posterior
            log_posterior = log_likelihood + log_prior
            g_log_posterior = g_log_likelihood 
            g_log_posterior[self.n_neurons:] += g_log_prior

            banded_h_log_posterior = -banded_h_log_likelihood
            banded_h_log_posterior[:2, self.n_neurons:] += - banded_h_log_prior

            log_prior_iterations += [log_prior]
            log_posterior_iterations += [log_posterior]

            old_log_posterior = 0 if len(log_posterior_iterations) < 2 else log_posterior_iterations[-2]
            if ii > 0 and np.abs((log_posterior - old_log_posterior) / old_log_posterior) < stop_cond:
                status += "\n Converged after {} iterations!. ".format(ii + 1)
                converged = True
                break
            if np.any(np.isnan(self.I)):
                status += "\n There are nan stimulus values. "
                nan_parameters = True
                converged = False
                break

            self.aux = banded_h_log_posterior
            print(ii)
            aux = solveh_banded(banded_h_log_posterior, g_log_posterior, lower=True)
            for n in range(self.n_neurons):
                self.Imu[n] = self.Imu[n] + learning_rate * aux[n]
            self.I = self.I + learning_rate * aux[self.n_neurons:]
            #self.I = self.I + learning_rate * g_log_posterior

        decoding_time = np.round((time.time() - t0) / 60., 2)
        status += 'Elapsed time: {} minutes \n'.format(decoding_time)

        self.log_prior_iterations = np.array(log_prior_iterations)
        self.log_posterior_iterations = np.array(log_posterior_iterations)
        self.I_dec = np.copy(self.I)

        if np.any(np.diff(self.log_posterior_iterations) < 0):
            status += "Log posterior is not monotonically increasing \n"
            log_posterior_monotonous = False
        else:
            status += "Log posterior is monotonous \n"
            log_posterior_monotonous = True

        self.dec_status = {'n_iterations': ii + 1, 'converged': converged, 'nan_stimulus': nan_parameters,
                           'decoding_time': decoding_time, 'log_posterior_monotonous': log_posterior_monotonous}

        return self

    def plot_decoding(self):
        
        fig = plt.figure(figsize=(10, 7.5))
        fig.tight_layout()
        axI = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        axr = plt.subplot2grid((3, 3), (1, 0), colspan=3, sharex=axI)
        ax_log_posterior = [plt.subplot2grid((3, 3), (2, col)) for col in range(3)]

        if self.files is not None:
            axI.set_title('neuron: ' + ', '.join(self.neurons) + ' file: ' + ', '.join(self.files))
        
        axI.plot(self.t, self.I_true, 'C0')
        axI.plot(self.t, self.I_dec, 'C1')

        for n in range(self.n_neurons):
            _I = self.I_dec * self.Isd[n] + self.Imu[n]
            r, v = self.glms[n].simulate_subthr(self.t, _I, self.mask_spk[n], full=True, iterating=True,
                                               I0=self.Ih[n])
            #axr.plot(self.t, r)
            t_spk = np.stack([self.t] * self.mask_spk[n].shape[1], 1)[self.mask_spk[n]]
            _r_spk = r[self.mask_spk[n]]
            n_spikes = np.sum(self.mask_spk[n])
            if n_spikes > 1000:
                samp = int(n_spikes / 1000)
                t_spk = t_spk[::samp]
                _r_spk = _r_spk[::samp]
            axr.plot(t_spk,_r_spk , '.')
        
        if self.log_posterior_iterations is not None:
            # log_log_posterior = -np.log(-self.log_posterior_iterations)
            iterations = np.arange(1, len(self.log_posterior_iterations) + 1, 1)
            ax_log_posterior[0].plot(iterations , self.log_posterior_iterations, 'C0-')
            ax_log_posterior[1].plot(iterations , self.log_posterior_iterations - self.log_prior_iterations, 'C0-')
            ax_log_posterior[2].plot(iterations , self.log_prior_iterations, 'C0-')
            
            #logL_norm = np.round(self.logL_norm, 2)
            #ax_log_posterior[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
                                              #transform=ax_time_rescale_transform[0].transAxes)

        return fig, (axI, axr, ax_log_posterior)
    
    def save_pdf(self, folder, pdf_name, off=True):
        
        if off:
            plt.ioff()
            
        fig, axs = self.plot_decoding()
        fig.savefig(folder + pdf_name + '.pdf')
        plt.close(fig)
