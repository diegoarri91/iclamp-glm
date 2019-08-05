import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
import time

from .glm import GLM
from .iclamp import IClamp
from .kernels import KernelRect
from .masks import shift_mask
from .signals import get_arg, diag_indices
from .spiketrain import SpikeTrain


class GLMFitter:

    def __init__(self, ic, glm=None, neuron=None, filename=None, spike_kwargs=None):

        self.neuron = neuron
        self.filename = filename
        self.ic = ic
        self.mask_spikes = None
        self.n_spikes = None

        if glm is None:
            self.glm = GLM()
        else:
            self.glm = glm

        if spike_kwargs is None:
            self.spike_kwargs = None
        else:
            self.set_mask_spikes(**spike_kwargs)
            
        self.n_subsample = 1
        self.average_stim = False
        
        self.Ih = 0
        self.Y = None
        self.Y_spikes = None

        self.prior = None
        self.prior_pars = None
        self.newton_kwargs = None  # saved
        self.theta0 = None  # saved
        self.theta_iterations = None
        self.log_posterior_iterations = None  # saved
        self.log_prior_iterations = None  # saved
        self.theta = None
        self.logL_norm = None
        self.fit_status = None  # saved

        self.z = None
        self.ks_stats = None  # saved

        self.mask_spikes_model = None
        self.mask_spikes_model_trials = None  # saved

        self.psth_kernel_info = None  # saved
        self.psth_kernel = None   # saved
        self.psth_exp = None
        self.psth_model = None
        self.Md = None  # saved
        self.Ma = None  # saved
        self.psth_Md = None  # saved
        self.psth_Ma = None  # saved

    def save(self, folder, file, neuron_folder, neuron_filename, extras=None, full=False):

        fit_params = {'neuron': neuron_folder, 'filename': neuron_filename, 'ic_class': self.ic.__class__,
                      'glm': self.glm, 'tbins_kappa': self.glm.kappa.tbins, 'tbins_eta': self.glm.eta.tbins,
                      'spike_kwargs': self.spike_kwargs, 'n_spikes': self.n_spikes,
                      'newton_kwargs': self.newton_kwargs, 'theta0': self.theta0,
                      'prior': self.prior, 'prior_pars': self.prior_pars,
                      'theta': self.theta, 'logL_norm': self.logL_norm, 'fit_status': self.fit_status,
                      'ks_stats': self.ks_stats, 'mask_spikes_model_trials': self.mask_spikes_model_trials,
                      'psth_kernel': self.psth_kernel, 'psth_kernel_info': self.psth_kernel_info, 'n_subsample': self.n_subsample, 'Ih': self.Ih,
                      'Md': self.Md, 'Ma': self.Ma, 'psth_Md': self.psth_Md, 'psth_Ma': self.psth_Ma}
        
        if extras is not None:
            fit_params = {**fit_params, **extras}

        if full:
            fit_params['theta_iterations'] = self.theta_iterations
            fit_params['log_posterior_iterations'] = self.log_posterior_iterations
            fit_params['log_prior_iterations'] = self.log_prior_iterations

        with open(folder + file + '.pk', "wb") as pickle_file:
            pickle.dump(fit_params, pickle_file)

    @classmethod
    def load(cls, fit_path, root_folder, only_ic=False, psth=True, set_mask_spikes=True):

        with open(fit_path, "rb") as fit_file:
            dic = pickle.load(fit_file)

        ic = dic['ic_class'].from_file(root_folder + dic['neuron'], dic['filename'])
        
        if not only_ic:
            
            if set_mask_spikes:
                glm_fitter = cls(ic, glm=dic['glm'], spike_kwargs=dic['spike_kwargs'])
            else:
                glm_fitter = cls(ic, glm=dic['glm'])
                glm_fitter.spike_kwargs = dic['spike_kwargs']
            
            for key, val in dic.items():
                if key not in ['ic_class', 'glm', 'spike_kwargs', 'tbins_kappa', 'tbins_eta']:
                    if val is not None:
                        setattr(glm_fitter, key, val)
        
            if glm_fitter.n_subsample > 1:
                glm_fitter = glm_fitter.subsample(glm_fitter.n_subsample)

            if psth:
                # TODO. get shouldnt be necessary
                trials = dic['mask_spikes_model_trials'] if dic['mask_spikes_model_trials'] is not None else 3
                glm_fitter.set_mask_spikes_model(trials=trials)
                glm_fitter.psth(glm_fitter.psth_kernel)
                
        else:
            
            glm_fitter = cls(ic)
        
        return glm_fitter
    
    @classmethod
    def from_IClamp(cls, folder, filename, tf=None):
        ic = IClamp.from_file(folder, filename)
        if tf is not None:
            ic = ic.restrict(t0=0, tf=tf)
        glm_fitter = cls(ic)
        return glm_fitter

    def set_mask_spikes(self, dvdt_thr=9., tl=4., tref=4., thr=-13., time_window=3., tf=None, use_derivative=True):

        self.spike_kwargs = {'dvdt_thr': dvdt_thr, 'tl': tl, 'tref': tref, 'thr': thr, 'time_window': time_window,
                             'use_derivative': use_derivative}
        
        mask_spk = self.ic.mask_spikes(tf=tf, thr=thr, time_window=time_window, use_derivative=use_derivative, dvdt_threshold=dvdt_thr, t_before_spike_peak=tl, tref=tref)

        self.mask_spikes = mask_spk
        self.n_spikes = np.sum(self.mask_spikes)
        
        return self
    
    def set_Ih(self, th):
        argh = get_arg(th, self.ic.dt)
        self.Ih = np.mean(self.ic.stim[:argh])
        return self
    
    def subsample(self, n_sample, average_stim=False):
        # FIRST set_mask_spikes should be called so mask_spikes should be set without subsampling!
        self.n_subsample = n_sample
        if average_stim:
            self.average_stim = True
            tf = n_sample * self.ic.dt
            self.ic.stim = KernelRect([0, tf], [1 / tf]).convolve_continuous(self.ic.t, self.ic.stim)
        self.ic = self.ic.subsample(n_sample=n_sample)
        arg_spikes = np.where(self.mask_spikes)
        arg_spikes = (np.array(np.floor(arg_spikes[0]/n_sample), dtype=int), ) + arg_spikes[1:]
        self.mask_spikes = np.zeros(self.ic.shape, dtype=bool)
        self.mask_spikes[arg_spikes] = True
        
        return self

    def set_mask_spikes_model(self, trials=5):
        self.mask_spikes_model_trials = trials
        _, _, _, self.mask_spikes_model = self.glm.simulate(self.ic.t, np.stack([np.mean(self.ic.stim, 1)] * trials, 1),
                                                            I0=self.Ih)
        return self

    def set_Ymatrix(self):

        n_kappa = self.glm.kappa.nbasis
        Y_kappa = self.glm.kappa.convolve_basis_continuous(self.ic.t, self.ic.stim - self.Ih)

        if self.glm.eta is not None:
            args = np.where(shift_mask(self.mask_spikes, 1, fill_value=False))
            t_spk = (self.ic.t[args[0]], ) + args[1:]
            n_eta = self.glm.eta.nbasis
            Y_eta = self.glm.eta.convolve_basis_discrete(self.ic.t, t_spk, shape=self.ic.shape)
            Y = np.zeros(self.ic.shape + (1 + n_kappa + n_eta,))
            Y[:, :, n_kappa + 1:] = -Y_eta
        else:
            Y = np.zeros(self.ic.shape + (1 + n_kappa, ))

        Y[:, :, 0] = -1.
        Y[:, :, 1:n_kappa + 1] = Y_kappa + np.diff(self.glm.kappa.tbins)[None, None, :] * self.Ih

        self.Y_spikes, self.Y = Y[self.mask_spikes, :], Y[np.ones(self.ic.shape, dtype=bool), :]

        return self.Y_spikes, self.Y

    def set_glm_parameters(self, theta, tbins_kappa, tbins_eta):

        if self.glm is None:
            self.glm = GLM()

        n_kappa = len(tbins_kappa) - 1

        self.glm.r0 = np.exp(-theta[0])
        self.glm.kappa = KernelRect(tbins_kappa, theta[1:n_kappa + 1])
        
        if tbins_eta is not None:
            self.glm.eta = KernelRect(tbins_eta, theta[n_kappa + 1:])
        
        self.theta = theta

        N_spikes = np.sum(self.mask_spikes, 0)
        N_spikes = N_spikes[N_spikes > 0]
        logL_poisson = np.sum(N_spikes * (np.log(N_spikes / len(self.ic.t)) - 1))  # L_poisson = rho0**n_neurons * np.exp(-rho0*T)
        
        if self.Y is None:
            self.set_Ymatrix()

        self.logL_norm = self.logLikelihood(theta, self.Y_spikes, self.Y)
        self.logL_norm = (self.logL_norm - logL_poisson) / np.log(2) / np.sum(N_spikes)
        
        return self

    def t_spikes(self):
        args = np.where(self.mask_spikes)
        t_spk = (self.ic.t[args[0]],) + args[1:]
        return t_spk

    ###################################################################################################
    # FITTING FUNCTIONS
    ###################################################################################################

    def gh_log_likelihood(self, theta, Y_spikes, Y):

        Yspk_theta = np.dot(Y_spikes, theta)
        Y_theta = np.dot(Y, theta)
        exp_Y_theta = np.exp(Y_theta)

        # Log Likelihood
        # I remove the last term from the likelihood so it doesnt have to be computed in the iterations
        # to full likelihood self.logLikelihood should be used
        L = np.sum(Yspk_theta) - self.ic.dt * np.sum(exp_Y_theta)# + np.sum(self.mask_spikes) * np.log(self.ic.dt)

        # Gradient
        G = np.sum(Y_spikes, axis=0) - self.ic.dt * np.matmul(Y.T, exp_Y_theta)

        # Hessian
        H = - self.ic.dt * np.dot(Y.T * exp_Y_theta, Y)

        return L, G, H

    def logLikelihood(self, theta, Y_spikes, Y):

        Yspk_theta = np.dot(Y_spikes, theta)
        Y_theta = np.dot(Y, theta)
        exp_Y_theta = np.exp(Y_theta)

        # Log Likelihood
        logL = np.sum(Yspk_theta) - self.ic.dt * np.sum(exp_Y_theta) + np.sum(self.mask_spikes) * np.log(self.ic.dt)

        return logL

    def gh_log_prior(self, theta, prior, prior_pars):
        
        n_kappa = self.glm.kappa.nbasis

        if prior == 'smooth_2nd_derivative':

            lam = prior_pars[0]

            eta_coefs = theta[n_kappa + 1:]
            log_prior = lam * np.sum((eta_coefs[:-2] + eta_coefs[2:] - 2 * eta_coefs[1:-1]) ** 2)

            g_log_prior = np.zeros(len(eta_coefs))
            g_log_prior[0] = 2 * lam * (eta_coefs[0] - 2 * eta_coefs[1] + eta_coefs[2])
            g_log_prior[1] = 2 * lam * (-2 * eta_coefs[0] + 5 * eta_coefs[1] - 4 * eta_coefs[2] + eta_coefs[3])
            g_log_prior[2:-2] = 2 * lam * (eta_coefs[:-4] - 4 * eta_coefs[1:-3] + 6 * eta_coefs[2:-2]
                                           - 4 * eta_coefs[3:-1] + eta_coefs[4:])
            g_log_prior[-2] = 2 * lam * (eta_coefs[-4] - 4 * eta_coefs[-3] + 5 * eta_coefs[-2] - 2 * eta_coefs[-1])
            g_log_prior[-1] = 2 * lam * (eta_coefs[-3] - 2 * eta_coefs[-2] + eta_coefs[-1])

            h_log_prior = np.zeros((len(eta_coefs), len(eta_coefs)))
            h_log_prior[0, 0], h_log_prior[0, 1], h_log_prior[0, 2]  = 1, -2, 1
            h_log_prior[1, 1], h_log_prior[1, 2], h_log_prior[1, 3]  = 5, -4, 1
            h_log_prior[2:-2, 2:-2][diag_indices(len(eta_coefs) - 4, k=0)] = 6
            h_log_prior[2:-2, 2:-2][diag_indices(len(eta_coefs) - 4, k=1)] = -4
            h_log_prior[2:-2, 2:-2][diag_indices(len(eta_coefs) - 4, k=2)] = 1
            h_log_prior[-2, -2], h_log_prior[-2, -1] = 5, -2
            h_log_prior[-1, -1] = 1

            h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[np.tril_indices_from(h_log_prior, k=-1)]

            h_log_prior = 2 * lam * h_log_prior

            g_log_prior = np.concatenate((np.zeros(n_kappa + 1), g_log_prior))
            _h_log_prior = np.zeros((len(theta), len(theta)))
            _h_log_prior[n_kappa + 1:, n_kappa + 1:] = h_log_prior

        elif prior == 'exponential':

            lam = prior_pars[0]
            if len(prior_pars) > 1:
                mu = prior_pars[1]
            else:
                mu = 1

            eta_coefs = theta[n_kappa + 1:]
            log_prior = lam * np.sum((eta_coefs[1:] - mu  * eta_coefs[:-1]) ** 2)

            g_log_prior = np.zeros(len(eta_coefs))
            g_log_prior[0] = -2 * lam * mu * (eta_coefs[1] - mu * eta_coefs[0])
            g_log_prior[1:-1] = 2 * lam * (-mu * eta_coefs[:-2] + (1 + mu ** 2) * eta_coefs[1:-1] - mu * eta_coefs[2:])
            g_log_prior[-1] = 2 * lam * (eta_coefs[-1] - mu * eta_coefs[-2])

            h_log_prior = np.zeros((len(eta_coefs), len(eta_coefs)))
            h_log_prior[0, 0], h_log_prior[0, 1] = mu ** 2, -mu
            h_log_prior[1:-1, 1:-1][diag_indices(len(eta_coefs) - 2, k=0)] = 1 + mu ** 2
            h_log_prior[1:-1, 1:-1][diag_indices(len(eta_coefs) - 2, k=1)] = -mu
            h_log_prior[-1, -1] = 1

            h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
                np.tril_indices_from(h_log_prior, k=-1)]

            h_log_prior = 2 * lam * h_log_prior

            g_log_prior = np.concatenate((np.zeros(n_kappa + 1), g_log_prior))
            _h_log_prior = np.zeros((len(theta), len(theta)))
            _h_log_prior[n_kappa + 1:, n_kappa + 1:] = h_log_prior
            
        elif prior == 'exponential2':

            lam = prior_pars[0]
            if len(prior_pars) > 1:
                mu = np.exp(-prior_pars[1] * np.diff(self.glm.eta.tbins[:-1]))
            else:
                mu = np.exp(-0 * np.diff(self.glm.eta.tbins[:-1]))

            eta_coefs = theta[n_kappa + 1:]
            log_prior = lam * np.sum((eta_coefs[1:] - mu * eta_coefs[:-1]) ** 2)

            g_log_prior = np.zeros(len(eta_coefs))
            g_log_prior[0] = -2 * lam * mu[0] * (eta_coefs[1] - mu[0] * eta_coefs[0])
            g_log_prior[1:-1] = 2 * lam * (-mu[:-1] * eta_coefs[:-2] + (1 + mu[1:] ** 2) * eta_coefs[1:-1] - mu[1:] * eta_coefs[2:])
            g_log_prior[-1] = 2 * lam * (eta_coefs[-1] - mu[-1] * eta_coefs[-2])

            h_log_prior = np.zeros((len(eta_coefs), len(eta_coefs)))
            h_log_prior[0, 0], h_log_prior[0, 1] = mu[0] ** 2, -mu[0]
            h_log_prior[1:-1, 1:-1][diag_indices(len(eta_coefs) - 2, k=0)] = 1 + mu[1:] ** 2
            h_log_prior[1:-1, 1:-1][diag_indices(len(eta_coefs) - 2, k=1)] = -mu[1:-1]
#             h_log_prior[1:-1, 1:-1][diag_indices(len(eta_coefs) - 2, k=-1)] = -mu[:-1]
            h_log_prior[-1, -1] = 1

            h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
                np.tril_indices_from(h_log_prior, k=-1)]

            h_log_prior = 2 * lam * h_log_prior

            g_log_prior = np.concatenate((np.zeros(n_kappa + 1), g_log_prior))
            _h_log_prior = np.zeros((len(theta), len(theta)))
            _h_log_prior[n_kappa + 1:, n_kappa + 1:] = h_log_prior

        return -log_prior, -g_log_prior, -_h_log_prior

    def fit(self, theta0=None, newton_kwargs=None, verbose=False, tbins_kappa=None, tbins_eta=None, prior=None, prior_pars=None):

        if theta0 is None:
            theta0_kappa = 1e-2 * np.exp(-tbins_kappa[:-1] / 50)
            theta0_eta = 1e0 * np.exp(-tbins_eta[:-1] / 50)
            theta0 = np.concatenate((np.array([-np.log(5e-5)]), theta0_kappa, theta0_eta))

        if newton_kwargs is None:
            newton_kwargs = {}

        max_iterations = newton_kwargs.get('max_iterations', 100)
        stop_cond = newton_kwargs.get('stop_cond', 5e-4)
        learning_rate = newton_kwargs.get('learning_rate', 1e0)
        initial_learning_rate = newton_kwargs.get('initial_learning_rate', learning_rate)
        newton_kwargs = {'max_iterations': max_iterations, 'stop_cond': stop_cond, 'learning_rate': learning_rate,
                         'initial_learning_rate': initial_learning_rate}

        self.newton_kwargs = newton_kwargs

        self.glm.kappa = KernelRect(tbins_kappa)
        
        if tbins_eta is not None:
            self.glm.eta = KernelRect(tbins_eta)
            
        self.set_Ymatrix()

        self.theta0 = theta0
        self.theta_iterations = []

        self.prior, self.prior_pars = prior, prior_pars

        theta = theta0
        self.log_prior_iterations = []
        self.log_posterior_iterations = []

        status = ''
        converged = nan_parameters = False
        n_iterations = 0
        log_prior = np.nan

        if verbose:
            print('Starting gradient ascent... \n')

        t0 = time.time()
        for ii in range(max_iterations):

            learning_rate = newton_kwargs['learning_rate']

            if ii <= 10:
                learning_rate = newton_kwargs['initial_learning_rate']

            if verbose and ii % 50 == 0:
                print('\r', 'Iteration {} of {}'.format(ii, max_iterations), 'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), end='')

            if prior is not None:
                log_prior, g_log_prior, h_log_prior = self.gh_log_prior(theta, self.prior, self.prior_pars)

            log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(theta, self.Y_spikes, self.Y)

            if prior is not None:
                log_posterior = log_likelihood + log_prior
                g_log_posterior = g_log_likelihood + g_log_prior
                h_log_posterior = h_log_likelihood + h_log_prior
            else:
                log_posterior = log_likelihood
                g_log_posterior = g_log_likelihood
                h_log_posterior = h_log_likelihood

            self.log_prior_iterations += [log_prior]
            self.log_posterior_iterations += [log_posterior]
            self.theta_iterations += [theta]

            old_log_posterior = 0 if len(self.log_posterior_iterations) < 2 else self.log_posterior_iterations[-2]
            if ii > 0 and normal_iteration and np.abs((log_posterior - old_log_posterior) / old_log_posterior) < stop_cond:
                status += "\n Converged after {} iterations!. ".format(ii + 1)
                converged = True
                nan_parameters = False
                n_iterations = ii + 1
                break
            elif np.any(np.isnan(theta)):
                status += "\n There are nan parameters. "
                nan_parameters = True
                converged = False
                n_iterations = ii + 1
                break
            elif ii == max_iterations - 1:
                status += '\n Not converged after {} iterations. '.format(max_iterations)
                converged = False
                nan_parameters = False
                n_iterations = ii + 1

            if len(self.log_posterior_iterations) == 1 or self.log_posterior_iterations[-1] - self.log_posterior_iterations[-2] > 0:
                normal_iteration = True
                old_theta = np.copy(theta)
                old_g_log_posterior = np.copy(g_log_posterior)
                old_h_log_posterior = np.copy(h_log_posterior)
                #theta = theta - learning_rate * np.dot(np.linalg.inv(h_log_posterior), g_log_posterior)
                theta = theta - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
            else:
                normal_iteration = False
                self.log_prior_iterations = self.log_prior_iterations[:-1] + [self.log_prior_iterations[-2]]
                self.log_posterior_iterations = self.log_posterior_iterations[:-1] + [self.log_posterior_iterations[-2]]
                self.theta_iterations = self.theta_iterations[:-1] + [self.theta_iterations[-2]]
                theta = np.copy(old_theta)
                g_log_posterior = np.copy(old_g_log_posterior)
                h_log_posterior = np.copy(old_h_log_posterior)
                learning_rate = learning_rate / 2
                error = True
                while error:
                    try:
                        error = False
                        #theta = theta - learning_rate * np.dot(np.linalg.inv(h_log_posterior), g_log_posterior)
                        theta = theta - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
                        if prior is not None:
                            log_prior, g_log_prior, h_log_prior = self.gh_log_prior(theta, prior, prior_pars)
                        log_likelihood = self.logLikelihood(theta, self.Y_spikes, self.Y)
                        if prior is not None:
                            log_posterior = log_likelihood + log_prior
                        else:
                            log_posterior = log_likelihood
                        if np.isinf(log_posterior):
                            raise(ValueError)
                    except(np.linalg.LinAlgError, ValueError) as e:
                        learning_rate = learning_rate / 2

        fitting_time = (time.time() - t0) / 60.

        status += 'Elapsed time: {} minutes \n'.format(fitting_time)

        if nan_parameters:
            logL_monotonous = None
        elif np.any(np.diff(self.log_posterior_iterations) < 0.):
            status += "Log likelihood is not monotonically increasing \n"
            logL_monotonous = False
        else:
            status += "Log likelihood is monotonous \n"
            logL_monotonous = True

        if verbose:
            print('\n', status)

        self.fit_status = {'n_iterations': n_iterations, 'converged': converged, 'nan_parameters': nan_parameters,
                           'fitting_time': fitting_time, 'logL_monotonous': logL_monotonous, 'status': status}

        self.theta_iterations = np.stack(self.theta_iterations, 1)
        self.log_posterior_iterations = np.array(self.log_posterior_iterations)
        self.log_prior_iterations= np.array(self.log_prior_iterations)
        
        theta = self.theta_iterations[:, np.argmax(self.log_posterior_iterations)]
        
        self.set_glm_parameters(theta, tbins_kappa, tbins_eta)

        return self

    def plot_filters(self, axs=None):

        if axs is None:
            fig, (ax_kappa, ax_eta) = plt.subplots(figsize=(12, 6.5), ncols=2)
            ax_kappa.set_xlabel('time'); ax_kappa.set_ylabel('kappa')
            ax_eta.set_xlabel('time'); ax_eta.set_ylabel('eta')
            ax_kappa.spines['right'].set_visible(False)
            ax_kappa.spines['top'].set_visible(False)
            ax_eta.spines['right'].set_visible(False)
            ax_eta.spines['top'].set_visible(False)
        else:
            ax_kappa = axs[0]
            ax_eta = axs[1]

        t_kappa = np.arange(0., self.glm.kappa.tbins[-1], .1)
        ax_kappa = self.glm.kappa.plot(t_kappa, ax=ax_kappa)
        t_eta = np.arange(0., self.glm.eta.tbins[-1], .1)
        ax_eta = self.glm.eta.plot(t_eta, ax=ax_eta)

        return fig, (ax_kappa, ax_eta)

    def plot_fit(self):

        fig = plt.figure(figsize=(10, 7.5))
        fig.tight_layout()
        axr = plt.subplot2grid((5, 3), (0, 0), rowspan=2, colspan=2)
        ax_log_posterior = plt.subplot2grid((5, 3), (0, 2), rowspan=1)
        ax_log_prior= plt.subplot2grid((5, 3), (1, 2), rowspan=1)
        ax_time_rescale_transform = [plt.subplot2grid((5, 2), (2, col)) for col in range(2)]
        ax_kappa = [plt.subplot2grid((5, 3), (3, col)) for col in range(3)]

        if self.filename is not None:
            axr.set_title('neuron: ' + self.neuron + ' file: ' + self.filename)

        if self.psth_exp is not None:
            self.plot_psth(ax=axr)

        if self.log_posterior_iterations is not None:
            log_logL = -np.log(-self.log_posterior_iterations)
            ax_log_posterior.plot(range(1, len(self.log_posterior_iterations) + 1), log_logL, 'C0-')
            log_logL = -np.log(-self.log_prior_iterations)
            ax_log_prior.plot(range(1, len(self.log_prior_iterations) + 1), log_logL, 'C1-')
            logL_norm = np.round(self.logL_norm, 2)
            ax_time_rescale_transform[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
                                              transform=ax_time_rescale_transform[0].transAxes)

        self.plot_time_rescale_transform(ax=ax_time_rescale_transform)

        t_kappa = np.arange(0., self.glm.kappa.tbins[-1], .1)
        self.glm.kappa.plot_lin_log(t_kappa, axs=ax_kappa)
        
        if self.glm.eta is not None:
            ax_eta = [plt.subplot2grid((5, 3), (4, col)) for col in range(3)]
            t_eta = np.arange(0., self.glm.eta.tbins[-1], .1)
            self.glm.eta.plot_lin_log(t_eta, axs=ax_eta)
            return fig, (axr, ax_log_posterior, ax_time_rescale_transform, ax_kappa, ax_eta)
        
        return fig, (axr, ax_log_posterior, ax_time_rescale_transform, ax_kappa)
    
    def save_pdf(self, folder, pdf_name, off=True):
        
        if off:
            plt.ioff()
            
        fig, axs = self.plot_fit()
        fig.savefig(folder + pdf_name + '.pdf')

    ###################################################################################################
    # GOODNESS OF FIT
    ###################################################################################################

    def psth(self, psth_kernel=None, biased=True, psth_kernel_info=None):

        st_exp = SpikeTrain(self.ic.t, self.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        self.psth_kernel = psth_kernel
        self.psth_kernel_info = psth_kernel_info
        self.psth_exp = st_exp.get_PSTH(psth_kernel)
        self.psth_model = st_model.get_PSTH(psth_kernel)
        self.psth_Ma = st_exp.Ma(st_model, psth_kernel, psth_kernel, biased=biased)
        self.psth_Md = st_exp.Md(st_model, psth_kernel, psth_kernel, biased=biased)

        return self

    def set_Md(self, kernel1, kernel2, biased=True):

        st_exp = SpikeTrain(self.ic.t, self.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        self.Md = st_exp.Md(st_model, kernel1, kernel2, biased=biased)

        return self.Md

    def Ma(self, kernel1, kernel2, biased=True):

        st_exp = SpikeTrain(self.ic.t, self.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        self.Ma = st_exp.Ma(st_model, kernel1, kernel2, biased=biased)

        return self.Ma

    def plot_psth(self, ax=None, lw=.6):

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), ncols=1)

        ax.plot(self.ic.t, self.psth_exp, lw=lw)
        ax.plot(self.ic.t, self.psth_model, lw=lw)
        ax.set_ylim(-.001, np.max(np.concatenate((self.psth_model, self.psth_exp), axis=1)) * 1.15)

        if self.Md is not None:
            Md = np.round(self.Md, 3)
            ax.text(.15, 0.92, 'Md=' + str(Md), transform=ax.transAxes)
        if self.Ma is not None:
            Ma = np.round(self.Ma, 3)
            ax.text(.75, 0.92, 'Ma=' + str(Ma), transform=ax.transAxes)
        if self.Md is None and self.Ma is None:
            Md, Ma = np.round(self.psth_Md, 3), np.round(self.psth_Ma, 3)
            ax.text(.15, 0.92, 'Md=' + str(Md), transform=ax.transAxes)
            ax.text(.75, 0.92, 'Ma=' + str(Ma), transform=ax.transAxes)

    def time_rescale_transform(self):

        r, _, _ = self.glm.simulate_subthr(self.ic.t, self.ic.stim, self.mask_spikes, I0=self.Ih)
        integral_r = np.cumsum(r * self.ic.dt, axis=0)

        z = []

        for sw in range(self.ic.nsweeps):
            Lambda = integral_r[self.mask_spikes[:, sw], sw]  # I think there is no need for shifting the mask
            z += [1. - np.exp(-(Lambda[1:] - Lambda[:-1]))]

        self.z = z
        self.ks_stats = kstest(np.concatenate(self.z), 'uniform', args=(0, 1))

        return self

    def plot_time_rescale_transform(self, ax=None):

        if self.z is None or self.ks_stats is None:
            return ax

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), ncols=2)

        bins = np.arange(0, 1.01, .01)

        z = np.concatenate(self.z)
        stat, p = np.round(self.ks_stats[0], 4), np.round(self.ks_stats[1], 3)
        ax[0].text(.05, .7, 'stat=' + str(stat) + '\np=' + str(p), transform=ax[0].transAxes)

        values, bins = np.histogram(z, bins=bins)
        cum = np.append(0., np.cumsum(values) / np.sum(values))
        ax[0].plot(bins, cum, '-')
        ax[0].plot([0, 1], [0, 1], 'k--')

        for sw in range(self.ic.nsweeps):
            ax[1].plot(self.z[sw][:-1], self.z[sw][1:], 'C0.')

        return ax

    def plot_raster(self):

        fig, axs = plt.subplots(figsize=(12, 7), nrows=3, sharex=True)
        fig.subplots_adjust(hspace=0)

        st_exp = SpikeTrain(self.ic.t, self.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        st_exp.plot(ax=axs[0], color='C0')
        st_model.plot(ax=axs[1], color='C1')

        self.plot_psth(axs[2], lw=1.5)

        axs[0].xaxis.set_visible(False)
        axs[0].set_yticks([])
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].set_ylabel('data')
        axs[1].xaxis.set_visible(False)
        axs[1].set_yticks([])
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['bottom'].set_visible(False)
        axs[1].set_ylabel('GLM')
        axs[2].set_xlabel('time')

        return fig, axs

