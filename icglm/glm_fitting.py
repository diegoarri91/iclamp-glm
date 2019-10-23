import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
import time

from models.glm import GLM
from .iclamp import IClamp
from .kernels import KernelFun, KernelRect
from .masks import shift_mask
from .signals import get_arg, diag_indices
from .spiketrain import SpikeTrain


class GLMFitter:

    def __init__(self, ic, glm=None, neuron=None, filename=None, spike_kwargs=None):

        self.neuron = neuron
        self.filename = filename
        self.ic = ic.new(ic.t, ic.data, ic.stim)
        self.mask_spikes = None
        self.n_spikes = None
        
        self.t0 = None
        self.tf = None
        self.th = None

        if glm is None:
            self.glm = GLM()
        else:
            self.glm = glm

        if spike_kwargs is None:
            self.spike_kwargs = None
        else:
            self.set_mask_spikes(**spike_kwargs)
            
        self.dt_subsample = None
        self.average_stim = False
        
        self.Ih = None
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

    def set_mask_spikes(self, dvdt_thr=9., tl=4., tref=4., thr=-13., time_window=3., tf=None, use_derivative=True):

        self.spike_kwargs = {'dvdt_thr': dvdt_thr, 'tl': tl, 'tref': tref, 'thr': thr, 'time_window': time_window,
                             'use_derivative': use_derivative}
        
        self.ic = self.ic.set_mask_spikes(tf=tf, thr=thr, time_window=time_window, use_derivative=use_derivative, dvdt_threshold=dvdt_thr, t_before_spike_peak=tl, tref=tref)

#         self.ic.mask_spikes = mask_spk
        self.n_spikes = np.sum(self.ic.mask_spikes)
        
        return self
    
    def set_Ih(self, th):
        self.th = th
        argh = get_arg(th, self.ic.dt)
        self.Ih = np.mean(self.ic.stim[:argh])
        return self
    
    def subtract_Ih(self, th):
        self.th = th if th is not None else 1000.1
        self.Ih = 0
        self.ic = self.ic.subtract_Ih(th=self.th)
        return self
    
    def restrict(self, t0=None, tf=None, reset_time=True):
        self.t0, self.tf = t0, tf
        self.ic = self.ic.restrict(t0=t0, tf=tf, reset_time=reset_time)
        return self
    
    def subsample(self, dt_sample, average_stim=False):
        self.dt_subsample = dt_sample
        self.average_stim = average_stim
        self.ic = self.ic.subsample(dt_sample, average_stim=average_stim)
        return self

    def set_mask_spikes_model(self, trials=5):
        self.mask_spikes_model_trials = trials
        _, _, _, self.mask_spikes_model = self.glm.simulate(self.ic.t, np.stack([np.mean(self.ic.stim, 1)] * trials, 1),
                                                            Ih=self.Ih)
        return self

    def set_Ymatrix(self):

        n_kappa = self.glm.kappa.nbasis
        Y_kappa = self.glm.kappa.convolve_basis_continuous(self.ic.t, self.ic.stim - self.Ih)

        if self.glm.eta is not None:
            args = np.where(shift_mask(self.ic.mask_spikes, 1, fill_value=False))
            t_spk = (self.ic.t[args[0]], ) + args[1:]
            n_eta = self.glm.eta.nbasis
            Y_eta = self.glm.eta.convolve_basis_discrete(self.ic.t, t_spk, shape=self.ic.shape)
            Y = np.zeros(self.ic.shape + (1 + n_kappa + n_eta,))
            Y[:, :, n_kappa + 1:] = -Y_eta
        else:
            Y = np.zeros(self.ic.shape + (1 + n_kappa, ))

        Y[:, :, 0] = -1.
        Y[:, :, 1:n_kappa + 1] = Y_kappa + np.diff(self.glm.kappa.tbins)[None, None, :] * self.Ih

        self.Y_spikes, self.Y = Y[self.ic.mask_spikes, :], Y[np.ones(self.ic.shape, dtype=bool), :]

        return self.Y_spikes, self.Y

    def set_glm_parameters(self, theta, tbins_kappa, tbins_eta):

        if self.glm is None:
            self.glm = GLM()

        n_kappa = len(tbins_kappa) - 1

        self.glm.u0 = theta[0]
        self.glm.kappa = KernelRect(tbins_kappa, theta[1:n_kappa + 1])
        
        if tbins_eta is not None:
            self.glm.eta = KernelRect(tbins_eta, theta[n_kappa + 1:])
        
        self.theta = theta

        N_spikes = np.sum(self.ic.mask_spikes, 0)
        N_spikes = N_spikes[N_spikes > 0]
        logL_poisson = np.sum(N_spikes * (np.log(N_spikes / len(self.ic.t)) - 1))  # L_poisson = rho0**n_neurons * np.exp(-rho0*T)
        
        if self.Y is None:
            self.set_Ymatrix()

        self.logL_norm = self.logLikelihood(theta, self.Y_spikes, self.Y)
        self.logL_norm = (self.logL_norm - logL_poisson) / np.log(2) / np.sum(N_spikes)
        
        return self

    ###################################################################################################
    # FITTING FUNCTIONS
    ###################################################################################################

    def logLikelihood(self, theta, Y_spikes, Y):

        Yspk_theta = np.dot(Y_spikes, theta)
        Y_theta = np.dot(Y, theta)
        exp_Y_theta = np.exp(Y_theta)

        # Log Likelihood
        logL = np.sum(Yspk_theta) - self.ic.dt * np.sum(exp_Y_theta) + np.sum(self.ic.mask_spikes) * np.log(self.ic.dt)

        return logL

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

        self.plot_time_rescale_transform(ax=ax_time_rescale_transform)
        logL_norm = np.round(self.logL_norm, 2)
        ax_time_rescale_transform[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
                                              transform=ax_time_rescale_transform[0].transAxes)

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


    def plot_ic(self, axv=None, axstim=None, spikes=False, **kwargs):
        return self.ic.plot(axv=axv, axstim=axstim, spikes=spikes, **kwargs)

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

    def plot_raster(self):

        fig, axs = plt.subplots(figsize=(12, 7), nrows=3, sharex=True)
        fig.subplots_adjust(hspace=0)

        st_exp = SpikeTrain(self.ic.t, self.ic.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        st_exp.plot(ax=axs[0], color='C0')
        st_model.plot(ax=axs[1], color='C1')

        self.plot_psth(axs[2], lw=1.5)

        axs[0].xaxis.set_visible(False)
        axs[0].set_yticks([])
        axs[0].spines['left'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].spines['top'].set_visible(False)
        axs[0].set_ylabel('data')
        axs[1].xaxis.set_visible(False)
        axs[1].set_yticks([])
        axs[1].spines['left'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['bottom'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_ylabel('GLM')
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].set_xlabel('time')

        return fig, axs

