import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest

from models.srm import SRM
from .iclamp import IClamp
from .kernels.base import KernelFun
from .spiketrain import SpikeTrain


class Fitter:

    def __init__(self, ic=None, model=None, neuron=None, filename=None, spike_kwargs=None):

        self.neuron = neuron
        self.filename = filename
        self.ic = ic.new(ic.t, ic.data, ic.stim) if ic else None
        self.mask_spikes = None
        self.n_spikes = None
        
        self.t0 = None
        self.tf = None
        self.th = None

        self.model = model

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
        self.log_likelihood_normed = None
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
                      'model': self.model, 'spike_kwargs': self.spike_kwargs, 'subthreshold_kwargs': self.subthreshold_kwargs,
                      'n_spikes': self.n_spikes, 'newton_kwargs': self.newton_kwargs, 'theta0': self.theta0, 'theta': self.theta,
                      'logL_norm': self.log_likelihood_normed, 'fit_status': self.model.fit_status,'ks_stats': self.ks_stats,
                      'mask_spikes_model_trials': self.mask_spikes_model_trials, 'psth_kernel': self.psth_kernel,
                      'psth_kernel_info': self.psth_kernel_info, 'dt_subsample': self.dt_subsample,
                      'average_stim': self.average_stim, 'Ih': self.Ih, 't0': self.t0, 'tf': self.tf, 'th': self.th,
                      'Md': self.Md, 'Ma': self.Ma, 'psth_Md': self.psth_Md, 'psth_Ma': self.psth_Ma}
        
        if extras is not None:
            fit_params = {**fit_params, **extras}

        if full:
            fit_params['theta_iterations'] = self.model.theta_iterations
            fit_params['log_posterior_iterations'] = self.model.log_posterior_iterations
            fit_params['log_prior_iterations'] = self.model.log_prior_iterations

        with open(folder + file + '.pk', "wb") as pickle_file:
            pickle.dump(fit_params, pickle_file)

    @classmethod
    def load(cls, fit_path, root_folder, only_ic=False, psth=True, set_mask_spikes=True):

        with open(fit_path, "rb") as fit_file:
            dic = pickle.load(fit_file)

        ic = dic['ic_class'].load_from_abf(root_folder + dic['neuron']+ '/' + dic['filename'] + '.abf')
        
        if not only_ic:
            
            glm_fitter = cls(ic, model=dic['model'])
            
            for key, val in dic.items():
                if key not in ['ic_class', 'glm', 'spike_kwargs', 'tbins_kappa', 'tbins_eta']:
                    if val is not None:
                        setattr(glm_fitter, key, val)
            glm_fitter = glm_fitter.subtract_Ih(th=glm_fitter.th)
            
            if glm_fitter.t0 is not None or glm_fitter.tf is not None:
                glm_fitter = glm_fitter.restrict(t0=glm_fitter.t0, tf=glm_fitter.tf, reset_time=True)
            
            if set_mask_spikes:
                glm_fitter = glm_fitter.set_mask_spikes(**dic['spike_kwargs'])
                if glm_fitter.dt_subsample > ic.dt:
                    glm_fitter = glm_fitter.subsample(glm_fitter.dt_subsample, average_stim=glm_fitter.average_stim)
            else:
                glm_fitter.spike_kwargs = dic['spike_kwargs']

            if psth:
                # TODO. get shouldnt be necessary
                trials = dic['mask_spikes_model_trials'] if dic['mask_spikes_model_trials'] is not None else 3
                glm_fitter.set_mask_spikes_model(trials=trials)
                psth_kernel = KernelFun.gaussian_delta(delta=200)
                glm_fitter.psth(psth_kernel)
                
        else:
            
            glm_fitter = cls(ic)
        
        return glm_fitter
    
    @classmethod
    def load_model(cls, fit_path):

        with open(fit_path, "rb") as fit_file:
            dic = pickle.load(fit_file)
            
        model = dic['model']
        
        return model

    @property
    def t(self):
        return self.ic.t

    @property
    def data(self):
        return self.ic.data

    @property
    def stim(self):
        return self.ic.stim

    def load_iclamp_file(self, path):
        self.path = path
        if path[-4:] == '.abf':
            self.ic = IClamp.load_from_abf(path=path)
        return self

    def set_mask_spikes(self, dvdt_thr=9., t_before_spike_peak=4., tref=4., thr=-13., time_window=3., tf=None, use_derivative=True):
        self.spike_kwargs = {'dvdt_thr': dvdt_thr, 't_before_spike_peak': t_before_spike_peak, 'tref': tref, 'thr': thr, 'time_window': time_window,
                             'use_derivative': use_derivative}
        self.ic = self.ic.set_mask_spikes(tf=tf, thr=thr, time_window=time_window, use_derivative=use_derivative, dvdt_threshold=dvdt_thr, t_before_spike_peak=t_before_spike_peak, tref=tref)
        self.n_spikes = np.sum(self.ic.mask_spikes)
        return self

    def set_mask_subthreshold(self, t_before_spike=3, t_after_spike=10, t0=None, tf=None):
        self.subthreshold_kwargs = {'t_before_spike': t_before_spike, 't_after_spike': t_after_spike, 't0': t0, 'tf': tf}
        self.mask_subthreshold = self.ic.get_mask_away_from_spikes(t_before_spike, t_after_spike, t0=t0, tf=tf)
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

    def sweeps(self, sweeps):
        self.sweeps = sweeps
        self.ic = self.ic.sweeps(sweeps)
        return self
    
    def subsample(self, dt_sample, average_stim=False):
        self.dt_subsample = dt_sample
        self.average_stim = average_stim
        self.ic = self.ic.subsample(dt_sample, average_stim=average_stim)
        return self

    def set_mask_spikes_model(self, trials=5):
        self.mask_spikes_model_trials = trials
        v, r, mask_spk = self.model.sample(self.ic.t, np.stack([np.mean(self.ic.stim, 1)] * trials, 1))
        self.v_model = v
        self.mask_spikes_model = mask_spk
        return self

    def set_v_subthreshold_given_spikes(self):
        self.v_subhreshold = self.model.simulate_subthreshold(self.ic.t, self.ic.stim, self.ic.mask_spikes, rate=False)
        return self

    def set_params(self, **kwargs):
        self.model.set_params(**kwargs)
        return self

    def t_spikes(self):
        args = np.where(self.ic.mask_spikes)
        t_spk = (self.ic.t[args[0]],) + args[1:]
        return t_spk

    def fit_subthreshold_voltage(self):
        self.model.fit_subthreshold_voltage(self.ic.t, self.ic.stim, self.ic.data, self.ic.mask_spikes,
                                           self.mask_subthreshold)
        return self

    def fit_supthreshold(self, theta0=None, newton_kwargs=None, verbose=False):
        if newton_kwargs is None:
            self.newton_kwargs = dict(learning_rate=5e-1, initial_learning_rate=1e-2)
        else:
            self.newton_kwargs = newton_kwargs
            self.newton_kwargs['learning_rate'] = newton_kwargs.get('learning_rate', 5e-1)
            self.newton_kwargs['initial_learning_rate'] = newton_kwargs.get('initial_learning_rate', 1e-2)
        self.newton_kwargs['max_iterations'] = newton_kwargs.get('max_iterations', 100)
        self.newton_kwargs['stop_cond'] = newton_kwargs.get('stop_cond', 5e-4)
        self.newton_kwargs['lr_scale'] = newton_kwargs.get('learning_rate_scaling', 0.2)
        self.newton_kwargs['warm_up_iterations'] = newton_kwargs.get('warm_up_iterations', 2)
        self.model.glm.fit(self.ic.t, self.v_subhreshold, self.ic.mask_spikes, theta0=theta0,
                           newton_kwargs=newton_kwargs, verbose=verbose)
        return self

    def fit(self, theta0=None, newton_kwargs=None, verbose=False):
        self.fit_subthreshold_voltage()
        self.set_v_subthreshold_given_spikes()
        self.fit_supthreshold(theta0=theta0, newton_kwargs=newton_kwargs, verbose=verbose)
        self.set_log_likelihood_normed()
        return self

    def plot_filters(self, axs=None):

        if axs is None:
            fig, (ax_kappa, ax_eta) = plt.subplots(figsize=(9, 4), ncols=2)
            ax_kappa.set_xlabel('time'); #ax_kappa.set_ylabel('$\kappa$')
            ax_kappa.set_ylabel('stim filter')
            ax_eta.set_xlabel('time'); #ax_eta.set_ylabel('$\eta$')
            ax_eta.set_ylabel('post-spike filter')
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

    def plot_subthreshold(self, ax=None):
        for sw in range(self.ic.mask_spikes.shape[1]):
            ax.plot(self.t[self.mask_subthreshold[:, sw]], self.data[self.mask_subthreshold[:, sw], sw], 'C0', lw=0.5)
            ax.plot(self.t[self.mask_subthreshold[:, sw]], self.v_model[self.mask_subthreshold[:, sw], sw], 'C1', lw=0.5)

    def plot_fit(self):

        if isinstance(self.model, SRM):
            n_rows, n_cols = 6, 4
            figsize = (15, 7.5)

        fig = plt.figure(figsize=figsize)
        # fig.tight_layout()
        axv = plt.subplot2grid((n_rows, n_cols), (0, 0), rowspan=2, colspan=n_cols - 1)
        axr = plt.subplot2grid((n_rows, n_cols), (2, 0), rowspan=2, colspan=n_cols - 1, sharex=axv)
        ax_log_posterior = plt.subplot2grid((n_rows, n_cols), (0, n_cols - 1), rowspan=1)
        ax_log_prior= plt.subplot2grid((n_rows, n_cols), (1, n_cols - 1), rowspan=1)
        ax_time_rescale_transform = [plt.subplot2grid((n_rows, n_cols), (4, col)) for col in range(2)]
        ax_filters = [plt.subplot2grid((n_rows, n_cols), (5, col)) for col in range(4)]

        if isinstance(self.model, SRM):
            self.plot_subthreshold(ax=axv)

        if self.filename is not None:
            axr.set_title('neuron: ' + self.neuron + ' file: ' + self.filename)

        if self.psth_exp is not None:
            self.plot_psth(ax=axr)

        if self.log_posterior_iterations is not None:
            log_logL = -np.log(-self.model.log_posterior_iterations)
            ax_log_posterior.plot(range(1, len(self.model.log_posterior_iterations) + 1), log_logL, 'C0-')
            log_logL = -np.log(-self.model.log_prior_iterations)
            ax_log_prior.plot(range(1, len(self.model.log_prior_iterations) + 1), log_logL, 'C1-')

        self.plot_time_rescale_transform(ax=ax_time_rescale_transform)
        logL_norm = np.round(self.log_likelihood_normed, 2)
        ax_time_rescale_transform[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
                                              transform=ax_time_rescale_transform[0].transAxes)

        self.model.plot_filters(axs=ax_filters)
        
        return fig, (axr, ax_log_posterior, ax_time_rescale_transform, ax_filters)
    
    def save_pdf(self, folder, pdf_name, off=True):
        
        if off:
            plt.ioff()
            
        fig, axs = self.plot_fit()
        fig.savefig(folder + pdf_name + '.pdf')

    ###################################################################################################
    # GOODNESS OF FIT
    ###################################################################################################

    def set_log_likelihood_normed(self):

        # if not(np.isnan(self.model.log_prior_iterations[-1])):
        #     log_likelihood = self.model.log_posterior_iterations[-1] - self.model.log_prior_iterations[-1]
        # else:
        #     log_likelihood = self.model.log_posterior_iterations[-1]

        if isinstance(self.model, SRM):
            Y_spikes, Y = self.model.glm.get_log_likelihood_kwargs(self.ic.t, self.v_subhreshold, self.ic.mask_spikes)
            theta = np.concatenate(([self.model.u0], self.model.psi.coefs, self.model.gamma.coefs))
            log_likelihood, _, _ = self.model.glm.gh_log_likelihood_kernels(theta, Y_spikes, Y, self.ic.dt)
            # print(self.model.log_posterior_iterations[-1], log_likelihood)
            # self.log_likelihood_normed = log_likelihood

        N_spikes = np.sum(self.ic.mask_spikes, 0)
        N_spikes = N_spikes[N_spikes > 0]
        log_likelihood += np.sum(N_spikes) * np.log(self.ic.dt)

        log_likelihood_poisson = np.sum(N_spikes * (np.log(N_spikes / len(self.ic.t)) - 1))  # L_poisson = rho0**n_neurons * np.exp(-rho0*T)

        self.log_likelihood_normed = (log_likelihood - log_likelihood_poisson) / np.log(2) / np.sum(N_spikes)

        return self

    def psth(self, psth_kernel=None, biased=True, psth_kernel_info=None):

        st_exp = SpikeTrain(self.ic.t, self.ic.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        self.psth_kernel = psth_kernel
        self.psth_kernel_info = psth_kernel_info
        self.psth_exp = st_exp.get_PSTH(psth_kernel)
        self.psth_model = st_model.get_PSTH(psth_kernel)
        self.psth_Ma = st_exp.Ma(st_model, psth_kernel, psth_kernel, biased=biased)
        self.psth_Md = st_exp.Md(st_model, psth_kernel, psth_kernel, biased=biased)

        return self

    def set_Md(self, kernel1, kernel2, biased=True):

        st_exp = SpikeTrain(self.ic.t, self.ic.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        self.Md = st_exp.Md(st_model, kernel1, kernel2, biased=biased)

        return self.Md

    def Ma(self, kernel1, kernel2, biased=True):

        st_exp = SpikeTrain(self.ic.t, self.ic.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        self.Ma = st_exp.Ma(st_model, kernel1, kernel2, biased=biased)

        return self.Ma

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

    def time_rescale_transform(self):

        r, _ = self.model.simulate_subthreshold(self.ic.t, self.ic.stim, self.ic.mask_spikes, I0=self.Ih, rate=True)
        integral_r = np.cumsum(r * self.ic.dt, axis=0)

        z = []

        for sw in range(self.ic.nsweeps):
            Lambda = integral_r[self.ic.mask_spikes[:, sw], sw]  # I think there is no need for shifting the mask
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

