from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from .spiketrain import SpikeTrainsPlotter
from ..spiketrain import SpikeTrain


class FitPlotter:

    def __init__(self, ic=None, model=None, optimizer=None, neuron=None, file=None, log_likelihood_normed=None,
                 z=None, ks_stats=None, mask_spikes_model=None, psth_exp=None, psth_model=None, Md=None, Ma=None):
        self.ic = ic
        self.model = model
        self.optimizer = optimizer
        self.file = file
        self.neuron = neuron

        self.log_likelihood_normed = log_likelihood_normed
        self.z = z
        self.ks_stats = ks_stats
        self.mask_spikes_model = mask_spikes_model
        self.psth_exp = psth_exp
        self.psth_model = psth_model
        self.Ma = Ma
        self.Md = Md

    @abstractmethod
    def plot_filters(self, axs):
        pass

    def plot_psth(self, ax=None, data=True, model=True, colors=None, **kwargs):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5), ncols=1)
        colors = colors if colors is not None else ['C0', 'C1']
        if data:
            ax.plot(self.ic.t, self.psth_exp, color=colors[0], **kwargs)
        if model:
            ax.plot(self.ic.t, self.psth_model, color=colors[1], **kwargs)
        ax.set_ylim(-.001, np.max(np.concatenate((self.psth_model, self.psth_exp), axis=1)) * 1.15)
        ax.set_xlabel('time')
        ax.set_ylabel('psth')

        if self.Md is not None:
            Md = np.round(self.Md, 3)
            ax.text(.15, 0.92, 'Md=' + str(Md), transform=ax.transAxes)

        if self.Ma is not None:
            Ma = np.round(self.Ma, 3)
            ax.text(.75, 0.92, 'Ma=' + str(Ma), transform=ax.transAxes)

        return fig, ax

    def plot_raster2(self, data=True, model=True, colors=('C0', 'C1'), psth_kwargs=None):

        # fig, axs = plt.subplots(figsize=(12, 7), nrows=3, sharex=True)
        fig = plt.figure(figsize=(12, 7))
        r1, r2 = self.ic.mask_spikes.shape[1], self.mask_spikes_model.shape[1]
        k = 2
        r3 = k * (r1 + r2)
        axs = []
        axs += [plt.subplot2grid((r1 + r2 + r3, 1), (0, 0), rowspan=r1)]
        axs += [plt.subplot2grid((r1 + r2 + r3, 1), (r1, 0), rowspan=r2, sharex=axs[0])]
        axs += [plt.subplot2grid((r1 + r2 + r3, 1), (r1 + r2, 0), rowspan=k * (r1 + r2), sharex=axs[0])]
        fig.subplots_adjust(hspace=-.1)

        st_exp = SpikeTrain(self.ic.t, self.ic.mask_spikes)
        st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        if data:
            st_exp.plot(ax=axs[0], color=colors[0])
        if model:
            st_model.plot(ax=axs[1], color=colors[1])

        psth_kwargs = psth_kwargs if psth_kwargs is not None else {}
        self.plot_psth(axs[2], data=data, model=model, colors=colors, **psth_kwargs)

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

    def plot_raster(self, data=True, model=True, colors=('C0', 'C1'), figsize=(7, 5), hspace=-0.1, spike_kwargs=None, psth_kwargs=None):

        # fig, axs = plt.subplots(figsize=(12, 7), nrows=3, sharex=True)

        spike_kwargs = spike_kwargs if spike_kwargs is not None else {}

        fig = plt.figure(figsize=figsize)
        r1, r2 = self.ic.mask_spikes.shape[1], self.mask_spikes_model.shape[1]
        k = 2
        r3 = k * (r1 + r2)
        axs = []
        axs += [plt.subplot2grid((r1 + r2 + r3, 1), (0, 0), rowspan=r1)]
        axs += [plt.subplot2grid((r1 + r2 + r3, 1), (r1, 0), rowspan=r2, sharex=axs[0])]
        axs += [plt.subplot2grid((r1 + r2 + r3, 1), (r1 + r2, 0), rowspan=k * (r1 + r2), sharex=axs[0])]
        fig.subplots_adjust(hspace=hspace)

        # st_exp = SpikeTrain(self.ic.t, self.ic.mask_spikes)
        # st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)

        if data:
            SpikeTrainsPlotter(self.ic.t, self.ic.mask_spikes).plot(axs[0], colors=colors[0], **spike_kwargs)
            axs[0].set_ylabel('data')
        if model:
            SpikeTrainsPlotter(self.ic.t, self.mask_spikes_model).plot(axs[1], colors=colors[1], **spike_kwargs)
            axs[1].set_ylabel('GLM')
        else:
            axs[1].xaxis.set_visible(False)
            axs[1].set_yticks([])
            axs[1].spines['left'].set_visible(False)
            axs[1].spines['right'].set_visible(False)
            axs[1].spines['bottom'].set_visible(False)
            axs[1].spines['top'].set_visible(False)


        psth_kwargs = psth_kwargs if psth_kwargs is not None else {}
        self.plot_psth(axs[2], data=data, model=model, colors=colors, **psth_kwargs)
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].set_xlabel('time')

        return fig, axs

    def plot_posterior_iterations(self, ax):
        fig = None
        if ax is None:
            fig, ax = plt.subplots(5, 5)

        # log_logL = -np.log(-self.optimizer.log_posterior_iterations)
        log_logL = self.optimizer.log_posterior_iterations
        ax.plot(range(1, len(self.optimizer.log_posterior_iterations) + 1), log_logL, 'C0-o')
        ax.set_xlabel('iterations')
        ax.set_ylabel('posterior')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        return fig, ax

    def plot_time_rescale_transform(self, axs=None):

        if axs is None:
            fig, axs = plt.subplots(figsize=(10, 5), ncols=2)

        bins = np.arange(0, 1.01, .01)

        z = np.concatenate(self.z)
        if self.ks_stats is not None:
            stat, p = np.round(self.ks_stats[0], 4), np.round(self.ks_stats[1], 3)
        else:
            stat, p = '', ''

        axs[0].text(.05, .7, 'stat=' + str(stat) + '\np=' + str(p), transform=axs[0].transAxes)

        values, bins = np.histogram(z, bins=bins)
        cum = np.append(0., np.cumsum(values) / np.sum(values))
        axs[0].plot(bins, cum, '-')
        axs[0].plot([0, 1], [0, 1], 'k--')

        for sw in range(self.ic.nsweeps):
            axs[1].plot(self.z[sw][:-1], self.z[sw][1:], 'C0.')

        return axs

    @abstractmethod
    def plot_summary(self, axs):
        pass

    @abstractmethod
    def save_summary(self, path, off=True):
        if off:
            plt.ioff()

        fig, axs = self.plot_summary()
        fig.savefig(path)

class GLMPlotter(FitPlotter):

    def plot_filters(self, axs=None, dt=.1, invert_kappa=True, invert_eta=True, exp_eta=False):
        fig = None
        if axs is None:
            fig, (ax_kappa, ax_eta) = plt.subplots(figsize=(9, 4), ncols=2)
        else:
            ax_kappa = axs[0]
            ax_eta = axs[1]

        ax_kappa.set_xlabel('time');  # ax_kappa.set_ylabel('$\kappa$')
        ax_kappa.set_ylabel('stim filter')
        ax_eta.set_xlabel('time');  # ax_eta.set_ylabel('$\eta$')
        ax_eta.set_ylabel('post-spike filter')
        ax_kappa.spines['right'].set_visible(False)
        ax_kappa.spines['top'].set_visible(False)
        ax_eta.spines['right'].set_visible(False)
        ax_eta.spines['top'].set_visible(False)

        t_kappa = np.arange(0., self.model.kappa.support[1], dt)
        ax_kappa = self.model.kappa.plot(t_kappa, ax=ax_kappa, invert_t=invert_kappa)
        t_eta = np.arange(0., self.model.eta.support[1], dt)
        ax_eta = self.model.eta.plot(t_eta, ax=ax_eta, invert_values=invert_eta, exp_values=exp_eta)

        return fig, (ax_kappa, ax_eta)

    def plot_summary(self):

        fig = plt.figure(figsize=(10, 7.5))
        fig.tight_layout()
        axr = plt.subplot2grid((5, 3), (0, 0), rowspan=2, colspan=2)
        ax_log_posterior = plt.subplot2grid((5, 3), (0, 2), rowspan=1)
        ax_log_prior= plt.subplot2grid((5, 3), (1, 2), rowspan=1)
        ax_time_rescale_transform = [plt.subplot2grid((5, 2), (2, col)) for col in range(2)]
        ax_kappa = [plt.subplot2grid((5, 3), (3, col)) for col in range(3)]

        if self.file is not None:
            axr.set_title('neuron: ' + self.neuron + ' file: ' + self.file)

        if self.psth_exp is not None:
            self.plot_psth(ax=axr)

        if self.optimizer is not None:
            log_logL = -np.log(-self.optimizer.log_posterior_iterations)
            ax_log_posterior.plot(range(1, len(self.optimizer.log_posterior_iterations) + 1), log_logL, 'C0-')
            log_logL = -np.log(-self.optimizer.log_prior_iterations)
            ax_log_prior.plot(range(1, len(self.optimizer.log_prior_iterations) + 1), log_logL, 'C1-')

        self.plot_time_rescale_transform(axs=ax_time_rescale_transform)
        logL_norm = np.round(self.log_likelihood_normed, 2)
        ax_time_rescale_transform[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
                                              transform=ax_time_rescale_transform[0].transAxes)

        t_kappa = np.arange(0., self.model.kappa.tbins[-1], .1)
        self.model.kappa.plot_lin_log(t_kappa, axs=ax_kappa)

        if self.model.eta is not None:
            ax_eta = [plt.subplot2grid((5, 3), (4, col)) for col in range(3)]
            t_eta = np.arange(0., self.model.eta.tbins[-1], .1)
            self.model.eta.plot_lin_log(t_eta, axs=ax_eta)
            return fig, (axr, ax_log_posterior, ax_time_rescale_transform, ax_kappa, ax_eta)

        return fig, (axr, ax_log_posterior, ax_time_rescale_transform, ax_kappa)

class SRMPlotter(FitPlotter):

    def __init__(self, ic=None, srm=None, mask_subthreshold=None, optimizer=None, neuron=None, file=None, log_likelihood_normed=None,
                 z=None, ks_stats=None, psth_exp=None, psth_model=None, Md=None, Ma=None, v_exp=None, v_model=None):
        super().__init__(ic=ic, model=srm, optimizer=optimizer, neuron=neuron, file=file,
                         log_likelihood_normed=log_likelihood_normed, z=z, ks_stats=ks_stats, psth_exp=psth_exp,
                         psth_model=psth_model, Md=Md, Ma=Ma)
        self.v_exp = v_exp
        self.v_model = v_model
        self.mask_subthreshold = mask_subthreshold

    def plot_filters(self, axs=None):

        if axs is None:
            fig, axs = plt.subplots(figsize=(12, 4), ncols=3)
            axs[0].set_ylabel('stim filter')
            axs[1].set_ylabel('post-spike v filter')
            axs[2].set_ylabel('post-spike thr filter')
            for ax in axs:
                ax.set_xlabel('time');  # ax_kappa.set_ylabel('$\kappa$')
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
        else:
            fig = None

        t_kappa = np.arange(0., self.model.kappa.tbins[-1], .1)
        self.model.kappa.plot(t_kappa, ax=axs[0])
        axs[0].text(.45, 0.86, 'vr=' + str(np.round(self.model.vr, 1)), transform=axs[0].transAxes, fontsize=12)
        t_eta = np.arange(self.model.eta.tbins[0], self.model.eta.tbins[-1], .1)
        self.model.eta.plot(t_eta, ax=axs[1])
        t_gamma = np.arange(self.model.gamma.tbins[0], self.model.gamma.tbins[-1], .1)
        self.model.gamma.plot(t_gamma, ax=axs[2])
        string = 'vt=' + str(np.round(self.model.vt, 1)) + '\n' + 'dv=' + str(np.round(self.model.dv, 2))
        axs[2].text(.45, 0.76, string, transform=axs[2].transAxes, fontsize=12)

        return fig, axs

    def plot_summary(self):

        n_rows, n_cols = 8, 3
        figsize = (12.5, 10)

        fig = plt.figure(figsize=figsize)
        fig.tight_layout()
        axv = plt.subplot2grid((n_rows, n_cols), (0, 0), rowspan=2, colspan=3)
        axr = plt.subplot2grid((n_rows, n_cols), (2, 0), rowspan=2, colspan=3, sharex=axv)
        ax_log_posterior = plt.subplot2grid((n_rows, n_cols), (4, 0), rowspan=1)
        ax_log_prior = plt.subplot2grid((n_rows, n_cols), (5, 0), rowspan=1)
        ax_time_rescale_transform = [plt.subplot2grid((n_rows, n_cols), (4, 1+col), rowspan=2) for col in range(2)]
        ax_filters = [plt.subplot2grid((n_rows, n_cols), (6, col), rowspan=2) for col in range(3)]

        if self.v_exp is not None:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v_exp = self.v_exp.copy()
                v_exp[~self.mask_subthreshold] = np.nan
                v_exp = np.nanmean(v_exp, 1)
                v_model = self.v_model.copy()
                v_model[~self.mask_subthreshold] = np.nan
                v_model = np.nanmean(v_model, 1)
            axv.plot(self.ic.t, v_exp, 'C0', lw=0.5)
            axv.plot(self.ic.t, v_model, 'C1', lw=0.5)

        if self.file is not None:
            axr.set_title('neuron: ' + self.neuron + ' file: ' + self.file)

        if self.psth_exp is not None:
            self.plot_psth(ax=axr, lw=.6)
        #
        if self.optimizer is not None:
            log_logL = -np.log(-self.optimizer.log_posterior_iterations)
            ax_log_posterior.plot(range(1, len(self.optimizer.log_posterior_iterations) + 1), log_logL, 'C0-')
            log_logL = -np.log(-self.optimizer.log_prior_iterations)
            ax_log_prior.plot(range(1, len(self.optimizer.log_prior_iterations) + 1), log_logL, 'C1-')

        self.plot_time_rescale_transform(axs=ax_time_rescale_transform)
        logL_norm = np.round(self.log_likelihood_normed, 2)
        ax_time_rescale_transform[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
                                          transform=ax_time_rescale_transform[0].transAxes)

        self.plot_filters(axs=ax_filters)

        return fig, (axv, axr, ax_log_posterior, ax_time_rescale_transform, ax_filters)




    # def plot_ic(self, axv=None, axstim=None, spikes=False, **kwargs):
    #     return self.ic.plot(axv=axv, axstim=axstim, spikes=spikes, **kwargs)
    #

