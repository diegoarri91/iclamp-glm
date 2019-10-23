import matplotlib.pyplot as plt
import numpy as np

from ..spiketrain import SpikeTrain
from .spiketrain import SpikeTrainsPlotter
from ..utils.time import get_dt, searchsorted

class DecodingPlotter:

    def __init__(self, t=None, model=None, optimizer=None, mask_spikes=None, neurons=None, files=None, stim_dec=None,
                 var_stim_dec=None, stim_true=None):
        self.t = t
        self.model = model
        self.optimizer = optimizer
        self.mask_spikes = mask_spikes
        self.files = files
        self.neurons = neurons
        self.stim_dec = stim_dec
        self.var_stim_dec = var_stim_dec
        self.stim_true = stim_true

    def plot_decoded_stimulus(self, ax_raster=None, axI=None, t0=None, tf=None, spike_kwargs=None, stim_kwargs=None):

        dt = get_dt(self.t)
        t0 = t0 if t0 is not None else self.t[0]
        tf = tf if tf is not None else self.t[-1] + dt
        stim_kwargs = stim_kwargs if stim_kwargs else {}
        spike_kwargs = spike_kwargs if spike_kwargs else {}

        arg0, argf = searchsorted(self.t, [t0, tf])

        n_neurons = len(self.mask_spikes)
        fig = None
        if axI is None:
            fig = plt.figure(figsize=(12, 4.5))
            r2 = 1
            r1 = 2 * r2 * n_neurons
            axI = plt.subplot2grid((r1 + r2 * n_neurons, 1), (n_neurons * r2, 0), rowspan=r1)
            ax_raster = [plt.subplot2grid((r1 + r2 * n_neurons, 1), (ii * r2, 0), rowspan=r2, sharex=axI) for ii in range(n_neurons)]

        axI.plot(self.t[arg0:argf], self.stim_true[arg0:argf], color='C0', zorder=1, **stim_kwargs)
        axI.plot(self.t[arg0:argf], self.stim_dec[arg0:argf], color='C1', zorder=2, **stim_kwargs)

        SpikeTrainsPlotter(self.t, self.mask_spikes).plot(ax=ax_raster, t0=t0, tf=tf, **spike_kwargs)

        if self.var_stim_dec is not None:
            sd = np.sqrt(self.var_stim_dec)[arg0:argf]
            axI.fill_between(self.t[arg0:argf], self.stim_dec[arg0:argf] - sd, self.stim_dec[arg0:argf] + sd, color='C1', alpha=.15, zorder=2)

        axI.spines['top'].set_visible(False)
        axI.spines['right'].set_visible(False)
        axI.set_xlabel('time')
        axI.set_ylabel('stim')

        return fig, (ax_raster, axI)

    def plot_summary(self):

        fig = plt.figure(figsize=(16, 7.5))
        fig.tight_layout()
#         r1, r2 = 2, 1
# #         ax_raster = [plt.subplot2grid((r1 + r2 + 1, 3), (0, 0), rowspan=r2, colspan=3) for
#         ax_raster = [plt.subplot2grid((r1 + r2 * n_neurons, 1), (ii * r2, 0), rowspan=r2, sharex=axI) for ii in range(n_neurons)]
#         axI = plt.subplot2grid((r1 + r2 * n_neurons + 1, 3), (r2, 0), rowspan=r1, colspan=3, sharex=ax_raster)
        n_neurons = len(self.mask_spikes)
        r1, r2 = 4, 1
        axI = plt.subplot2grid((r1 + r2 * n_neurons + 1, 1), (n_neurons * r2, 0), rowspan=r1)
        ax_raster = [plt.subplot2grid((r1 + r2 * n_neurons + 1, 1), (ii * r2, 0), rowspan=r2, sharex=axI) for ii in range(n_neurons)]
        ax_log_posterior = [plt.subplot2grid((r1 + r2 * n_neurons + 1, 3), (r1 + n_neurons * r2, col)) for col in range(3)]

        if self.files is not None:
            ax_raster[0].set_title('neuron: ' + ', '.join(self.neurons) + ' file: ' + ', '.join(self.files))

        self.plot_decoded_stimulus(ax_raster=ax_raster, axI=axI)
        # if self.var_I_dec is not None:
        #     sd = np.sqrt(self.var_I_dec)
        #     axI.fill_between(self.t, self.I_dec - sd, self.I_dec + sd, color='C1', alpha=.4, zorder=2)
        # elif self.cov_I_dec is not None:
        #     sd = np.sqrt(np.diag(self.cov_dec))
        #     axI.fill_between(self.t, self.I_dec - sd, self.I_dec + sd, color='C1', alpha=.4, zorder=2)

        if self.optimizer is not None:
            iterations = np.arange(1, len(self.optimizer.log_posterior_iterations) + 1, 1)
            ax_log_posterior[0].plot(iterations, self.optimizer.log_posterior_iterations, 'C0-')
            ax_log_posterior[1].plot(iterations, self.optimizer.log_posterior_iterations - self.optimizer.log_prior_iterations, 'C0-')
            ax_log_posterior[2].plot(iterations, self.optimizer.log_prior_iterations, 'C0-')

            # logL_norm = np.round(self.logL_norm, 2)
            # ax_log_posterior[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
            # transform=ax_time_rescale_transform[0].transAxes)

        return fig, (axI, ax_raster, ax_log_posterior)

    def save_summary(self, path, off=True):
        if off:
            plt.ioff()

        fig, axs = self.plot_summary()
        fig.savefig(path)

