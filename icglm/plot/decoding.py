from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class DecodingPlotter:

    def __init__(self, t=None, model=None, optimizer=None, neurons=None, files=None, I_dec=None, I_true=None):
        self.t = t
        self.model = model
        self.optimizer = optimizer
        self.files = files
        self.neurons = neurons
        self.I_dec = I_dec
        self.I_true = I_true

    def plot_summary(self):

        fig = plt.figure(figsize=(10, 7.5))
        fig.tight_layout()
        axI = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        axr = plt.subplot2grid((3, 3), (1, 0), colspan=3, sharex=axI)
        ax_log_posterior = [plt.subplot2grid((3, 3), (2, col)) for col in range(3)]

        if self.files is not None:
            axI.set_title('neuron: ' + ', '.join(self.neurons) + ' file: ' + ', '.join(self.files))

        axI.plot(self.t, self.I_true, 'C0', zorder=1)
        axI.plot(self.t, self.I_dec, 'C1', zorder=1)
        # if self.var_I_dec is not None:
        #     sd = np.sqrt(self.var_I_dec)
        #     axI.fill_between(self.t, self.I_dec - sd, self.I_dec + sd, color='C1', alpha=.4, zorder=2)
        # elif self.cov_I_dec is not None:
        #     sd = np.sqrt(np.diag(self.cov_dec))
        #     axI.fill_between(self.t, self.I_dec - sd, self.I_dec + sd, color='C1', alpha=.4, zorder=2)

        # for n in range(len(self.files)):
        #     _I = self.Isd[n] * self.I_dec + self.Imu[n]
        #     # _I = self.I_dec
        #     r, v = self.glms[n].simulate_subthr(self.t, _I, self.mask_spk[n], full=True, iterating=True,
        #                                         stim_h=self.Ih[n])
        #     # axr.plot(self.t, r)
        #     t_spk = np.stack([self.t] * self.mask_spk[n].shape[1], 1)[self.mask_spk[n]]
        #     _r_spk = r[self.mask_spk[n]]
        #     n_spikes = np.sum(self.mask_spk[n])
        #     if n_spikes > 1000:
        #         samp = int(n_spikes / 1000)
        #         t_spk = t_spk[::samp]
        #         _r_spk = _r_spk[::samp]
        #     axr.plot(t_spk, _r_spk, '.')

        if self.optimizer is not None:
            # log_log_posterior = -np.log(-self.log_posterior_iterations)
            iterations = np.arange(1, len(self.optimizer.log_posterior_iterations) + 1, 1)
            ax_log_posterior[0].plot(iterations, self.optimizer.log_posterior_iterations, 'C0-')
            ax_log_posterior[1].plot(iterations, self.optimizer.log_posterior_iterations - self.optimizer.log_prior_iterations, 'C0-')
            ax_log_posterior[2].plot(iterations, self.optimizer.log_prior_iterations, 'C0-')

            # logL_norm = np.round(self.logL_norm, 2)
            # ax_log_posterior[0].text(.6, 0.1, 'logL_norm=' + str(logL_norm),
            # transform=ax_time_rescale_transform[0].transAxes)

        return fig, (axI, axr, ax_log_posterior)

    def save_summary(self, path, off=True):
        if off:
            plt.ioff()

        fig, axs = self.plot_summary()
        fig.savefig(path)

