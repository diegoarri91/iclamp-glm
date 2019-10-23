import numpy as np
import matplotlib.pyplot as plt

from ..utils.time import get_dt


def plot_mask(t, mask_spikes, ax, t0, tf, offset, lw, delta, color):

    for sw in range(mask_spikes.shape[1]):
        t_spikes_ = t[mask_spikes[:, sw]]
        t_spikes_ = t_spikes_[(t_spikes_ >= t0) & (t_spikes_ < tf)]
        t_spikes_ = np.array([t_spikes_] * 2)

        bars = np.zeros((2, t_spikes_.shape[1])) * np.nan
        bars[0, :] = -sw + delta / 2. - offset
        bars[1, :] = -sw - delta / 2. - offset

        ax.plot(t_spikes_, bars, '-', linewidth=lw, color=color)


class SpikeTrainsPlotter:

    def __init__(self, t, mask_spikes):
        self.t = t
        self.mask_spikes = mask_spikes
        if isinstance(mask_spikes, list):
            self.n_trains = len(mask_spikes)
            self._is_list = True
        else:
            self.n_trains = 1
            self._is_list = False

    def plot(self, ax=None, t0=None, tf=None, offset=0, colors=None, lw=1, delta=0.8, same_ax=False, **kwargs):

        colors = colors if colors else ['C' + str(ii % 10) for ii in range(self.n_trains)]

        t0 = t0 if t0 else 0
        tf = tf if tf else self.t[-1]

        if ax is None:
            figsize = kwargs.get('figsize', (8, 5))
            fig, ax = plt.subplots(figsize=figsize, nrows=self.n_trains, sharex=True)

        for n in range(self.n_trains):
            if self._is_list:
                _ax = ax[n]
                _mask = self.mask_spikes[n]
                _color = colors[n]
            else:
                _ax = ax
                _mask = self.mask_spikes
                _color = colors[0]
            if same_ax:
                pass
            else:
                plot_mask(self.t, _mask, _ax, t0, tf, offset, lw, delta, _color)
                _ax.xaxis.set_visible(False)
                _ax.set_yticks([])
                _ax.spines['left'].set_visible(False)
                _ax.spines['right'].set_visible(False)
                _ax.spines['bottom'].set_visible(False)
                _ax.spines['top'].set_visible(False)

        # extra_range = (self.t[-1] - self.t[0]) * .005
        # ax.set_xlim(self.t[0] - extra_range, self.t[-1] + extra_range)
        # ax.set_ylim(-ii - offset - (delta / 2. + (1 - delta)), delta / 2. + (1 - delta))

        return ax
