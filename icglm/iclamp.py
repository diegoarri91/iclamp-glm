import itertools
import matplotlib.pyplot as plt
import numpy as np

from utils.abf import load_data, load_protocol
from .masks import extend_trues, shift_mask
from .signals import get_dt, searchsorted, mask_extrema, threshold_crossings
from .utils.time import get_arg
from .spiketrain import SpikeTrain


class IClamp:

    def __init__(self, t=None, data=None, stim=None, mask_spikes=None, path=None, file=None, neuron=None):

        if len(t) != data.shape[0] or len(t) != stim.shape[0]:
            raise

        if data.shape[1] != stim.shape[1]:
            raise

        self.t = t
        self.data = data
        self.stim = stim
        self.mask_spikes = mask_spikes

        self.path = path
        self.file = file
        self.neuron = neuron

    def new(self, t, data, stim, mask_spikes=None):
        metadata = {key: val for key, val in vars(self).items() if key not in ['t', 'data', 'stim', 'mask_spikes']}
        return self.__class__(t=t, data=data, stim=stim, mask_spikes=mask_spikes, **metadata)

    @property
    def v(self):
        return self.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def npoints(self):
        return self.data.shape[0]

    @property
    def nsweeps(self):
        return self.data.shape[1]

    @property
    def dt(self):
        return get_dt(self.t)

    @classmethod
    def load_from_abf(cls, path, use_protocol=False, data_channel=0, stim_channel=1,
                      sweeps=None):

        t, dt, datastim = load_data(path, channels=[data_channel, stim_channel], sweeps=sweeps)

        if use_protocol:
            stim = load_protocol(path, sweeps=sweeps)
            epd = cls(t, datastim[..., 0], stim, path=path)
        else:
            epd = cls(t, datastim[..., 0], datastim[..., 1], path=path)

        return epd

    def sweeps(self, sweeps):
        if self.mask_spikes is not None:
            mask_spikes = self.mask_spikes[:, sweeps]
        else:
            mask_spikes = None
        return self.new(self.t, self.data[:, sweeps], self.stim[:, sweeps], mask_spikes=mask_spikes)

    def plot(self, axv=None, axstim=None, spikes=False, **kwargs):

        sweeps = kwargs.get('sweeps', np.arange(self.nsweeps))
        sweeps = np.array(sweeps)
        lw = kwargs.get('lw', 1)

        if axv is None:
            fig = plt.figure(figsize=(12, 5))
            fig.subplots_adjust(hspace=.5)
            r = 8
            axv = plt.subplot2grid((10, 1), (0, 0), rowspan=r)
            axv.xaxis.set_visible(False)
            axv.set_ylabel('voltage')
            axv.spines['right'].set_visible(False)
            axv.spines['top'].set_visible(False)
            axv.spines['bottom'].set_visible(False)
            axstim = plt.subplot2grid((10, 1), (r, 0), rowspan=10 - r)
            axstim.spines['top'].set_visible(False)
            axstim.spines['right'].set_visible(False)
            axstim.set_xlabel('time')
            axstim.set_ylabel('stim')
        else:
            fig = None

        axv.plot(self.t, self.v[:, sweeps], lw=lw)
        axstim.plot(self.t, self.stim[:, sweeps], lw=lw)

        if spikes:
            mask_spikes = self.mask_spikes
            for sw in sweeps:
                axv.plot(self.t[mask_spikes[:, sw]], self.data[mask_spikes[:, sw], sw], 'o', lw=.7)


        return fig, (axv, axstim)

    def plot_raster(self, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7), nrows=1)
        else:
            fig = None

        st= SpikeTrain(self.t, self.mask_spikes)

        st.plot(ax=ax, color='dodgerblue', **kwargs)

        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('time')

        return fig, ax

    def plot_psth(self, ax=None, kernel=None, lw=1):

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 7), nrows=1)
        else:
            fig = None

        st= SpikeTrain(self.t, self.mask_spikes)
        psth = st.get_PSTH(kernel)

        ax.plot(self.t, psth, color='dodgerblue', lw=lw)

        # ax.set_yticks([])
        # ax.spines['left'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        ax.set_xlabel('time')

        return fig, ax

    def restrict(self, t0=None, tf=None, reset_time=True):
        
        t0 = t0 if t0 is not None else self.t[0]
        tf = tf if tf is not None else self.t[-1] + self.dt
        arg0, argf = searchsorted(self.t, [t0, tf])
        
        t = self.t[arg0:argf]
        data = self.data[arg0:argf]
        stim = self.stim[arg0:argf]
        if self.mask_spikes is not None:
            mask_spikes = self.mask_spikes[arg0:argf]
        else:
            mask_spikes = None

        if reset_time:
            t = t - t[0]

        return self.new(t, data, stim, mask_spikes=mask_spikes)

    def subsample(self, dt, average_stim=False):
        # FIRST set_mask_spikes should be called so mask_spikes should be set without subsampling!
        n_sample = get_arg(dt, self.dt)

        if average_stim:
            bins = np.arange(0, len(self.t) + n_sample, n_sample)
            bins[-1] = len(self.t)
            stim = np.array([np.mean(self.stim[bins[ii]:bins[ii + 1]], 0) for ii in range(len(bins) - 1)])
            t = self.t[::n_sample]
            data = self.data[::n_sample]
        else:
            t = self.t[::n_sample]
            data = self.data[::n_sample]
            stim = self.stim[::n_sample]
        
        if self.mask_spikes is not None:
            arg_spikes = np.where(self.mask_spikes)
            arg_spikes = (np.array(np.floor(arg_spikes[0]/n_sample), dtype=int), ) + arg_spikes[1:]
            mask_spikes = np.zeros(data.shape, dtype=bool)
            mask_spikes[arg_spikes] = True
        else:
            mask_spikes = None
        
        return self.new(t, data, stim, mask_spikes=mask_spikes)
    
    def get_Ih(self, th=200):

        argh = searchsorted(self.t, th)
        Ih = np.mean(self.stim[:argh, ...], 0)

        return Ih

    def subtract_Ih(self, th=200):

        Ih = self.get_Ih(th)

        return self.new(self.t, self.data, self.stim - Ih, mask_spikes=self.mask_spikes)

    # =============================================================================
    # Spikes methods
    # =============================================================================

    def set_mask_spikes(self, t0=None, tf=None, thr=-13., time_window=3,
                    use_derivative=False, dvdt_threshold=9, t_before_spike_peak=0, tref=4):
        
        self.mask_spikes = self.get_mask_spikes(t0=t0, tf=tf, thr=thr, time_window=time_window,
                    use_derivative=use_derivative, dvdt_threshold=dvdt_threshold, t_before_spike_peak=t_before_spike_peak, tref=tref)
        
        return self
    
    def get_mask_spikes(self, t0=None, tf=None, thr=-13, time_window=3,
                    use_derivative=False, dvdt_threshold=9, t_before_spike_peak=0, tref=4):
        # 24/09/2018
        # updated 23/04/2019 to tref and so it's equal to GLMFitter

        t0 = t0 if t0 is not None else self.t[0]
        tf = tf if tf is not None else self.t[-1] + self.dt
        arg0, argf = searchsorted(self.t, [t0, tf])

        arg_order = get_arg(time_window, self.dt)
        arg_before_spk = get_arg(t_before_spike_peak, self.dt)
        argref = get_arg(tref, self.dt)

        mask_spk_peak = self.data > thr
        mask_spk_peak = mask_extrema(self.data, mask_data=mask_spk_peak, order=arg_order, left_comparator=np.greater,
                                     right_comparator=np.greater_equal)
        # greater_equal is necessary to the rights because of the discrete values of sampling. 
        # one spike could have its peak formed by two values of the same size. the function will return only the first one

        if use_derivative:

            mask_dvdt_thres_crossings = threshold_crossings(self.get_dvdt(), dvdt_threshold)

            # when they cross the threshold and come t_before_spike_peak ms before a spike peak
            mask_spk_dvdt = mask_dvdt_thres_crossings & extend_trues(mask_spk_peak, arg_before_spk, 0)
            # previous thing shouldnt find more than one value in each. I give back the first
            mask_spk_dvdt = mask_spk_dvdt & ~extend_trues(shift_mask(mask_spk_dvdt, 1, fill_value=False), 0,
                                                          arg_before_spk)
            # if for any spike no value is found for the derivative I get the value given by t_before_spk
            mask_spk = mask_spk_dvdt | shift_mask(mask_spk_peak, -arg_before_spk, fill_value=False)
            mask_spk = mask_spk & ~extend_trues(shift_mask(mask_spk, -1, fill_value=False), argref, 0)

        else:

            mask_spk = shift_mask(mask_spk_peak, -arg_before_spk, fill_value=False)
            # there can be no two spikes separated by less than tref. I give back the first
            mask_spk = mask_spk & ~extend_trues(shift_mask(mask_spk, 1, fill_value=False), 0, argref)

        mask_spk[:arg0] = False
        mask_spk[argf:] = False

        return mask_spk

    def get_mask_away_from_spikes(self, tl, tr, t0=None, tf=None, mask_spikes_kwargs=None):

        t0 = t0 if t0 is not None else self.t[0]
        tf = tf if tf is not None else self.t[-1] + self.dt

        arg0, argf = searchsorted(self.t, [t0, tf])
        argl, argr = get_arg([tl, tr], self.dt)

        if self.mask_spikes is not None:
            mask = ~extend_trues(self.mask_spikes, argl, argr)
        else:
            mask = ~extend_trues(self.get_mask_spikes(**mask_spikes_kwargs), argl, argr)

        mask[:arg0] = False
        mask[argf:] = False

        return mask

    def get_SpikeTrain(self):
        return SpikeTrain(self.t, self.mask_spikes)

    @property
    def spiking_sweeps(self):
        mask_spk = self.get_mask_spikes()
        return np.where(np.any(mask_spk, 0))[0]

    @property
    def non_spiking_sweeps(self):
        mask_spk = self.get_mask_spikes()
        return np.where(~np.any(mask_spk, 0))[0]

    def mask_spiking_sweeps(self):
        mask_spk = self.get_mask_spikes()
        return np.any(mask_spk, 0)

    def get_spike_count(self, time_bins, average_sweeps=False, mask_spikes_kwargs=None):
        # 08/09/2018
        # Given arbitrary time bins computes the spike counts in those bins for each sweep
        # unless average=True

        if mask_spikes_kwargs is None:
            mask_spikes_kwargs = {}

        arg_bins = searchsorted(self.t, time_bins)

        spk_count = np.zeros((len(arg_bins) - 1, self.nsweeps))

        mask_spk = self.get_mask_spikes(**mask_spikes_kwargs)

        for sw in range(self.nsweeps):
            arg_spk = np.where(mask_spk[:, sw])[0]

            spk_count[:, sw], _ = np.histogram(arg_spk, bins=arg_bins)

        if average_sweeps:
            spk_count = np.mean(spk_count, 1)
            if spk_count.size == 1:
                return spk_count[0]
            else:
                return spk_count
        else:
            return np.squeeze(spk_count)

    def get_firing_rate(self, time_bins, average_sweeps=False, mask_spikes_kwargs=None):
        """
        27/08/2018 
        Given arbitrary time bins computes the firing rate in Hz in those bins for each sweep
        unless average=True # 08/09/2018 Modified using get_spike_count
        """

        rate = self.get_spike_count(time_bins, average_sweeps=average_sweeps, mask_spikes_kwargs=mask_spikes_kwargs)
        rate = rate / np.diff(time_bins)[:, None] * 1000

        return rate

    def fano_factor(self, bins, mask_spikes_kwargs={}):

        spk_count = self.get_spike_count(bins, average_sweeps=average_sweeps, mask_spikes_kwargs=mask_spikes_kwargs)

        return np.var(spk_count, 1) / np.mean(spk_count, 1)

    def get_PSTH(self, kernel, average_sweeps=True):
        psth = self.get_SpikeTrain().get_PSTH(kernel, average_sweeps=average_sweeps)
        return psth

    def get_ISI_distribution(self, mask_spikes_kwargs={}):
        # 12/10/2018

        isi_dist = []

        mask_spk = self.get_mask_spikes(**mask_spikes_kwargs)

        for sw in range(self.nsweeps):
            t_spk = self.t[mask_spk[:, sw]]
            isi_dist += [np.diff(t_spk)]

        return isi_dist

    def get_dvdt(self):
        dvdt = np.diff(self.v, axis=0) / self.dt
        dvdt = np.concatenate((dvdt, dvdt[-1:, ...]), axis=0)
        return dvdt

    def get_MSE_matrix(self, mask):
        # 28/09/2018
        mse = np.zeros((self.nsweeps, self.nsweeps))
        T = np.sum(mask)

        for sw1, sw2 in itertools.combinations(range(self.nsweeps), 2):
            mse[sw1, sw2] = 1. / T * np.sum((self.data[mask, sw1] - self.data[mask, sw2]) ** 2.)
            mse[sw2, sw1] = mse[sw1, sw2]

        return mse

    def get_RMSE(self, mask):
        triu_indices = np.triu_indices(self.nsweeps, k=1)
        return np.sqrt(np.mean(self.get_MSE_matrix(mask)[triu_indices]))

    # =============================================================================
    #
    # =============================================================================

    def get_vhIh(self, th=200., argh=None):
        """
        Returns average value from 0 until t=th or arg=argh of voltage and current 
        
        Parameters
        ----------
        th : float, optional
            Final time of averaging range
        argh : int, optional
            Final argument of averaging range
        Returns
        ----------
        vh : float
            average value of voltage until th
        Ih : float
            average value of current until th
        """
        if argh is None:
            argh = searchsorted(self.t, th)

        vh = np.mean(self.data[:argh, ...], axis=0)
        Ih = np.mean(self.stim[:argh, ...], axis=0)

        return vh, Ih

    def get_vrst(self, tref=4.5, **kwargs):

        t0 = kwargs.get('t0', 0.)
        tf = kwargs.get('tf', self.t[-1] + self.dt)

        arg_ref = get_arg(tref, self.dt)

        mask_spk = self.get_mask_spikes(t0=t0, tf=tf)
        arg_spk = np.where(mask_spk)

        arg_spike_ref = (arg_spk[0] + arg_ref, arg_spk[1])

        vrst = np.mean(self.v[arg_spike_ref])

        return vrst

    def get_vr(self, t0, tf, **kwargs):
        sweeps = kwargs.get('sweeps', range(self.nsweeps))

        t_sweeps = np.ones((self.t.shape[0], len(sweeps))) * self.t[:, None]

        mask_vr = (t_sweeps < t0) | (t_sweeps > tf)

        vr = np.mean(self.v[mask_vr])
        return vr


class HoldNoise(IClamp):
    
    pass