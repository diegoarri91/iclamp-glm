#!/usr/bin/env

import itertools
import matplotlib.pyplot as plt
import numpy as np

from .spiketrain import SpikeTrain

from fun_signals import get_arg, searchsorted, mask_args_away_from_maskinp, mask_extrema, mask_thres_crossings, extend_trues, shift_array


class IClamp:
    
    def __init__(self, t=None, data=None, stim=None, path=None, file=None, neuron=None):
        
        if len(t) != data.shape[0] or len(t) != stim.shape[0]:
            raise
        
        if data.shape[1] != stim.shape[1]:
            raise
            
        self.t = t
        self.data = data
        self.stim = stim
        
        self.path = path
        self.file = file
        self.neuron = neuron

    def new(self, t, data, stim):
        metadata = {key: val for key, val in vars(self).items() if key not in ['t', 'data', 'stim']}
        return self.__class__(t=t, data=data, stim=stim, **metadata) 
        
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
    def from_file(cls, folder, file, use_protocol=False, data_channel=0, stim_channel=1, sweeps=None):
        
        t, dt, datastim = load_data(folder + '/' + file + '.abf', channels=[data_channel, stim_channel], sweeps=sweeps)
        
        if use_protocol:
            stim = load_protocol(folder, file, sweeps=sweeps)
            epd = cls(t, datastim[...,0], stim, path = folder + file, neuron=folder, file=file)
        else:
            epd = cls(t, datastim[...,0], datastim[...,1], path=folder + '/' + file + '.abf', neuron=folder, file=file)

        return epd
    
    def sweeps(self, sweeps):
        return self.new(self.t, self.data[:, sweeps], self.stim[:, sweeps])

    def plot(self, axv=None, axstim=None, mask_spikes_kwargs={}, **kwargs):
        
        sweeps = kwargs.get('sweeps', np.arange(self.nsweeps))
        spikes = kwargs.get('spikes', False)
        
        if axv is None:
            figsize = kwargs.get('figsize', (8, 5) )
            fig, (axv, axstim) = plt.subplots( figsize = figsize, nrows = 2, sharex = True )
            fig.tight_layout()

        mask_spike = self.mask_spikes(**mask_spikes_kwargs)
        
        axv.plot(self.t, self.v[:,sweeps])
        axstim.plot(self.t, self.stim[:,sweeps])
        
        if spikes:
            for sw in sweeps:
                axv.plot( self.t[mask_spike[:, sw]], self.data[mask_spike[:, sw], sw], 'o' )
            
        return axv, axstim
    
    def restrict(self, t0=None, tf=None, arg0=None, argf=None, reset_time=False):
        
        if arg0 is None and argf is None:
            t, data = restrict(self.t, self.data, t0, tf)
            _, stim = restrict(self.t, self.stim, t0, tf)
            
        else:
            t = self.t[arg0:argf]
            data = self.data[arg0:argf]
            stim = self.stim[arg0:argf]
        
        if reset_time:
            t = t - t[0]
            
        return self.new(t, data, stim)
    
    def subsample(self, dt=None, n_sample=None):
        
        if not(dt is None):
            n_sample = get_arg(dt, self.dt)
            
        t = self.t[::n_sample]
        data = self.data[::n_sample]
        stim = self.stim[::n_sample]
            
        return self.new(t, data, stim)

# =============================================================================
# Spikes methods
# =============================================================================
    
    def mask_spikes(self, t0=None, tf=None, thr=-13., time_window=3, 
                    use_derivative=False, dvdt_threshold=9, t_before_spike_peak=0, tref=4):
        # 24/09/2018
        # updated 23/04/2019 to tref and so it's equal to GLMFitter
        
        if t0 is None:
            t0 = self.t[0]
        
        if tf is None:
            tf = self.t[-1] + self.dt
        
        arg0, argf = get_arg([t0, tf], self.dt)
        arg_order = get_arg(time_window, self.dt)
        arg_before_spk = get_arg(t_before_spike_peak, self.dt)
        argref = get_arg(tref, self.dt)
        
        mask0 = np.zeros(self.shape, dtype = bool)
        maskf = np.zeros(self.shape, dtype = bool)
        mask0[arg0:,...] = True
        maskf[:argf,...] = True
        
        mask_spk_peak = self.data > thr
        mask_spk_peak = mask_extrema(self.data, mask_data=mask_spk_peak, order=arg_order, left_comparator=np.greater, right_comparator=np.greater_equal)
        # greater_equal is necessary to the rights because of the discrete values of sampling. 
        # one spike could have its peak formed by two values of the same size. the function will return only the first one
        
        if use_derivative:

            mask_dvdt_thres_crossings = mask_thres_crossings(self.get_dvdt(), dvdt_threshold)
            
            # when they cross the threshold and come t_before_spike_peak ms before a spike peak
            mask_spk_dvdt = mask_dvdt_thres_crossings & extend_trues(mask_spk_peak, arg_before_spk, 0)
            # previous thing shouldnt find more than one value in each. I give back the first
            mask_spk_dvdt = mask_spk_dvdt & ~extend_trues(shift_array(mask_spk_dvdt, 1, fill_value=False), 0, arg_before_spk)        
            # if for any spike no value is found for the derivative I get the value given by t_before_spk
            mask_spk = mask_spk_dvdt | shift_array(mask_spk_peak, -arg_before_spk, fill_value=False)
            mask_spk = mask_spk & ~extend_trues(shift_array(mask_spk, -1, fill_value=False), argref, 0)
            
        else:
            
            mask_spk = shift_array(mask_spk_peak, -arg_before_spk, fill_value=False)
            # there can be no two spikes separated by less than tref. I give back the first
            mask_spk = mask_spk & ~extend_trues(shift_array(mask_spk, 1, fill_value=False), 0, argref)        
        
        return mask_spk & mask0 & maskf
    
    def mask_away_from_spikes(self, tl, tr, t0=0, tf=None, mask_spikes_kwargs={}):
        
        if tf is None:
            tf = self.t[-1] + self.dt

#         searchsorted(self.t, [t0, tf])
        arg0, argf = searchsorted(self.t, [t0, tf])
        argl, argr = get_arg([tl, tr], self.dt)

        mask = mask_args_away_from_maskinp(self.mask_spikes(**mask_spikes_kwargs), argl, argr, arg0=arg0, argf=argf)
        
        return mask
    
    def get_SpikeTrain(self, **kwargs):
        return SpikeTrain(self.t, self.mask_spikes(**kwargs))
    
#    def t_spikes(self, **kwargs):
#        mask_spk = self.mask_spikes(**kwargs)
#        t_spk = [self.t[mask_spk[:,sw]] for sw in range(self.nsweeps)]
#        return t_spk
    
    @property
    def spiking_sweeps(self):
        mask_spk = self.mask_spikes()
        return np.where( np.any( mask_spk , 0) )[0]
    
    @property
    def non_spiking_sweeps(self):
        mask_spk = self.mask_spikes()
        return np.where( ~np.any( mask_spk , 0) )[0]
    
    def mask_spiking_sweeps(self):
        mask_spk = self.mask_spikes()
        return np.any(mask_spk, 0)
    
    def get_spike_count(self, time_bins, average_sweeps=False, mask_spikes_kwargs=None):
        # 08/09/2018
        # Given arbitrary time bins computes the spike counts in those bins for each sweep
        # unless average=True
        
        if mask_spikes_kwargs is None:
            mask_spikes_kwargs = {}
            
        arg_bins = searchsorted(self.t, time_bins)
        
        spk_count = np.zeros((len(arg_bins) - 1, self.nsweeps))
        
        mask_spk = self.mask_spikes(**mask_spikes_kwargs)
        
        for sw in range(self.nsweeps):
            
            arg_spk = np.where(mask_spk[:, sw])[0]
            
            spk_count[:, sw], _ = np.histogram(arg_spk, bins=arg_bins)
        
        if average_sweeps:
            spk_count = np.mean(spk_count, 1)
            if spk_count.size==1:
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
        rate = rate / np.diff(time_bins)[:,None] * 1000
        
        return rate
    
    def fano_factor(self, bins, mask_spikes_kwargs={}):
        
        spk_count = self.get_spike_count(bins, average_sweeps = average_sweeps, mask_spikes_kwargs=mask_spikes_kwargs)
        
        return np.var(spk_count, 1) / np.mean(spk_count, 1)

    def get_PSTH(self, kernel, average_sweeps=True, mask_spikes_kwargs={}):
        psth = self.get_SpikeTrain(**mask_spikes_kwargs).get_PSTH(kernel, average_sweeps=average_sweeps)
        return psth
    
    def get_ISI_distribution(self, mask_spikes_kwargs={}):
        # 12/10/2018

        isi_dist = []

        mask_spk = self.mask_spikes(**mask_spikes_kwargs)

        for sw in range(self.nsweeps):

            t_spk = self.t[mask_spk[:, sw]]
            isi_dist += [np.diff(t_spk)]

        return isi_dist
    
    def get_dvdt(self):
        dvdt = np.diff( self.v, axis=0 )/self.dt
        dvdt = np.concatenate((dvdt, dvdt[-1:,...]), axis = 0)
        return dvdt
    
    def get_MSE_matrix(self, mask):
        # 28/09/2018
        mse = np.zeros((self.nsweeps, self.nsweeps))
        T = np.sum(mask)
        
        for sw1, sw2 in itertools.combinations(range(self.nsweeps), 2):
            
            mse[sw1, sw2] = 1./T * np.sum( (self.data[mask, sw1] - self.data[mask, sw2])**2. )
            mse[sw2, sw1] = mse[sw1, sw2]
        
        return mse
    
    def get_RMSE(self, mask):
        triu_indices = np.triu_indices(self.nsweeps, k=1)
        return np.sqrt(np.mean(self.get_MSE_matrix(mask)[triu_indices] ) )
    
    #=============================================================================
    #
    #=============================================================================
    
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
            argh = get_arg(th, self.dt)
        
        vh = np.mean(self.data[:argh, ...], axis=0)
        Ih = np.mean(self.stim[:argh, ...], axis=0)
        
        return vh, Ih
        
    def get_vrst(self, tref = 4.5, **kwargs):

        t0 = kwargs.get('t0', 0.)
        tf = kwargs.get('tf', self.t[-1] + self.dt)
        
        arg_ref = get_arg(tref, self.dt)
        
        mask_spk = self.mask_spikes(t0 = t0, tf = tf)
        arg_spk = np.where( mask_spk )
        
        arg_spike_ref = ( arg_spk[0] + arg_ref  , arg_spk[1] )
        
        vrst = np.mean( self.v[arg_spike_ref] )
        
        return vrst
    
    def get_vr(self, t0, tf, **kwargs):
        sweeps = kwargs.get('sweeps', range(self.nsweeps))
        
        t_sweeps = np.ones( (self.t.shape[0], len(sweeps)) ) * self.t[:,None]
        
        mask_vr = (t_sweeps<t0) | (t_sweeps>tf)
        
        vr = np.mean(self.v[mask_vr] )
        return vr
    
    def get_KernelRect(self, tbins, tl = None, tr = None, **kwargs):
        
        from KernelRect import KernelRect
        
        kappa = KernelRect(tbins)
            
        t0 = kwargs.get('t0', 0.)
        tf = kwargs.get('tf', self.t[-1] + self.dt)
        arg0, argf = get_arg([t0, tf], self.dt)
        
        th = kwargs.get('th', len(self.t)/64) * self.dt
        vh, Ih = self.get_vhIh(th = th)
        
        if tl is None or tr is None:
            mask = None
        else:
            mask = self.mask_away_from_spikes(tl, tr, t0 = t0, tf = tf)
                         
        R2, rmse = kappa.deconvolve_continuous(self.t, self.stim - Ih, self.v - vh, mask = mask, R2=False, rmse=True)
        
        return kappa, R2, rmse