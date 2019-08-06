import numpy as np

from scipy.signal import savgol_filter

from fun_abfs import load_data, load_protocol

from fun_signals import restrict, butter_filter, get_arg, get_dt


class EPD:
    
    def __init__(self, t=None, data=None, stim=None, path=None, file=None, neuron=None, **kwargs):
        
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
    
    def get_time_from_mask_sweeps(self, mask, concatenate=True):
        
        '''
        Returns an array of times determined by mask handling many sweeps
        Much faster than np.stack(self.t, 1)[mask]. Especially for masks with few Trues (like spikes)
        '''
        
        list_t = [self.t[mask[:, sw]] for sw in range(self.nsweeps)]
        
        if concatenate:
            return np.concatenate(list_t)
        else:
            return list_t
    
    def subtract_baseline(self, t0=0, tf=None):
        arg0, argf = get_arg([t0, tf], self.dt)
        self.data = self.data - np.mean(self.data[arg0:argf], 0)
        return self
    
    def remove_outlier_sweeps(self, b = 1.4826):
        stds = np.sum( (self.data - np.mean(self.data, 1)[:,None] )**2., 0 )/len(self.t)
        abs_dist = np.abs(stds - np.median(stds) )
        sweeps = abs_dist< b*np.median(abs_dist)
        return self.sweeps(sweeps)