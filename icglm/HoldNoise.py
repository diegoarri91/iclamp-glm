import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/diego/Dropbox/hold_noise/iclamp-glm/")

from icglm.iclamp import IClamp

from fun_signals import get_arg, mask_thres_sustained_crossings, shift_array, extend_trues

class HoldNoise(IClamp):
    
#     def __init__(self, t, v, stim=None, **metadata):
#         super().__init__(t, v, stim, **metadata)
    
    def get_current_mu_std(self, average_sweeps = False):
        arg0, argf = self.arg0(), self.argf()
        mu, std = np.mean(self.stim[arg0+1:argf+1,:], 0), np.std(self.stim[arg0+1:argf+1,:], 0)
        if average_sweeps:
            mu, std = np.mean(mu), np.mean(std)
        return mu, std
    
    def arg0(self, th=500.):
        
        argh = get_arg(th, self.dt)
        mu_noise, std_noise = np.mean(self.stim[:argh,:], 0), np.std(self.stim[:argh,:], 0)
        
        arg0 = []
        mu_noise, std_noise = np.mean(self.stim[:argh,:], 0), np.std(self.stim[:argh,:], 0)

        for sw in range(self.nsweeps):
            aux = np.where( mask_thres_sustained_crossings(np.abs(self.stim[:,sw] - mu_noise[sw]), 3.*std_noise[sw], 5, upcrossing = True) )[0]
            
            if len(aux)>0:
                arg0 += [aux[0]]
        
        return int(np.median(arg0))
    
    def argf(self, th=500.):
        
        argh = get_arg(th, self.dt)
        
        hn = HoldNoise(self.t, self.data[::-1,:], self.stim[::-1,:])
        argf = hn.arg0(th = th)
        
        argf = len(self.t) - argf
        
        return argf
    
    def STA(self, tl, tr, kernel=None, t_ref=None, mask_spikes_kwargs=None):
        
        argl, argr = get_arg([tl, tr], self.dt)
        arg_ref = get_arg(t_ref, self.dt)
        
        mask_spk = self.mask_spikes(t0=tl, **mask_spikes_kwargs) # spikes
        stim = np.copy(self.stim)
        mask_around_spk = extend_trues(shift_array(mask_spk, 1, fill_value=False), 0, arg_ref)
        stim[mask_around_spk] = np.nan
        
        index = np.where(mask_spk)
        sta = [stim[i_spk-argl:i_spk-argr+1, sw] for i_spk, sw in zip(*index)]
        sta = np.stack(sta, 1)
        
        t_sta = np.arange(-argl, -argr + 1, 1) * self.dt
        
        return t_sta, sta
    
    def STA2(self, tl, tr, kernel=None, return_mask=False):
        
        # 4/5/18 (28/01/19 updated. Function was wrong. Didn't return STA correctly)
        # Computes the STA between tl and tr before spike peak
        # spikes that have another spike before them less than tl are thrown away
        
        argl, argr = get_arg([tl, tr], self.dt)
        
        mask_spk = self.mask_spikes(t0=tl) # spikes
        mask_spk = mask_spk & ~extend_trues(shift_array(mask_spk, 1, fill_value=False), 0, argl) # throw spikes for which a spike occurs less than tl before
        
        mask_STA = shift_array(mask_spk, -argr, fill_value=False) # shift argr
        mask_STA = extend_trues(mask_STA, argl - argr, 0) # extend to argl - argr
        
        if kernel is not None:
            stim = kernel.convolve_continuous(self.t, self.stim)
        else:
            stim = np.copy(self.stim)
            
        spike_triggered = stim[mask_STA]
        t_sta = np.arange(-argl, -argr + 1, 1) * self.dt
        
        spike_triggered = np.stack(np.split(spike_triggered, np.sum(mask_spk)), 1)
        
        if return_mask:
            return t_sta, spike_triggered, mask_STA
        else:
            return t_sta, spike_triggered