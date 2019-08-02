import numpy as np

from SpikeTrain import SpikeTrain

from fun_signals import get_dt, extend_trues, shift_array


class SpikeTrainGenerator:

    @classmethod
    def homogeneous_poisson(self, t, nu=None, T=None, trials=1, t0=None, tf=None):

        dt = get_dt(t)
        if t0 is None:
            t0 = t[0]
        if tf is None:
            tf = t[-1]
        
        if 'float' in str(type(T)) or type(T) is int:
            rand = np.random.rand(len(t), trials)
        else:
            rand = np.random.rand(len(t), len(T))

        if nu is not None:
            mask = nu * dt > rand 
        else:
            t_ = np.stack([t]*rand.shape[1], 1)
            mask = (dt / T > rand) & ~(t_ < t0) & (t_ < tf)

        return SpikeTrain(t, mask)
    
    @classmethod
    def homogeneous_poisson_refractory(self, t, nu=None, T=None, trials=1, t0=None, tf=None):

        dt = get_dt(t)
        if t0 is None:
            t0 = t[0]
        if tf is None:
            tf = t[-1]
        
        rand = np.random.rand(len(t), trials)

        if nu is not None:
            mask = nu * dt > rand 
        else:
            t_ = np.stack([t]*rand.shape[1], 1)
            mask = (dt / T > rand) & ~(t_ < t0) & (t_ < tf)

        return SpikeTrain(t, mask)

    @classmethod
    def inhomogeneous_poisson(self, t, r, trials=1):

        dt = get_dt(t)

        rand = np.random.rand(*(r.shape + (trials,)))

        mask = (r[..., None] * dt > rand)

        return SpikeTrain(t, mask)

    @classmethod
    def inhomogeneous_poisson_refractory(self, t, r, trials=1, t0=None, tf=None):
        #TODO
        dt = get_dt(t)
        if t0 is None:
            t0 = t[0]
        if tf is None:
            tf = t[-1]

        rand = np.random.rand(*(r.shape + (trials,)))

        # t_ = np.stack([t] * rand.shape[1], 1)
        mask = (r[..., None] * dt > rand)# & ~(t_ < t0) & (t_ < tf)
        
        # there can be no two spikes separated by less than tref. I give back the last
        if tref > 0:
            argref = int(tref/dt)
            mask = mask & ~extend_trues(shift_array(mask, -1, fill_value=False), argref, 0)

        return SpikeTrain(t, mask)

