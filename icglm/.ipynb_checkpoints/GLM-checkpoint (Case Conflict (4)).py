import numpy as np

from .kernels import KernelFun
from .signals import get_dt, shift_array


class GLM:

    def __init__(self, r0=None, kappa=None, eta=None):

        self.r0 = r0
        self.kappa = kappa
        self.eta = eta

    def simulate(self, t, I, I0=0.):

        np.seterr(over='ignore')  # Ignore overflow warning when calculating r[j+1] which can be very big if dV small

        r0, kappa = self.r0, self.kappa
        u0 = -np.log(r0)

        dt = get_dt(t)

        if I.ndim == 1:
            shape = (len(t), 1)
            I = I.reshape(len(t), 1)
        else:
            shape = I.shape

        # rmax = 3./dt

        r = np.zeros(shape) * np.nan
        eta = np.zeros(shape)
        mask_spk = np.zeros(shape, dtype=bool)

        kappa_conv = self.kappa.convolve_continuous(t, I - I0) + I0 * self.kappa.area(dt=dt)

        j = 0

        while j < len(t):

            r[j, ...] = np.exp(kappa_conv[j, ...] - eta[j, ...] - u0)
            #r[j, r[j, ...] > rmax] = rmax

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            #p_spk = r[j, ...] * dt
            aux = np.random.rand(*shape[1:])

            mask_spk[j, ...] = p_spk > aux

            if self.eta is not None and np.any(mask_spk[j, ...]) and j < len(t) - 1:
                eta[j + 1:, mask_spk[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1

        return r, kappa_conv, eta, mask_spk

    def simulate_subthr(self, t, I, mask_spk, I0=0., full=False, iterating=False):

        u0 = -np.log(self.r0)

        if I.ndim == 1:
            shape = (len(t), 1)
            I = I.reshape(len(t), 1)
        else:
            shape = I.shape

        arg_spikes = np.where(shift_array(mask_spk, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        dt = get_dt(t)
        # rmax = 3. / dt

        kappa_conv = self.kappa.convolve_continuous(t, I - I0, iterating=iterating) + I0 * self.kappa.area(dt=dt)

        if self.eta is not None and len(t_spikes[0]) > 0:
            eta = self.eta.convolve_discrete(t, t_spikes, shape=mask_spk.shape[1:])
        else:
            eta = np.zeros(shape)

        v = kappa_conv - eta - u0
        r = np.exp(v)
        # r[r>rmax] = rmax

        if full:
            return r, v
        else:
            return r, kappa_conv, eta
        
