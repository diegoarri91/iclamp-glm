import pickle
import numpy as np

from ..signals import shift_mask
from ..utils.time import get_dt


class GIF:

    def __init__(self, vr, tau, eta, vt, dv, gamma):
        self.vr = vr
        self.tau = tau
        self.eta = eta
        self.vt = vt
        self.dv = dv
        self.gamma = gamma

    def sample(self, t, stim, stim_h=0, full=False):

        dt = get_dt(t)

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        else:
            shape = stim.shape

        v = np.zeros(shape) * np.nan
        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        gamma_conv = np.zeros(shape)
        mask_spikes = np.zeros(shape, dtype=bool)

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)

        j = 0
        while j < len(t):

            v[j, ...] = kappa_conv[j, ...] - eta_conv[j, ...] + self.vr
            r[j, ...] = np.exp((v[j, ...] - self.vt - gamma_conv[j, ...]) / self.dv)

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spikes[j, ...] = p_spk > aux

            if np.any(mask_spikes[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spikes[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]
                gamma_conv[j + 1:, mask_spikes[j, ...]] += self.gamma.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1

        if full:
            return kappa_conv, eta_conv, gamma_conv, v, r, mask_spikes
        else:
            return v, r, mask_spikes

    def get_log_likelihood(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import log_likelihood_normed
        dt = get_dt(t)
        kappa_conv, eta_conv, gamma_conv, v, r = self.simulate_subthreshold(t, stim, mask_spikes,
                                                                            stim_h=stim_h, full=True)
        u = (v - self.vt - gamma_conv) / self.dv
        log_like_normed = log_likelihood_normed(dt, mask_spikes, u, r)
        return log_like_normed

    def set_params(self, theta):
        n_kappa = self.kappa.nbasis
        n_eta = self.eta.nbasis
        self.vr = theta[0]
        self.kappa.coefs = theta[1:n_kappa + 1]
        self.eta.coefs = theta[n_kappa + 1:n_kappa + 1 + n_eta]
        self.vt = theta[n_kappa + 1 + n_eta] / theta[-1]
        self.gamma.coefs = theta[n_kappa + 2 + n_eta:-1] / theta[-1]
        self.dv = 1 / theta[-1]
        return self

    def simulate_subthreshold(self, t, stim, mask_spikes, stim_h=0., full=False):

        dt = get_dt(t)
        arg_spikes = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        gamma_conv = self.gamma.convolve_discrete(t, t_spikes, shape=shape[1:])

        v = kappa_conv - eta_conv + self.vr
        r = np.exp((v - self.vt - gamma_conv) / self.dv)

        if full:
            return kappa_conv, eta_conv, gamma_conv, v, r
        else:
            return v, r
