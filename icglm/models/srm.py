import pickle
import numpy as np

from .glm import GLM
from ..decoding import BayesianDecoder
from ..kernels.base import KernelValues
from ..signals import shift_mask
from ..utils.time import get_dt


class SRM(BayesianDecoder):

    def __init__(self, vr=None, kappa=None, eta=None, vt=None, dv=None, gamma=None):
        self.vr = vr
        self.kappa = kappa
        self.eta = eta
        self.vt = vt
        self.dv = dv
        self.gamma = gamma

    def copy(self):
        kappa = self.kappa.copy()
        eta = self.eta.copy()
        gamma = self.gamma.copy()
        srm = self.__class__(vr=self.vr, kappa=kappa, eta=eta, vt=self.vt, dv=self.dv, gamma=gamma)
        return srm

    def save(self, path):
        params = dict(vr=self.vr, kappa=self.kappa, eta=self.eta, vt=self.vt, dv=self.dv, gamma=self.gamma)
        with open(path, "wb") as fit_file:
            pickle.dump(params, fit_file)

    @classmethod
    def load(cls, path):

        with open(path, "rb") as fit_file:
            params = pickle.load(fit_file)
        srm = cls(vr=params['vr'], kappa=params['kappa'], eta=params['eta'], vt=params['vt'], dv=params['dv'],
                  gamma=params['gamma'])
        return srm

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

        u = (v - self.vt - gamma_conv) / self.dv
        if full:
            return kappa_conv, eta_conv, gamma_conv, v, r, mask_spikes
        else:
            return v, u, r, mask_spikes

    def get_log_likelihood(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import log_likelihood_normed
        dt = get_dt(t)
        _, u, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h)
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

        # if stim.ndim == 1:
        #     shape = (len(t), 1)
        #     stim = stim.reshape(len(t), 1)
        #     mask_spikes = mask_spikes.reshape(len(t), 1)
        # else:
        #     shape = stim.shape

        if stim.ndim == 1:
            stim = stim.reshape(len(t), 1)
        if mask_spikes.ndim == 1:
            mask_spikes = mask_spikes.reshape(len(t), 1)

        shape = mask_spikes.shape
        dt = get_dt(t)
        arg_spikes = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        gamma_conv = self.gamma.convolve_discrete(t, t_spikes, shape=shape[1:])

        v = kappa_conv - eta_conv + self.vr
        u = (v - self.vt - gamma_conv) / self.dv
        r = np.exp(u)

        if full:
            return kappa_conv, eta_conv, gamma_conv, u, r
        else:
            return v, u, r

    def get_glm(self):
        u0 = -(self.vr - self.vt) / self.dv
        kappa = self.kappa.copy()
        kappa.coefs = kappa.coefs / self.dv
        t0_eta = np.min([self.eta.tbins[0], self.gamma.tbins[0]])
        tf_eta = np.max([self.eta.tbins[-1], self.gamma.tbins[-1]])
        dt_eta = np.min(np.concatenate((np.diff(self.eta.tbins), np.diff(self.gamma.tbins))))
        t_eta = np.arange(t0_eta, tf_eta + dt_eta, dt_eta)
        from ..kernels.rect import KernelRect
        eta = KernelRect(tbins=t_eta, coefs=(self.eta.interpolate(t_eta[:-1]) + self.gamma.interpolate(t_eta[:-1])) / self.dv)
        glm = GLM(kappa=kappa, eta=eta, u0=u0)
        return glm

    def sum_convolution_kappa_t_spikes(self, t, mask_spikes, sd_stim=1):

        inverted_t = -t[::-1]
        arg_spikes = np.where(mask_spikes)
        minus_t_spikes = (-t[arg_spikes[0]], arg_spikes[1])
        sum_conv_kappa_t_spikes = sd_stim / self.dv * self.kappa.convolve_discrete(inverted_t, minus_t_spikes,
                                                                            shape=(mask_spikes.shape[1],))

        sum_conv_kappa_t_spikes = np.sum(sum_conv_kappa_t_spikes[::-1, :], 1)

        return sum_conv_kappa_t_spikes

    def build_K(self, dt):

        K = []
        t_support = self.kappa.support
        kappa_vals = self.kappa.interpolate(np.arange(0, t_support[1], dt))
        arg_support = int(t_support[1] / dt)
        for v in range(arg_support):
            K.append(KernelValues(values=kappa_vals[v:] * kappa_vals[:len(kappa_vals) - v],
                                  support=[v * dt, t_support[1]]))

        return K

    def get_max_band(self, dt):
        return int(self.kappa.support[1] / dt)

    def gh_log_likelihood_stim(self, stim, t, mask_spikes, sum_convolution_kappa_t_spikes, K, max_band, mu_stim=0,
                               sd_stim=1, stim_h=0):

        dt = get_dt(t)
        v_noise_factor = np.exp(1 / (2 * self.lam * self.dv**2))

        I = sd_stim * stim + mu_stim
        _, u, r = self.simulate_subthreshold(t, I, mask_spikes, stim_h=stim_h)

        log_likelihood = np.sum(u[mask_spikes]) - dt * v_noise_factor * np.sum(r)

        g_log_likelihood = dt * sum_convolution_kappa_t_spikes - \
                           dt * v_noise_factor * sd_stim / self.dv * self.kappa.correlate_continuous(t, np.sum(r, 1))

        t_support = self.kappa.support
        arg_support = int(t_support[1] / dt)
        h_log_likelihood = np.zeros((max_band, len(t)))
        for v in range(arg_support):
            h_log_likelihood[v, :] += -(dt * sd_stim / self.dv) ** 2 * v_noise_factor * K[v].correlate_continuous(t, np.sum(r, 1))

        return log_likelihood, g_log_likelihood, h_log_likelihood

    def decode_stim_from_spikes(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        dt = get_dt(t)
        self.kappa.set_values(dt)
        self.kappa.fix_values = True
        stim_dec, optimizer = super().decode(t, mask_spikes, stim0=stim0, mu_stim=mu_stim, sd_stim=sd_stim,
                                             stim_h=stim_h, prior=prior, newton_kwargs=newton_kwargs, verbose=verbose)
        self.kappa.fix_values = False
        return stim_dec, optimizer
