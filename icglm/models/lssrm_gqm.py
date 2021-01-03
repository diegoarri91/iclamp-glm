import numpy as np

from ..kernels.rect import KernelRect
from .gqm_kappa import GQMKappa
from ..signals import shift_mask
from ..utils.time import get_dt


class LSSRMGQM:

    def __init__(self, vr=None, kappa=None, eta=None, psi=None, quad_psi=None, vt=None, gamma=None):
        self.vr = vr
        self.kappa = kappa
        self.eta = eta
        self.psi = psi
        self.quad_psi = quad_psi
        self.vt = vt
        self.gamma = gamma

    def simulate_subthreshold(self, t, stim, mask_spikes, stim_h=0., full=False):

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
            mask_spikes = mask_spikes.reshape(len(t), 1)
        else:
            shape = stim.shape

        dt = get_dt(t)
        # if dt is None:
        #     dt = 1
        arg_spikes = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)
        eta_conv = self.eta.convolve_discrete(t, t_spikes, shape=shape[1:])
        gamma_conv = self.gamma.convolve_discrete(t, t_spikes, shape=shape[1:])

        v = kappa_conv - eta_conv + self.vr
        psi_conv = self.psi.convolve_continuous(t, v)
        quad_psi_conv = self.quad_psi.convolve_continuous(t, v)
        r = np.exp(psi_conv + quad_psi_conv - self.vt - gamma_conv)

        if full:
            return kappa_conv, eta_conv, psi_conv, quad_psi_conv, gamma_conv, v, r
        else:
            return v, r

    def fit_subthreshold_voltage(self, t, stim, v, mask_spikes, mask_subthreshold, stim_h=0):

        n_kappa, n_eta = self.kappa.nbasis, self.eta.nbasis
        # arg_ref = searchsorted(t, t_ref)

        X = np.zeros((np.sum(mask_subthreshold), 1 + n_kappa + n_eta))
        X_kappa = self.kappa.convolve_basis_continuous(t, stim - stim_h)
        arg_shifted_spikes = np.where(shift_mask(mask_spikes, 1, fill_value=False))
        t_shifted_spikes = (t[arg_shifted_spikes[0]],) + arg_shifted_spikes[1:]
        X_eta = self.eta.convolve_basis_discrete(t, t_shifted_spikes)

        X[:, 0] = 1.
        X[:, 1:n_kappa + 1] = X_kappa[mask_subthreshold, :]
        X[:, n_kappa + 1:] = -X_eta[mask_subthreshold, :]

        theta_sub, _, _, _ = np.linalg.lstsq(X, v[mask_subthreshold], rcond=None)

        self.set_subthreshold_params(theta_sub[0], theta_sub[1:n_kappa + 1], theta_sub[n_kappa + 1:])

        return self

    def set_subthreshold_params(self, vr, kappa_coefs, eta_coefs):
        self.vr = vr
        self.kappa.coefs = kappa_coefs
        self.eta.coefs = eta_coefs
        return self

    def set_supthreshold_params(self, psi_coefs, quad_psi_coefs, vt, gamma_coefs):
        # self.quad_psi.coefs = np.zeros((self.quad_psi.n, self.quad_psi.n))
        # self.quad_psi.coefs[np.triu_indices(self.quad_psi.n)] = quad_psi_coefs
        # self.quad_psi.coefs[np.tril_indices(self.quad_psi.n)] = self.quad_psi.coefs.T[np.tril_indices(self.quad_psi.n)]
        self.psi.coefs = psi_coefs
        from ..kernels.rect2d import KernelRect2d
        self.quad_psi = KernelRect2d(self.quad_psi.tbins_x, self.quad_psi.tbins_y, quad_psi_coefs)
        # self.quad_psi.coefs = quad_psi_coefs
        self.vt = vt
        self.gamma.coefs = gamma_coefs
        return self

    def time_rescale_transform(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import time_rescale_transform
        dt = get_dt(t)
        _, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h)
        z, ks_stats = time_rescale_transform(dt, mask_spikes, r)
        return z, ks_stats

    def fit_supthreshold(self, t, stim, mask_spikes, stim_h=0, newton_kwargs=None, verbose=False):
        v_simu, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h, full=False)
        dt = get_dt(t)
        gqm = GQMKappa(kappa=self.psi.copy(), eta=self.gamma.copy(), quad_kappa=self.quad_psi.copy(),
                  u0=self.vt)
        # gqm.kappa.coefs = gqm.kappa.coefs
        # gqm.quad_kappa.coefs = gqm.quad_kappa.coefs
        # gqm.eta.coefs = gqm.eta.coefs
        optimizer = gqm.fit(t, v_simu, mask_spikes, stim_h=np.mean(v_simu[0]), newton_kwargs=newton_kwargs, verbose=verbose)
        self.set_supthreshold_params(gqm.kappa.coefs, gqm.quad_kappa.coefs, gqm.u0, gqm.eta.coefs)
        return optimizer

    def fit(self, t, stim, mask_spikes, v, mask_subthreshold, stim_h=0, newton_kwargs=None, verbose=False):
        self.fit_subthreshold_voltage(t, stim, v, mask_spikes, mask_subthreshold, stim_h=stim_h)
        optimizer = self.fit_supthreshold(t, stim, mask_spikes, newton_kwargs=newton_kwargs, verbose=verbose)
        return optimizer

    def decode(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        pass

    def sample(self, t, stim, stim_h=0, full=False):

        dt = get_dt(t)

        if stim.ndim == 1:
            shape = (len(t), 1)
            stim = stim.reshape(len(t), 1)
        else:
            shape = stim.shape

        v = np.zeros(shape) * np.nan
        psi_conv = np.zeros(shape) * np.nan
        quad_psi_conv = np.zeros(shape) * np.nan
        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        gamma_conv = np.zeros(shape)
        mask_spikes = np.zeros(shape, dtype=bool)

        kappa_conv = self.kappa.convolve_continuous(t, stim - stim_h) + stim_h * self.kappa.area(dt=dt)

        arg = 40
        j = 0
        while j < len(t):

            v[j, ...] = kappa_conv[j, ...] - eta_conv[j, ...] + self.vr
            if j + 1 - arg >= 0:
                psi_conv[j, ...] = self.psi.convolve_continuous(t[j + 1 - arg:j + 1], v[j + 1 - arg:j + 1, ...])[-1]
                quad_psi_conv[j, ...] = self.quad_psi.convolve_continuous(t[j + 1 - arg:j + 1], v[j + 1 - arg:j + 1, ...])[-1]
            else:
                psi_conv[j, ...] = self.psi.convolve_continuous(t[:j + 1], v[:j + 1, ...])[-1]
                quad_psi_conv[j, ...] = self.quad_psi.convolve_continuous(t[:j + 1], v[:j + 1, ...])[-1]
            r[j, ...] = np.exp(psi_conv[j, ...] + quad_psi_conv[j, ...] - self.vt - gamma_conv[j, ...])

            p_spk = 1. - np.exp(-r[j, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spikes[j, ...] = p_spk > aux

            if np.any(mask_spikes[j, ...]) and j < len(t) - 1:
                eta_conv[j + 1:, mask_spikes[j, ...]] += self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]
                gamma_conv[j + 1:, mask_spikes[j, ...]] += self.gamma.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1

        if full:
            return kappa_conv, eta_conv, quad_psi_conv, gamma_conv, v, r, mask_spikes
        else:
            return v, r, mask_spikes

    def get_log_likelihood(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import log_likelihood_normed
        dt = get_dt(t)
        kappa_conv, eta_conv, psi_conv, quad_psi_conv, gamma_conv, v, r = self.simulate_subthreshold(t, stim, mask_spikes,
                                                                            stim_h=stim_h, full=True)
        u = psi_conv + quad_psi_conv - gamma_conv - self.vt
        # u = np.log(r)
        log_like_normed = log_likelihood_normed(dt, mask_spikes, u, r)
        return log_like_normed