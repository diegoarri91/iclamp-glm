import numpy as np

from ..kernels.rect import KernelRect
from .glm import GLM
from .srm import SRM
from ..signals import shift_mask
from ..utils.time import get_dt


class LSSRM(SRM):

    def __init__(self, vr=None, kappa=None, eta=None, vt=None, dv=None, gamma=None, noise='poisson'):
        super().__init__(vr=vr, kappa=kappa, eta=eta, vt=vt, dv=dv, gamma=gamma)
        self.noise = noise

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

    def set_supthreshold_params(self, vt, dv, gamma_coefs):
        self.vt = vt
        self.dv = dv
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
        # print(dt, t[1])
        glm = GLM(kappa=KernelRect([0, dt], [1 / self.dv / dt]), eta=self.gamma.copy(), 
                  u0=self.vt / self.dv, noise=self.noise)
        glm.eta.coefs = glm.eta.coefs / self.dv
        optimizer = glm.fit(t, v_simu, mask_spikes, stim_h=np.mean(v_simu[0]), newton_kwargs=newton_kwargs, verbose=verbose)
        # optimizer = glm.fit(t, v_simu - np.mean(v_simu[0]), mask_spikes, stim_h=0, newton_kwargs=newton_kwargs,
        #                     verbose=verbose)
        self.set_supthreshold_params(glm.u0 / glm.kappa.coefs[0] / dt, 1 / glm.kappa.coefs[0] / dt,
                                     glm.eta.coefs / glm.kappa.coefs[0] / dt)
        return optimizer

    def fit(self, t, stim, mask_spikes, v, mask_subthreshold, stim_h=0, newton_kwargs=None, verbose=False):
        self.fit_subthreshold_voltage(t, stim, v, mask_spikes, mask_subthreshold, stim_h=stim_h)
        optimizer = self.fit_supthreshold(t, stim, mask_spikes, newton_kwargs=newton_kwargs, verbose=verbose)
        return optimizer

    def decode(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        pass