import numpy as np

from ..kernels import KernelRect
from .glm import GLM
from .srm import SRM
from ..signals import shift_mask
from ..utils.time import get_dt


class LSSRM(SRM):

    def __init__(self, vr=None, kappa=None, eta=None, vt=None, dv=None, gamma=None):
        super().__init__(vr=vr, kappa=kappa, eta=eta, vt=vt, dv=dv, gamma=gamma)

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

    def get_log_likelihood(self, t, stim, mask_spikes, stim_h=0):
        from ..metrics.spikes import log_likelihood_normed
        dt = get_dt(t)
        v, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h)
        log_like_normed = log_likelihood_normed(dt, mask_spikes, v, r)
        return log_like_normed

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
        v_simu, r = self.simulate_subthreshold(t, stim, mask_spikes, stim_h=stim_h, full=True)
        dt = get_dt(t)
        glm = GLM(kappa=KernelRect([0, dt], [1 / self.dv]), eta=self.gamma.copy(), u0=self.vt / self.dv)
        glm.eta.coefs = glm.eta.coefs / self.dv
        optimizer, log_likelihood_normed = glm.fit(t, v_simu, mask_spikes, stim_h=v_simu[0], newton_kwargs=newton_kwargs, verbose=verbose)
        self.set_supthreshold_params(glm.u0 / glm.kappa.coefs[0], 1 / glm.kappa.coefs[0], glm.eta.coefs / glm.kappa.coefs[0])
        return optimizer, log_likelihood_normed

    def fit(self, t, stim, mask_spikes, v, mask_subthreshold, stim_h=0, newton_kwargs=None, verbose=False):
        self.fit_subthreshold_voltage(t, stim, v, mask_spikes, mask_subthreshold, stim_h=stim_h)
        optimizer, log_likelihood_normed = self.fit_supthreshold(t, stim, mask_spikes, newton_kwargs=newton_kwargs, verbose=verbose)
        return optimizer, log_likelihood_normed

    def decode(self, t, mask_spikes, stim0=None, mu_stim=0, sd_stim=1, stim_h=0, prior=None, newton_kwargs=None,
               verbose=False):
        pass