import matplotlib.pyplot as plt
import numpy as np

from models.glm import GLM
from signals import shift_mask
from utils.time import get_dt, searchsorted


class TwoStepsSRM:

    def __init__(self, vr=None, kappa=None, eta=None, u0=None, psi=None, gamma=None):

        self.vr = vr
        self.kappa = kappa
        self.eta = eta

        # self.u0 = u0
        # self.psi = psi
        # self.gamma = gamma
        self.glm = GLM(kappa=psi, eta=gamma, u0=u0)

        # self.tref = tref
        # self.vt0, self.dV = vt0, dV
        # self.r0 = r0
        # self.vpeak = vpeak

    def copy(self):
        kappa = self.kappa.copy()
        eta = self.eta.copy()
        psi = self.psi.copy()
        gamma = self.gamma.copy()
        srm = SRM(vr=self.vr, kappa=kappa, eta=eta, u0=self.u0, psi=psi, gamma=gamma)
        return srm

    @property
    def u0(self):
        return self.glm.u0

    @property
    def psi(self):
        return self.glm.kappa

    @property
    def gamma(self):
        return self.glm.eta

    @property
    def log_prior_iterations(self):
        return self.glm.log_prior_iterations

    @property
    def log_posterior_iterations(self):
        return self.glm.log_posterior_iterations

    @property
    def theta_iterations(self):
        return self.glm.theta_iterations

    @property
    def fit_status(self):
        return self.glm.fit_status

    def set_params(self, vr=None, kappa_coefs=None, eta_coefs=None, u0=None, psi_coefs=None, gamma_coefs=None):
        self.vr = vr
        self.kappa.coefs = kappa_coefs
        self.eta.coefs = eta_coefs
        self.u0 = u0
        self.psi.coefs = psi_coefs
        self.gamma.coefs = gamma_coefs
        return self

    def fit_subthreshold_voltage(self, t, stim, v, mask_spikes, mask_subthreshold, Ih=0):

        n_kappa, n_eta = self.kappa.nbasis, self.eta.nbasis
        # arg_ref = searchsorted(t, t_ref)

        X = np.zeros((np.sum(mask_subthreshold), 1 + n_kappa + n_eta))
        X_kappa = self.kappa.convolve_basis_continuous(t, stim - Ih)
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

    def simulate_subthreshold(self, t, I, mask_spk, I0=0., rate=True):

        if I.ndim == 1:
            shape = (len(t), 1)
            I = I.reshape(len(t), 1)
        else:
            shape = I.shape

        arg_spikes = np.where(shift_mask(mask_spk, 1, fill_value=False))
        t_spikes = (t[arg_spikes[0]], arg_spikes[1])

        dt = get_dt(t)

        kappa_conv = self.kappa.convolve_continuous(t, I - I0) + I0 * self.kappa.area(dt=dt)

        if self.eta is not None and len(t_spikes[0]) > 0:
            eta = self.eta.convolve_discrete(t, t_spikes, shape=mask_spk.shape[1:])
        else:
            eta = np.zeros(shape)

        v = kappa_conv - eta + self.vr

        if rate:
            psi_conv = self.psi.convolve_continuous(t, v)
            if self.eta is not None and len(t_spikes[0]) > 0:
                gamma_conv = self.eta.convolve_discrete(t, t_spikes, shape=mask_spk.shape[1:])
            else:
                gamma_conv = np.zeros(shape)
            r = np.exp(psi_conv - gamma_conv - self.u0)
            return r, v
        else:
            return v

    def fit_supthreshold(self, t, stim, mask_spikes, theta0=None, newton_kwargs=None, verbose=False):
        v_simu = self.simulate_subthreshold(t, stim, mask_spikes, rate=False)
        self.glm.fit(t, v_simu, mask_spikes, theta0=theta0, newton_kwargs=newton_kwargs, verbose=verbose)
        return self

    def fit(self, t, stim, v, mask_spikes, mask_subthreshold, theta0=None, newton_kwargs=None, verbose=False):
        self.fit_subthreshold_voltage(t, stim, v, mask_spikes, mask_subthreshold)
        self.fit_supthreshold(t, stim, mask_spikes, theta0=theta0, newton_kwargs=newton_kwargs, verbose=verbose)

    def plot_filters(self, axs=None):

        if axs is None:
            fig, (ax_kappa, ax_eta, ax_psi, ax_gamma) = plt.subplots(figsize=(15, 5), ncols=4)
            ax_kappa.set_xlabel('time')
            ax_kappa.set_ylabel('membrane')
            ax_eta.set_xlabel('time')
            ax_eta.set_ylabel('post-spike voltage')
            ax_psi.set_xlabel('time')
            ax_psi.set_ylabel('voltage')
            ax_gamma.set_xlabel('time')
            ax_gamma.set_ylabel('post-spike threshold')
            ax_kappa.spines['right'].set_visible(False)
            ax_kappa.spines['top'].set_visible(False)
            ax_eta.spines['right'].set_visible(False)
            ax_eta.spines['top'].set_visible(False)
            ax_psi.spines['right'].set_visible(False)
            ax_psi.spines['top'].set_visible(False)
            ax_gamma.spines['right'].set_visible(False)
            ax_gamma.spines['top'].set_visible(False)
        else:
            fig = None
            ax_kappa = axs[0]
            ax_eta = axs[1]
            ax_psi = axs[2]
            ax_gamma = axs[3]

        t_kappa = np.arange(0., self.kappa.tbins[-1], .1)
        ax_kappa = self.kappa.plot(t_kappa, ax=ax_kappa, invert_t=True)
        t_eta = np.arange(self.eta.tbins[0], self.eta.tbins[-1], .1)
        ax_eta = self.eta.plot(t_eta, ax=ax_eta)
        t_psi = np.arange(0., self.psi.tbins[-1], .1)
        ax_psi = self.psi.plot(t_psi, ax=ax_psi)
        t_gamma = np.arange(0., self.gamma.tbins[-1], .1)
        ax_gamma = self.gamma.plot(t_gamma, ax=ax_gamma)

        return fig, (ax_kappa, ax_eta, ax_psi, ax_gamma)

    def fit_subthreshold_voltage2(self, th=500., t0=0., tf=None, tl=0., fit_kwargs={}, rmser=False, rmser_kwargs={}, plot=False):

        self.th = th
        self.t0_subthr = t0
        self.tf_subthr = tf if not (tf is None) else self.ic.t[-1] + self.ic.dt
        self.tl_subthr = tl
        mask_sub = self.get_mask_v_subthr(t0=self.t0_subthr, tf=self.tf_subthr, tl=self.tl_subthr)

        self._fit_vsubth(mask_sub, **fit_kwargs)
        self.simulate_v_subthr()
        self.set_v_subthr_R2()

        if rmser:
            self.set_RMSER(**rmser_kwargs)

        if plot:
            return self.plot_v_subthr()

    @property
    def R(self):
        return self.kappa.area()

    def sample(self, t, I, full=False):

        # Ignore overflow warning when calculating r[j+1] which can be very big if dV small
        np.seterr(over='ignore')

        dt = get_dt(t)

        if I.ndim == 1:
            I = I.reshape(len(t), 1)
            shape = (len(t), 1)
        else:
            shape = I.shape

        v = self.vr + self.kappa.convolve_continuous(t, I)
        r = np.zeros(shape) * np.nan
        eta_conv = np.zeros(shape)
        psi_conv = np.zeros(shape)
        argf_psi = searchsorted(t, self.psi.support[1])
        gamma_conv = np.zeros(shape)
        mask_spk = np.zeros(shape, dtype=bool)

        j = 0

        while j < len(t):

            # print(self.psi.interpolate(t[max(j - argf_psi, 0):j + 1] - t[max(j - argf_psi, 0)]).shape, v.shape)
            # psi_conv[j, ...] = dt * np.sum(self.psi.interpolate(t[max(j - argf_psi, 0):j + 1] - t[max(j - argf_psi, 0)])[::-1] * v[max(j - argf_psi, 0):j + 1, :])
            arg0 = max(j + 1 - argf_psi, 0)
            psi_conv[j, ...] = dt * np.sum(self.psi.interpolate(t[arg0:j + 1] - t[arg0])[::-1, None] * v[arg0:j + 1, :], 0)
            r[j, ...] = np.exp(psi_conv[j, ...] - gamma_conv[j, ...] - self.u0)
            p_spk = 1. - np.exp(-r[j, ...] * dt)
            rand = np.random.rand(*shape[1:])
            mask_spk[j, ...] = p_spk > rand

            if np.any(mask_spk[j, ...]) and j < len(t) - 1:
                if self.eta is not None:
                    _eta_conv = self.eta.interpolate(t[j + 1:] - t[j + 1])[:, None]
                    eta_conv[j + 1:, mask_spk[j, ...]] += _eta_conv
                    v[j + 1:, mask_spk[j, ...]] -= _eta_conv
                if self.gamma is not None:
                    gamma_conv[j + 1:, mask_spk[j, ...]] += self.gamma.interpolate(t[j + 1:] - t[j + 1])[:, None]

            j += 1
        if full:
            return v, r, eta_conv, psi_conv, gamma_conv, mask_spk
        else:
            return v, r, mask_spk
