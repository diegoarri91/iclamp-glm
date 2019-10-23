import numpy as np
from scipy.linalg import cholesky_banded, solveh_banded

from .utils.linalg import diag_indices, band_matrix, unband_matrix
from .utils.time import get_arg, get_dt


class OUProcess:
    """
    Class implementing an Ornstein Uhlenbeck process
    
    Parameters
    ----------
    mu : float
        Mean of the OU process
    sd : float
        Standard deviation of the OU process
    tau : float
        Time scale of the OU process
    """
    def __init__(self, mu=0, sd=1, tau=3):
        self.mu = mu
        self.sd = sd
        self.tau = tau

    def sample(self, t, shape=(1,), seed=None):
        """
        Produces samples of Ornstein Uhlenbeck process
        
        Parameters
        ----------
        t : 1d array-like
            time points
        shape : tuple of ints, optional
            output shape of sample is (t, shape)
        seed : int, optional
            Random seed used to initialize np.random.seed
        Returns
        ----------
        eta : ndarray
            Ornstein Uhlenbeck process samples
        """
        np.random.seed(seed)
        
        dt = get_dt(t)
        
        eta = np.zeros((len(t),) + shape) * np.nan
        
        eta[0] = self.mu + self.sd * np.random.randn(*shape)

        for j in range(len(t)-1):
            eta[j+1] = (eta[j] - self.mu) * np.exp(-dt / self.tau) + self.mu + self.sd * np.sqrt(1 - np.exp(-2 * dt / self.tau)) * np.random.randn(*shape)
            
        return eta
    
    def sample2(self, t, shape=(1,), seed=None):
        
        np.random.seed(seed)
        
        dt = get_dt(t)
        
        eta = np.zeros((len(t),) + shape) * np.nan
        
        eta[0] = self.mu + self.sd * np.random.randn(*shape)

        for j in range(len(t)-1):
            eta[j+1, :] = eta[j, :] + (self.mu - eta[j, :]) / self.tau * dt + ( 2. * dt / self.tau )**.5 * self.sd * np.random.randn(*shape)
            
        return eta

    def get_log_prior_kwargs(self, t):
        dt = get_dt(t)
        log_prior_kwargs = dict(dt=dt)
        return log_prior_kwargs
    
    def g_log_prior(self, eta, dt):

        lam = np.exp(-dt / self.tau)
        _mu = self.mu * (1 - np.exp(-dt / self.tau))
        _sd2 = (1 - np.exp(-2 * dt / self.tau)) * self.sd ** 2

        log_prior = -1 / (2 * _sd2) * np.sum((eta[1:] - lam * eta[:-1] - _mu) ** 2) + \
                    -1 / (2 * self.sd ** 2.) * (eta[0] - self.mu) ** 2

        g_log_prior = np.zeros(len(eta))
        g_log_prior[0] = lam / _sd2 * (eta[1] - _mu - lam * eta[0]) - 1 / (self.sd ** 2.) * (eta[0] - self.mu)
        g_log_prior[1:-1] = 1 / _sd2 * ((1 - lam) * _mu - (1 + lam**2.) * eta[1:-1] + lam * (eta[2:] + eta[:-2]))
        g_log_prior[-1] = 1. / _sd2 * (_mu - eta[-1] + lam * eta[-2] )
        
        return log_prior, g_log_prior
    
    def h_log_prior(self, eta, dt, banded=True):
            
        lam = np.exp(-dt / self.tau)
        _sd2 = (1 - np.exp(-2 * dt / self.tau)) * self.sd ** 2
        
        if banded:
            h_log_prior = np.zeros((2, len(eta)))
            h_log_prior[0, 0] = lam ** 2 + _sd2 / self.sd ** 2
            h_log_prior[0, -1] = 1
            h_log_prior[0, 1:-1] = 1 + lam ** 2
            h_log_prior[1, :] = -lam
            h_log_prior = -h_log_prior / _sd2
            
        else:
            h_log_prior = np.zeros((len(eta), len(eta)))
            h_log_prior[0, 0] = lam ** 2 + _sd2 / self.sd ** 2
            # TODO FIX
            # diag_rows, diag_rows = diag_indices(len(eta), k=0)
            # diag_rows, diag_rows = diag_rows[1:-1], diag_rows[1:-1]
            # h_log_prior[(diag_rows, cols)] = 1 + lam ** 2
            h_log_prior[diag_indices(len(eta), k=1)] = -lam
            h_log_prior[diag_indices(len(eta), k=-1)] = -lam
            h_log_prior[-1, -1] = 1
            h_log_prior = -h_log_prior / _sd2
            
        return h_log_prior
    
    def cov(self, eta, dt, banded=True):
        return solveh_banded(-self.h_log_prior(eta, dt, banded=True), np.eye(len(eta)), lower=True)
    
    def log_det_prior_covariance(self, eta, dt):
        ch = cholesky_banded(-self.h_log_prior(eta, dt, banded=True), lower=True)
        return -2 * np.sum(np.log(ch[0, :]))

class AutoCorr:

    def __init__(self, mu=None, sd=None, autocorr_fun=None, pars=None):
        self.mu = mu
        self.sd = sd
        self.autocorr_fun = autocorr_fun
        self.pars = pars
        self.inv_cov = None
        self.inv_cov_banded = None
        self.ch_lower = None

    def set_prior_on_t(self, t, t_max_band_cov=None, eps_max_band_inv_cov=1e-6, inv_cov=True, cholesky=False):

        dt = get_dt(t)
        t_max_band_cov = t_max_band_cov if t_max_band_cov is not None else t[-1]
        max_band = min(get_arg(t_max_band_cov, dt, func='ceil'), len(t))
        cov = np.zeros((max_band, len(t)))
        for v in range(max_band):
            cov[v, :len(t) - v] = self.sd**2 * self.autocorr_fun(t[v], **self.pars)

        if inv_cov:
            self.inv_cov = solveh_banded(cov, np.eye(len(t)), lower=True)
            self.inv_cov_banded = band_matrix(self.inv_cov, max_band=max_band)
            max_band_inv_cov = np.where(np.all(np.abs(self.inv_cov_banded) < eps_max_band_inv_cov, axis=1))[0][0]
            self.inv_cov_banded = self.inv_cov_banded[:max_band_inv_cov, :]

        if cholesky:
            ch = cholesky_banded(cov, lower=True)
            self.ch_lower = unband_matrix(ch, symmetric=False, lower=True)

        return self

    def sample(self, t, shape=(1,), seed=None):
        np.random.seed(seed)
        xi = self.ch_lower @ np.random.randn(self.ch_lower.shape[0], *shape) + self.mu
        return xi

    def get_log_prior_kwargs(self, t):
        return {}

    def g_log_prior(self, xi):
        g_log_prior = -self.inv_cov @ (xi - self.mu)
        log_prior = 1 / 2 * (xi - self.mu) @ g_log_prior
        return log_prior, g_log_prior

    def h_log_prior(self, xi):
        return -self.inv_cov_banded

class GeneralCov:

    def __init__(self, mu=None, cov=None):
        self.mu = mu
        self.cov = cov
        self.inv_cov = None
        self.inv_cov_banded = None
        self.ch_lower = None

    def set_prior_on_t(self, t, eps_max_band_inv_cov=1e-6, inv_cov=True, cholesky=False):

        if inv_cov:
            self.inv_cov = solveh_banded(cov, np.eye(len(t)), lower=True)
            self.inv_cov_banded = band_matrix(self.inv_cov, max_band=max_band)
            max_band_inv_cov = np.where(np.all(np.abs(self.inv_cov_banded) < eps_max_band_inv_cov, axis=1))[0][0]
            self.inv_cov_banded = self.inv_cov_banded[:max_band_inv_cov, :]

        if cholesky:
            ch = cholesky_banded(cov, lower=True)
            self.ch_lower = unband_matrix(ch, symmetric=False).T

        return self

    def sample(self, shape=(1,), seed=None):
        np.random.seed(seed)
        xi = self.ch_lower @ np.random.randn(self.ch_lower.shape[0], *shape) + self.mu
        return xi

    def g_log_prior(self, xi):
        g_log_prior = -self.inv_cov @ (xi - self.mu)
        log_prior = 1 / 2 * (xi - self.mu) @ g_log_prior
        return log_prior, g_log_prior

    def h_log_prior(self, xi):
        return -self.inv_cov_banded

