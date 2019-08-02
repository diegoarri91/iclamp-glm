import time
import numpy as np

from fun_signals import get_dt, shift_array

from KernelFun import KernelFun


class LNP:

    def __init__(self, r0=None, kappa=None):

        self.r0 = r0
        self.kappa = kappa

        self.logL_iterations = None

    def simulate(self, t, I, I0=0.):

        # np.seterr(over='ignore')  # Ignore overflow warning when calculating r[j+1] which can be very big if dV small

        r0, kappa = self.r0, self.kappa
        u0 = -np.log(r0)

        dt = get_dt(t)

        if I.ndim == 1:
            shape = (len(t), 1)
            I = I.reshape(len(t), 1)
        else:
            shape = I.shape

        # rmax = 3./dt

        #kappa_convolution = self.kappa.convolve_continuous(t, I)
#         kappa_convolution = self.kappa.convolve_continuous(t, I - I0)
        kappa_convolution = self.kappa.convolve_continuous(t, I - I0) + I0 * self.kappa.area(dt=dt)

        r = np.exp(kappa_convolution - u0)

        rand = np.random.rand(*shape)
        mask_spk = 1. - np.exp(-r * dt) > rand

        return r, kappa_convolution, mask_spk

    def simulate_subthreshold(self, t, I, I0=0.):

        u0 = -np.log(self.r0)

        if I.ndim == 1:
            shape = (len(t), 1)
            I = I.reshape(len(t), 1)
        else:
            shape = I.shape

        dt = get_dt(t)
        # rmax = 3. / dt

        #kappa_conv = self.kappa.convolve_continuous(t, I) # opcion 1
        kappa_conv = self.kappa.convolve_continuous(t, I - I0) + I0 * self.kappa.area()

        r = np.exp(kappa_conv - u0)
        # r[r>rmax] = rmax

        return r, kappa_conv
        
    def _logL_G_H(self, theta, Y_spikes, Y, dt):

        Yspk_theta = np.dot(Y_spikes, theta)
        Y_theta = np.dot(Y, theta)
        exp_Y_theta = np.exp(Y_theta)

        # Log Likelihood
        L = np.sum(Yspk_theta) - dt * np.sum(exp_Y_theta)

        # Gradient
        G = np.sum(Y_spikes, axis=0) - dt * np.matmul(Y.T, exp_Y_theta)

        # Hessian
        H = - dt * np.dot(Y.T * exp_Y_theta, Y)

        return L, G, H
    
    def _Ymatrix(self, I, s, t):

        n_kappa = self.kappa.nbasis
        Y_kappa = self.kappa.convolve_basis_continuous(t, I)

        Y = np.zeros(I.shape + (n_kappa + 1, ))

        Y[:, :, 0] = -1.
        Y[:, :, 1:n_kappa + 1] = Y_kappa

        Y_spikes, Y = Y[s, :], Y[np.ones(s.shape, dtype=bool), :]

        return Y_spikes, Y
        
    def fit(self, I, s, theta0=None, dt=1, max_iterations=int(2e3), stop_cond=5e-4,
            learning_rate=1e-1, initial_learning_rate=1e-2, verbose=False):
        
        t = np.arange(0, s.shape[0], 1) * dt
        Y_spikes, Y = self._Ymatrix(I, s, t)
        
        _logL, G, H = self._logL_G_H(theta0, Y_spikes, Y, dt)
        theta = theta0
        self.logL_iterations = [_logL]

        status = ''
        old_L = np.nan
        converged = nan_parameters = False
        n_iterations = 0

        if verbose:
            print('Starting gradient ascent \n')

        t0 = time.time()
        for ii in range(max_iterations):

            lr = learning_rate

            if ii <= 10:
                lr = initial_learning_rate

            if verbose and ii % 50 == 0:
                print('\r', 'Iteration {} of {}'.format(ii, max_iterations), 'Elapsed time: {} seconds'.format(time.time() - t0), end='')

            theta = theta - lr * np.dot(np.linalg.inv(H), G)

            _logL, G, H = self._logL_G_H(theta, Y_spikes, Y, dt)
            self.logL_iterations += [_logL]

            if ii > 0 and np.abs((_logL - old_L) / old_L) < stop_cond:
                status += "\n Converged after {} iterations!. ".format(ii + 1)
                converged = True
                nan_parameters = False
                n_iterations = ii + 1
                break
            if np.any(np.isnan(theta)):
                status += "\n There are nan parameters. "
                nan_parameters = True
                converged = False
                n_iterations = ii + 1
                break
            if ii == max_iterations - 1:
                status += '\n Not converged after {} iterations. '.format(max_iterations)
                converged = False
                nan_parameters = False
                n_iterations = ii + 1

            old_L = _logL

        fitting_time = (time.time() - t0) / 60.

        status += 'Elapsed time: {} minutes \n'.format(fitting_time)

        if nan_parameters:
            logL_monotonous = None
        elif np.any(np.diff(self.logL_iterations) < 0.):
            logL_monotonous = False
            status += "Log likelihood is not monotonically increasing \n"
        else:
            logL_monotonous = True
            status += "Log likelihood is monotonous \n"

        if verbose:
            print('\n', status)

        self.fit_status_ = {'n_iterations': n_iterations, 'converged': converged, 'nan_parameters': nan_parameters,
                           'fitting_time': fitting_time, 'logL_monotonous': logL_monotonous, 'status': status}
        
        self.set_params(theta)

        return self

    def set_params(self, theta):

        self.r0 = np.exp(-theta[0])
        self.kappa.coefs = theta[1:]

        # N_spikes = np.sum(self.mask_spikes, 0)
        # N_spikes = N_spikes[N_spikes > 0]
        # logL_poisson = np.sum(
        #     N_spikes * (np.log(N_spikes / len(self.ic.t)) - 1))  # L_poisson = rho0**n_neurons * np.exp(-rho0*T)
        #
        # self.logL_norm, _, _ = self.logL_G_H(theta, self.Y_spikes, self.Y)
        # self.logL_norm = (self.logL_norm - logL_poisson) / np.log(2) / np.sum(N_spikes)
        #
        #
