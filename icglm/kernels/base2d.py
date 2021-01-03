from abc import abstractmethod
import numpy as np
from scipy.fftpack.helper import next_fast_len
from scipy.signal import fftconvolve

from ..masks import shift_mask
from ..utils.linalg import diag_indices
from ..utils.time import get_dt, searchsorted


class Kernel2d:

    def __init__(self, prior=None, prior_pars=None):
        self.prior = prior
        self.prior_pars = np.array(prior_pars)

    @abstractmethod
    def interpolate(self, t):
        pass

    def convolve_continuous(self, t, I):

        dt = get_dt(t)
        arg0x = int(self.support_x[0] / dt)
        argfx = int(np.ceil(self.support_x[1] / dt))
        arg0y = int(self.support_y[0] / dt)
        argfy = int(np.ceil(self.support_y[1] / dt))

        tx = np.arange(arg0x, argfx + 1, 1) * dt
        ty = np.arange(arg0y, argfy + 1, 1) * dt
        # tx, ty = np.meshgrid(tx, ty)
        kernel_values = self.interpolate(tx, ty)
        kernel_values = kernel_values.reshape(kernel_values.shape + tuple([1] * (I.ndim-1)))
        # print(kernel_values.shape)

        conv_rows = fftconvolve(kernel_values, I[None, ...], mode='full', axes=1)[:, :len(t)]  # convolucion sobre columnas para cada fila de C
        # print(I.shape, conv_rows.shape)
        # convolution = np.array([np.sum(I[:u + 1] * conv_rows[:u + 1, u, ...][::-1, ...], 0) for u in range(len(t) - 1)]) * dt ** 2
        n_rows = conv_rows.shape[0]
        convolution = np.zeros(I.shape)
        for u in range(len(t)):
            if u < n_rows:
                convolution[u] = np.sum(I[u::-1] * conv_rows[:u + 1, u, ...], 0)
            elif u < len(t):
                convolution[u] = np.sum(I[u:u - n_rows:-1] * conv_rows[:n_rows, u, ...], 0)
        convolution = np.array(convolution) * dt ** 2
        # convolution = fftconvolve(I.reshape((I.shape[0], 1) + I.shape[1:]), conv_rows, mode='full', axes=0)
        # convolution = convolution[:len(t), ...]
        # convolution = np.moveaxis(np.diagonal(convolution), -1, 0) * dt**2

        return convolution

    def convolve_basis_continuous(self, t, I):

        # Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values

        dt = get_dt(t)
        n_diag = len(self.tbins_x) - 1

        arg_binsx = searchsorted(t, self.tbins_x)
        arg_binsy = searchsorted(t, self.tbins_y)

        basis_shape = tuple([arg_binsx[1], len(t)] + [1 for ii in range(I.ndim - 1)] + [n_diag])
        basis = np.zeros(basis_shape)

        # for k, (i, j) in enumerate(zip(*np.triu_indices(len(self.tbins_x) - 1))):
        for k in range(n_diag):
            arg0x, argfx = arg_binsx[0], arg_binsx[0 + 1]
            arg0y, argfy = arg_binsy[k], arg_binsy[k + 1]
            if k == 0:
                basis[arg0x:argfx, arg0y:argfy, ..., k] = 1.
            else:
                basis[arg0x:argfx, arg0y:argfy, ..., k] = 2.

        conv_rows = fftconvolve(basis, I[None, ..., None], mode='full', axes=1)[:, :len(t)]
        del basis
        n_rows = conv_rows.shape[0]
        convolution = np.zeros(I.shape + (n_diag, ))
        for u in range(len(t)):
            if u < n_rows:
                convolution[u] = np.sum(I[u::-1, ..., None] * conv_rows[:u + 1, u, ...], 0)
            elif u < len(t):
                convolution[u] = np.sum(I[u:u - n_rows:-1, ..., None] * conv_rows[:n_rows, u, ...], 0)
        convolution = np.array(convolution) * dt ** 2
        del conv_rows
        # conv_diags = np.moveaxis(np.diagonal(conv_diags), -1, 0) * dt ** 2

        n_independent_coefs = (len(self.tbins_x) - 1) * len(self.tbins_x) // 2
        X = np.zeros(I.shape + (n_independent_coefs,))
        for k, (i, j) in enumerate(zip(*np.triu_indices(len(self.tbins_x) - 1))):
            X[arg_binsx[i]:, ..., k] = convolution[:len(t) - arg_binsx[i], ..., j - i]

        return X

    def convolve_basis_continuous2(self, t, I):

        # Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values

        dt = get_dt(t)
        n_independent_coefs = (len(self.tbins_x) - 1) * len(self.tbins_x) // 2

        arg_binsx = searchsorted(t, self.tbins_x)
        arg_binsy = searchsorted(t, self.tbins_y)

        basis_shape = tuple([len(t), len(t)] + [1 for ii in range(I.ndim - 1)] + [n_independent_coefs])
        basis = np.zeros(basis_shape)

        # for k, (arg0x, argfx, arg0y, argfy) in enumerate(zip(arg_binsx[:-1], arg_binsx[1:], arg_binsy[:-1],
        #                                                      arg_binsy[1:])):
        for k, (i, j) in enumerate(zip(*np.triu_indices(len(self.tbins_x) - 1))):
            arg0x, argfx = arg_binsx[i], arg_binsx[i + 1]
            arg0y, argfy = arg_binsy[j], arg_binsy[j + 1]
            if i == j:
                basis[arg0x:argfx, arg0y:argfy, ..., k] = 1.
            else:
                basis[arg0x:argfx, arg0y:argfy, ..., k] = 2.

        conv_rows = fftconvolve(basis, I[None, ..., None], mode='full', axes=1)[:, :len(t)]
        del basis
        X = fftconvolve(I.reshape((I.shape[0], 1) + I.shape[1:] + (1,)), conv_rows, mode='full', axes=0)[:len(t), :]
        X = np.moveaxis(np.diagonal(X), -1, 0) * dt**2

        return X

    def convolve_basis_continuous3(self, t, I):

        # Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values

        # dt = get_dt(t)
        n_independent_coefs = (len(self.tbins_x) - 1) * len(self.tbins_x) // 2
        shape = (len(self.tbins_x) - 1, len(self.tbins_x) - 1)
        X = np.zeros(I.shape + (n_independent_coefs,))

        for k, (i, j) in enumerate(zip(*np.triu_indices(len(self.tbins_x) - 1))):
            _coefs = np.zeros(shape)
            _coefs[i, j] = 1
            X[..., k] = KernelRect2d(tbins_x=self.tbins_x, tbins_y=self.tbins_y, coefs=_coefs).convolve_continuous(t, I)
            if i != j:
                X[..., k] = 2 * X[..., k]

        return X

    def deconvolve_continuous(self, t, I, v):

        mask = np.ones(I.shape, dtype=bool)
        X = self.convolve_basis_continuous(t, I)
        X = X[mask, :]
        v = v[mask]
        shape = (len(self.tbins_x) - 1, len(self.tbins_x) - 1)

        coefs = np.linalg.lstsq(X, v, rcond=None)[0]
        self.coefs = np.zeros(shape)
        self.coefs[np.triu_indices(len(self.tbins_x) - 1)] = coefs
        self.coefs[np.tril_indices(len(self.tbins_x) - 1)] = self.coefs.T[np.tril_indices(len(self.tbins_x) - 1)]

    def convolve_discrete(self, t, s, shape=None, single_spike_contribution=False):

        if type(s) is not tuple:
            s = (s,)

        if shape is None:
            shape = tuple([max(s[dim]) + 1 for dim in range(1, len(s))])

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)

        convolution = np.zeros((len(t),) + shape)

        for ii, arg_i in enumerate(arg_s):
            if single_spike_contribution:
                argsj = arg_s[ii:]
            else:
                argsj = arg_s[ii + 1:]
            for jj, arg_j in enumerate(argsj):
                interpolation = np.moveaxis(np.diagonal(self.interpolate(t[arg_i:] - t[arg_i], t[arg_j:] - t[arg_i])), -1, 0)
                # print(t[arg_i:].shape, t[arg_j:].shape, interpolation.shape)
                index = tuple([slice(arg_j, None)] + [s[dim][ii] for dim in range(1, len(s))])
                # index = tuple([slice(max(arg_i, arg_j), None)] + [s[dim][ii] for dim in range(1, len(s))])
                convolution[index] += interpolation

        return convolution

    def convolve_basis_discrete(self, t, s, shape=None, single_spike_contribution=False):

        if isinstance(s, np.ndarray):
            s = (s,)

        triu_mat = np.zeros((len(self.tbins_x) - 1, len(self.tbins_x) - 1), dtype=int)
        for k, (i, j) in enumerate(zip(*np.triu_indices(len(self.tbins_x) - 1))):
            triu_mat[i, j] = k

        arg_s = searchsorted(t, s[0], side='left')
        arg_s = np.atleast_1d(arg_s)
        arg_bins_tbins_in_t = searchsorted(t, self.tbins_x)
        arg_bins = np.searchsorted(self.tbins_x, t, side='right') - 1
        n_independent_coefs = (len(self.tbins_x) - 1) * len(self.tbins_x) // 2

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [n_independent_coefs])
        else:
            shape = shape + (n_independent_coefs,)

        X = np.zeros(shape)

        for ii, arg_i in enumerate(arg_s):
            if single_spike_contribution:
                argsj = arg_s[ii:]
            else:
                argsj = arg_s[ii + 1:]
            for jj, arg_j in enumerate(argsj):

                diag_offset = arg_bins[arg_j - arg_i]
                idx_basis =triu_mat[diag_indices(len(self.tbins_x) - 1, k=diag_offset)]
                for l, k in enumerate(idx_basis):
                    arg0, argf = arg_bins_tbins_in_t[l:l + 2]
                    indices = tuple([slice(arg_j + arg0, arg_j + argf)] + [s[dim][ii] for dim in range(1, len(s))] + [k])
                    X[indices] += 1.

        return X

    def deconvolve_discrete(self, t, s, v):

        mask = np.ones(v.shape, dtype=bool)
        X = self.convolve_basis_discrete(t, s, shape=v.shape)
        X = X[mask, :]
        v = v[mask]
        shape = (len(self.tbins_x) - 1, len(self.tbins_x) - 1)

        coefs = np.linalg.lstsq(X, v, rcond=None)[0]
        self.coefs = np.zeros(shape)
        self.coefs[np.triu_indices(len(self.tbins_x) - 1)] = coefs
        self.coefs[np.tril_indices(len(self.tbins_x) - 1)] = self.coefs.T[np.tril_indices(len(self.tbins_x) - 1)]