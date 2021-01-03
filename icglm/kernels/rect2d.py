import numpy as np
from scipy.fftpack.helper import next_fast_len
from scipy.signal import fftconvolve

from ..masks import shift_mask
from ..utils.linalg import band_matrix, diag_indices
from ..utils.time import get_dt, searchsorted


class KernelRect2d:

    def __init__(self, tbins_x, tbins_y, coefs=None, prior=None, prior_pars=None):
        self.tbins_x = tbins_x
        self.tbins_y = tbins_y
        self.support_x = np.array([tbins_x[0], tbins_x[-1]])
        self.support_y = np.array([tbins_y[0], tbins_y[-1]])
        self.coefs = np.array(coefs)
        # self.banded_coefs = band_matrix(self.coefs, fill_with_nan=True)
        self.n = len(self.tbins_x) - 1
        self.n_coefs = (len(tbins_x) - 1) * len(tbins_x) // 2

        # self.banded_idx = np.ones((self.n, self.n), dtype=int) * -1
        # cum = 0
        # for k in range(self.n):
        #     self.banded_idx[:self.n - k, k] = np.arange(cum, cum + self.n - k, 1)
        #     cum += self.n - k
        # self.idx = np.where(self.banded_idx != -1)[::-1]
        self.prior = prior
        self.prior_pars = prior_pars

    def copy(self):
        kernel = KernelRect2d(self.tbins_x.copy(), self.tbins_y.copy(), coefs=self.coefs.copy(), prior=self.prior,
                              prior_pars=self.prior_pars.copy())
        return kernel

    def interpolate(self, tx, ty, sorted_t=True):

        if sorted_t:
            res = self.interpolate_sorted(tx, ty)
        else:
            tx, ty = np.meshgrid(tx, ty)
            arg_binsx = np.searchsorted(self.tbins_x, tx, side='right') - 1
            arg_binsy = np.searchsorted(self.tbins_y, ty, side='right') - 1

            mask = (arg_binsx >= 0) & (arg_binsy >= 0) & (arg_binsx < len(self.tbins_x) - 1) & (
                        arg_binsy < len(self.tbins_y) - 1)
            idx = np.meshgrid(np.arange(0, tx.shape[0]), np.arange(0, tx.shape[1]))
            res = np.zeros(tx.shape)
            res[(idx[0][mask], idx[1][mask])] = self.coefs[(arg_binsx[mask], arg_binsy[mask])]

        return res

    def interpolate_sorted(self, tx, ty):

        res = np.zeros((len(tx), len(ty)))

        arg_binsx = searchsorted(tx, self.tbins_x, side='left')
        arg_binsy = searchsorted(ty, self.tbins_y, side='left')
        # print(arg_binsx, arg_binsy)

        for ii, (arg0x, argfx) in enumerate(zip(arg_binsx[:-1], arg_binsx[1:])):
            for jj, (arg0y, argfy) in enumerate(zip(arg_binsy[:-1], arg_binsy[1:])):
                res[arg0x:argfx, arg0y:argfy] = self.coefs[ii, jj]

        return res

    def convolve_continuous(self, t, I, sorted_t=True):

        dt = get_dt(t)

        arg0x = int(self.support_x[0] / dt)
        argfx = int(np.ceil(self.support_x[1] / dt))
        arg0y = int(self.support_y[0] / dt)
        argfy = int(np.ceil(self.support_y[1] / dt))

        tx = np.arange(arg0x, argfx + 1, 1) * dt
        ty = np.arange(arg0y, argfy + 1, 1) * dt
        # tx, ty = np.meshgrid(tx, ty)
        kernel_values = self.interpolate(tx, ty, sorted_t=sorted_t)
        kernel_values = kernel_values.reshape(kernel_values.shape + tuple([1] * (I.ndim-1)))
        # print(kernel_values)

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
                # print(arg_s, argsj)
                # print(arg_i, arg_j)
                if arg_i == arg_j:
                    interpolation = np.moveaxis(np.diagonal(self.interpolate(t[arg_j:] - t[arg_i], t[arg_j:] - t[arg_j], sorted_t=True)), -1, 0)
                else:
                    interpolation = 2 * np.moveaxis(np.diagonal(self.interpolate(t[arg_j:] - t[arg_i], t[arg_j:] - t[arg_j], sorted_t=True)), -1, 0)
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

    def gh_log_prior(self, coefs):

        n = self.n

        log_prior = 0
        g_log_prior = np.zeros(self.n_coefs)
        h_log_prior = np.zeros((self.n_coefs, self.n_coefs))

        if 'smooth_1st_derivative_diagonals' == self.prior or 'smooth_1st_derivative_diagonals' in self.prior:

            lam_diag = self.prior_pars['lam_diag']

            banded_coefs = np.zeros((self.n, self.n)) * np.nan
            banded_coefs[self.idx] = coefs
            bulk_idx = self.banded_idx.copy()
            diag_lim_idx = (n - 1 - np.arange(n), np.arange(n))
            diag_lim_idx = (diag_lim_idx[0][1:-1], diag_lim_idx[1][1:-1])
            flat_diag_lim_idx = bulk_idx[diag_lim_idx]

            next_diag = bulk_idx[1:-1, 2:] - bulk_idx[1:-1, 1:-1]
            bulk_idx[diag_lim_idx] = -1
            bulk_idx = bulk_idx[1:-1, 1:-1]
            flat_bulk_idx = bulk_idx.T[bulk_idx != -1]
            flat_next_diag = next_diag.T[bulk_idx != -1]

            log_prior += -lam_diag * 0.5 * np.nansum(np.diff(banded_coefs, axis=1) ** 2)

            g_log_prior[0] += -lam_diag * banded_coefs[0, 0] + lam_diag * banded_coefs[0, 1]
            g_log_prior[1:self.n - 1] += -lam_diag * banded_coefs[1:-1, 0] + lam_diag * banded_coefs[1:-1, 1]
            g_log_prior[self.banded_idx[0, 1:-1]] += lam_diag * banded_coefs[0, :-2] - 2 * lam_diag * banded_coefs[0, 1:-1] + \
                                                     lam_diag * banded_coefs[0, 2:]
            g_log_prior[flat_bulk_idx] += (lam_diag * banded_coefs[1:-1, :-2] - 2 * lam_diag * banded_coefs[1:-1, 1:-1] + \
                                           lam_diag * banded_coefs[1:-1, 2:]).T[bulk_idx != -1]
            g_log_prior[flat_diag_lim_idx] += lam_diag * banded_coefs[(diag_lim_idx[0], diag_lim_idx[1] - 1)] + \
                                             -lam_diag * banded_coefs[diag_lim_idx]
            g_log_prior[n * (n + 1) // 2 - 1] += lam_diag * (banded_coefs[0, -2] - banded_coefs[0, -1])

            h_log_prior[0, 0] += -lam_diag
            h_log_prior[0, n] += lam_diag
            # first col/row except 0,0
            h_log_prior[(self.banded_idx[1:-1, 0], self.banded_idx[1:-1, 0])] += -lam_diag
            h_log_prior[(self.banded_idx[1:-1, 0], self.banded_idx[1:-1, 1])] += lam_diag
            # main diag except 0,0
            h_log_prior[(self.banded_idx[0, 1:-1], self.banded_idx[0, 1:-1])] += -2 * lam_diag
            h_log_prior[(self.banded_idx[0, 1:-1], self.banded_idx[0, 2:])] += lam_diag
            h_log_prior[(self.banded_idx[0, -1], self.banded_idx[0, -1])] += -lam_diag
            # # bulk
            h_log_prior[(flat_bulk_idx, flat_bulk_idx)] += -2 * lam_diag
            h_log_prior[(flat_bulk_idx, flat_bulk_idx + flat_next_diag)] += lam_diag
            h_log_prior[(flat_diag_lim_idx, flat_diag_lim_idx)] += -lam_diag

        if self.prior == 'smooth_1st_derivative_columns' or 'smooth_1st_derivative_columns' in self.prior:
            matrix_coefs = np.zeros((self.n, self.n)) * np.nan
            matrix_coefs[np.triu_indices(self.n)] = coefs
            matrix_idx = np.ones((self.n, self.n), dtype=int) * -1
            matrix_idx[np.triu_indices(self.n)] = np.arange(self.n_coefs)
            # print(matrix_idx)
            if isinstance(self.prior_pars, dict):
                lam_col = self.prior_pars['lam_col']
            else:
                lam_col = self.prior_pars[-1]

            log_prior += -lam_col * 0.5 * np.nansum(np.diff(matrix_coefs, axis=0) ** 2) + \
                         -lam_col * 0.5 * np.nansum(np.diff(matrix_coefs, axis=1) ** 2)

            g_log_prior[0] += -lam_col * matrix_coefs[0, 0] + lam_col * matrix_coefs[0, 1]
            g_log_prior[1:self.n - 1] += lam_col * matrix_coefs[0, :-2] - 3 * lam_col * matrix_coefs[0, 1:-1] + \
                                        lam_col * matrix_coefs[0, 2:] + lam_col * matrix_coefs[1, 1:-1]
            g_log_prior[self.n - 1] += lam_col * matrix_coefs[0, self.n - 2] - 2 * lam_col * matrix_coefs[0, self.n - 1] + \
                                      lam_col * matrix_coefs[1, self.n - 1]
            diag_idx = np.diag_indices(self.n)
            diag_idx = (diag_idx[0][1:-1], diag_idx[1][1:-1])
            g_log_prior[matrix_idx[diag_idx]] += -2 * lam_col * np.diagonal(matrix_coefs)[1:-1] + \
                                     lam_col * np.diagonal(matrix_coefs, offset=1)[:-1] + \
                                     lam_col * np.diagonal(matrix_coefs, offset=1)[1:]

            bulk_idx = matrix_idx[1:-1, 1:-1].copy()
            bulk_idx[np.diag_indices(bulk_idx.shape[0])] = -1
            bulk_idx = bulk_idx.reshape(-1)
            bulk = (lam_col * matrix_coefs[:-2, 1:-1] + lam_col * matrix_coefs[1:-1, :-2] + \
                    -4 * lam_col * matrix_coefs[1:-1, 1:-1] + lam_col * matrix_coefs[1:-1, 2:] + \
                    + lam_col * matrix_coefs[2:, 1:-1]).reshape(-1)
            bulk = bulk[bulk_idx != -1]
            bulk_idx = bulk_idx[bulk_idx != -1]
            g_log_prior[bulk_idx] += bulk

            g_log_prior[matrix_idx[1:-1, -1]] += lam_col * matrix_coefs[:-2, -1] + lam_col * matrix_coefs[1:-1, -2] + \
                                             -3 * lam_col * matrix_coefs[1:-1, -1] + lam_col * matrix_coefs[2:, -1]
            g_log_prior[self.n_coefs - 1] += -lam_col * (matrix_coefs[-1, -1] - matrix_coefs[-2, -1])

            h_log_prior[0, 0] += -lam_col
            h_log_prior[0, 1] += lam_col
            # # first row except 0,0
            h_log_prior[(matrix_idx[0, 1:-1], matrix_idx[0, 1:-1])] += -3 * lam_col
            h_log_prior[(matrix_idx[0, 1:-1], matrix_idx[0, 2:])] += lam_col
            h_log_prior[(matrix_idx[0, 1:-1], matrix_idx[1, 1:-1])] += lam_col
            # last element of first row
            h_log_prior[matrix_idx[0, -1], matrix_idx[0, -1]] += -2 * lam_col
            h_log_prior[matrix_idx[0, -1], matrix_idx[1, -1]] += lam_col
            # #main diag except 0,0
            h_log_prior[(matrix_idx[diag_idx], matrix_idx[diag_idx])] += -2 * lam_col
            diag_off = diag_indices(self.n, k=1)
            # print(matrix_idx[diag_idx], matrix_idx[diag_off][1:])
            h_log_prior[(matrix_idx[diag_idx], matrix_idx[diag_off][1:])] += lam_col
            # # bulk
            bulk_idx = matrix_idx[1:-1, 1:-1].copy()
            bulk_idx[np.diag_indices(bulk_idx.shape[0])] = -1
            # print(bulk_idx)
            bulk_idx = bulk_idx.reshape(-1)
            next_col = matrix_idx[1:-1, 2:].reshape(-1)[bulk_idx != -1]
            next_row = matrix_idx[2:, 1:-1].reshape(-1)[bulk_idx != -1]
            bulk_idx = bulk_idx[bulk_idx != -1]
            # print(matrix_idx[2:, 1:-1])
            # print(bulk_idx, next_row)
            h_log_prior[(bulk_idx, bulk_idx)] += -4 * lam_col
            h_log_prior[(bulk_idx, next_col)] += lam_col
            h_log_prior[(bulk_idx, next_row)] += lam_col
            # last column
            h_log_prior[(matrix_idx[1:-1, -1], matrix_idx[1:-1, -1])] += -3 * lam_col
            h_log_prior[(matrix_idx[1:-1, -1], matrix_idx[2:, -1])] += lam_col
            # right bottom corner
            h_log_prior[(matrix_idx[-1, -1], matrix_idx[-1, -1])] += -lam_col

        if self.prior == 'L2' or 'L2' in self.prior:

            if isinstance(self.prior_pars, dict):
                lam_l2 = self.prior_pars['lam_l2']
            else:
                lam_l2 = self.prior_pars[0]
                # print('holi')

            log_prior += -lam_l2 * 0.5 * np.sum(coefs ** 2)
            g_log_prior += -lam_l2 * coefs
            h_log_prior += -lam_l2

        if self.prior == 'smooth_1st_derivative':
            lam_col, lam_diag = self.prior_pars[0], self.prior_pars[1]

            log_prior = -lam_diag * np.nansum(np.diff(banded_coefs, axis=1) ** 2) \
                        -lam_col * np.nansum(np.diff(banded_coefs, axis=0) ** 2)

            g_log_prior = np.zeros(self.n_coefs)
            g_log_prior[0] = -(lam_col + lam_diag) * banded_coefs[0, 0] + lam_col * banded_coefs[1, 0] + lam_diag * banded_coefs[0, 1]
            g_log_prior[1:self.n - 1] = lam_col * banded_coefs[:-2, 0] - (lam_col + lam_diag) * banded_coefs[1:-1, 0] + \
                               lam_col * banded_coefs[2:, 0] + lam_diag * banded_coefs[1:-1, 1]
            g_log_prior[self.n - 1] = lam_col * banded_coefs[self.n - 2, 0] - lam_col * banded_coefs[self.n - 1, 0]
            g_log_prior[self.banded_idx[0, 1:-1]] = lam_diag * banded_coefs[0, :-2] -(lam_col + 2 * lam_diag) * banded_coefs[0, 1:-1] + \
                             lam_col * banded_coefs[1, 1:-1] + lam_diag * banded_coefs[0, 2:]
            g_log_prior[n * (n + 1) // 2 - 1] = lam_diag * (banded_coefs[0, -2] - banded_coefs[0, -1])
            g_log_prior[flat_bulk_idx] = (-2 * lam_diag * (banded_coefs[1:-1, :-2] - banded_coefs[1:-1, 1:-1]) \
                                   -2 * lam_diag * (banded_coefs[1:-1, 2:] - banded_coefs[1:-1, 1:-1]) \
                                   -2 * lam_col * (banded_coefs[:-2, 1:-1] - banded_coefs[1:-1, 1:-1]) \
                                   -2 * lam_col * (banded_coefs[2:, 1:-1] - banded_coefs[1:-1, 1:-1]))[bulk_idx != -1]
            g_log_prior[flat_diag_lim_idx] = lam_diag * banded_coefs[(diag_lim_idx[0], diag_lim_idx[1] - 1)] + lam_col * banded_coefs[(diag_lim_idx[0] - 1, diag_lim_idx[1])] \
                                 -(lam_col + lam_diag) * banded_coefs[diag_lim_idx]

            h_log_prior = np.zeros((self.n_coefs, self.n_coefs))
            # h_log_prior[0, 0], h_log_prior[0, 1], h_log_prior[0, n] = -(lam_col + lam_diag), lam_col, lam_diag
            # # first row except 0,0
            # h_log_prior[(self.banded_idx[1:-1, 0], self.banded_idx[1:-1, 0])] = -(2 * lam_col + lam_diag)
            # h_log_prior[(self.banded_idx[1:-1, 0], self.banded_idx[1:-1, 0] + 1)] = lam_col
            # h_log_prior[(self.banded_idx[1:-1, 0], self.banded_idx[1:-1, 1])] = lam_diag
            # #main diag except 0,0
            # h_log_prior[(self.banded_idx[0, 1:-1], self.banded_idx[0, 1:-1])] = -(lam_col + 2 * lam_diag)
            # h_log_prior[(self.banded_idx[0, 1:-1], self.banded_idx[0, 1:-1] + 1)] = lam_col
            # h_log_prior[(self.banded_idx[0, 1:-1] + 1, self.banded_idx[0, 1:-1] + 1)] = lam_diag
            # h_log_prior[(self.banded_idx[0, :-1], self.banded_idx[0, :-1])] = -lam_diag
            # # bulk
            # h_log_prior[(diag, diag)] = -2 * (lam_col + lam_diag)
            # h_log_prior[(diag, diag + 1)] = lam_col
            # h_log_prior[(diag[:-1], diag[1:])] = lam_diag
            #
            # h_log_prior[self.banded_idx[lim]] = -lam_col
            #
            # h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[
            #     np.tril_indices_from(h_log_prior, k=-1)]

        h_log_prior[np.tril_indices_from(h_log_prior, k=-1)] = h_log_prior.T[np.tril_indices_from(h_log_prior, k=-1)]

        return log_prior, g_log_prior, h_log_prior