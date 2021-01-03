import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

from ..utils.time import get_dt, searchsorted
from .base import Kernel


class KernelFunSum(Kernel):

    # def __init__(self, fun, key_par, vals_par, shared_kwargs=None, support=None, coefs=None, prior=None, prior_pars=None):
    def __init__(self, fun, basis_kwargs, shared_kwargs=None, support=None, coefs=None, prior=None,
                     prior_pars=None):
        self.fun = fun
        # self.key_par = key_par
        # self.vals_par = vals_par
        self.basis_kwargs = basis_kwargs
        self.shared_kwargs = shared_kwargs if shared_kwargs is not None else {}
        self.support = np.array(support)
        self.coefs = np.array(coefs) if coefs is not None else np.ones(len(vals_par))
        # self.nbasis = len(self.basis_kwargs)
        self.nbasis = len(list(self.basis_kwargs.values())[0])
        super().__init__(prior=prior, prior_pars=prior_pars)

    def copy(self):
        kernel = KernelFunSum(self.fun, basis_kwargs=self.basis_kwargs.copy(),
                              shared_kwargs=self.shared_kwargs.copy(), support=self.support.copy(),
                              coefs=self.coefs.copy(), prior=self.prior, prior_pars=self.prior_pars.copy())
        return kernel

    def area(self, dt):
        return np.sum(self.interpolate(np.arange(self.support[0], self.support[1] + dt, dt))) * dt

    def interpolate(self, t):
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key:vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        return np.sum(self.coefs[None, :] * self.fun(t[:, None], **kwargs), 1)

    def interpolate_basis(self, t):
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        return self.fun(t[:, None], **kwargs)

    def convolve_basis_continuous(self, t, I):
        """# Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values
        Assumes sorted t"""

        dt = get_dt(t)
        arg0, argf = searchsorted(t, self.support)
        X = np.zeros(I.shape + (self.nbasis, ))

        basis_shape = tuple([argf] + [1 for ii in range(I.ndim - 1)] + [self.nbasis])
        # basis = np.zeros(basis_shape)
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}
        basis = self.fun(t[:argf, None], **kwargs).reshape(basis_shape)

        X = fftconvolve(basis, I[..., None], axes=0)
        X = X[:len(t), ...] * dt

        return X

    def convolve_basis_discrete(self, t, s, shape=None):

        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg0, argf = searchsorted(t, self.support)
        # print(argf)

        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )

        X = np.zeros(shape)
        # print(X.shape)
        # kwargs = {self.key_par: self.vals_par[None, :], **self.shared_kwargs}
        kwargs = {**{key: vals[None, :] for key, vals in self.basis_kwargs.items()}, **self.shared_kwargs}

        for ii, arg in enumerate(arg_s):
            # indices = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
            indices = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))] + [slice(0, self.nbasis)])
            #print(indices)
            #print(X[indices].shape)
            # print(arg)
            # print(indices)
            # print(self.fun(t[arg:, None], **kwargs).shape)
            # print(self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]),) + shape[1:]).shape)
            # print(X[indices].shape)
            # aux = self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]),) + shape[1:])
            # print(aux.shape)
            # X[indices] += aux
            #print(self.fun(t[arg:, None, None] - t[arg], **kwargs).shape)
            #print(self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]), self.nbasis)).shape)
            #X[indices] += self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]), ) + shape[1:])
            X[indices] += self.fun(t[arg:, None] - t[arg], **kwargs).reshape((len(t[arg:]), self.nbasis))

        return X

class KernelFun(Kernel):

    def __init__(self, fun=None, pars=None, support=None):
        self.fun = fun
        self.pars = pars
        self.support = np.array(support)
        self.values = None

    def interpolate(self, t):
        return self.fun(t, *self.pars)

    def area(self, dt):
        return np.sum(self.interpolate(np.arange(self.support[0], self.support[1] + dt, dt))) * dt

    @classmethod
    def exponential(cls, tau, A, support=None):
        if support is None:
            support = [0, 10 * tau]
        return cls(fun=lambda t, tau, A: A * np.exp(-t / tau), pars=[tau, A], support=support)

    @classmethod
    def gaussian(cls, tau, A):
        return cls(fun=lambda t, tau, A: A * np.exp(-(t / tau)**2.), pars=[tau, A], support=[-5 * tau, 5 * tau + .1])

    @classmethod
    def gaussian_delta(cls, delta):
        return cls.gaussian(np.sqrt(2.) * delta, 1. / np.sqrt(2. * np.pi * delta ** 2.))
