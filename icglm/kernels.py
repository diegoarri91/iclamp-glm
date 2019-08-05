import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, convolve

from .signals import get_dt, searchsorted

class Kernel:

    def interpolate(self, t):
        pass

    def plot(self, t=None, ax=None, **kwargs):

        if t is None:
            dt = .1
            t = np.arange(self.support[0], self.support[1] + dt, dt)

        if ax is None:
            figsize = kwargs.get('figsize', (8, 5) )
            fig, ax = plt.subplots(figsize = figsize)

        y = self.interpolate(t)
        ax.plot(t, y, **kwargs)

        return ax
    
    def plot_lin_log(self, t=None, axs=None, **kwargs):

        if t is None:
            dt = .1
            t = np.arange(self.support[0], self.support[1] + dt, dt)

        if axs is None:
            figsize = kwargs.get('figsize', (12, 5) );
            fig, axs = plt.subplots(figsize = figsize, ncols = 3);
            
        y = self.interpolate(t)
        axs[0].plot(t, y)
        axs[1].plot(t, y)
        axs[1].set_yscale('log')
        axs[2].plot(t, y)
        axs[2].set_xscale('log'); axs[2].set_yscale('log')
        
        return axs
    
    def set_values(self, dt, ndim):
        arg0 = int(self.support[0] / dt)
        argf = int(np.ceil(self.support[1] / dt))

        t_support = np.arange(arg0, argf + 1, 1) * dt
        t_shape = (len(t_support), ) + tuple([1] * (ndim-1))
        self.values = self.interpolate(t_support).reshape(t_shape)
    
    def convolve_continuous(self, t, I, mode='fft', iterating=False):
        
        # Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns convolution,
        # the convolution of the kernel with axis 0 of I for all other axis values
        # so that convolution.shape = I.shape
        
        dt = get_dt(t)

        arg0 = int(self.support[0] / dt)
        argf = int(np.ceil(self.support[1] / dt))

        if iterating or 'KernelVals' in str(self.__class__):
            # need previous setting of self.values using self.set_values()
            kernel_values = self.values
        else:
            t_support = np.arange(arg0, argf + 1, 1) * dt
            kernel_values = self.interpolate(t_support)
        
        shape = (kernel_values.shape[0], ) + tuple([1] * (I.ndim-1))
        kernel_values = kernel_values.reshape(shape)

        convolution = np.zeros(I.shape)
        
        if mode == 'fft':
        
            full_convolution = fftconvolve(kernel_values, I, mode='full', axes=0)
            # ME COSTO UN HUEVO LOGRAR LAS DOS LINEAS A CONTINUACION Y PARECE FUNCIONAR. NO TOCAR
#            print(arg0, argf)

            if arg0 < 0 and argf > 0 and arg0 + argf - 1 >= 0:
                convolution[arg0 + argf - 1:, ...] = full_convolution[argf - 1:len(t) - arg0, ...]
            elif arg0 >= 0 and argf > 0:
                convolution[arg0:, ...] = full_convolution[:len(t) - arg0, ...]
            elif arg0 < 0:
                #convolution[:len(t) + argf - 1, ...] = full_convolution[-argf + 1:len(t), ...]
                convolution[:len(t) + arg0 + 1, ...] = full_convolution[-arg0:len(t) + 1, ...]
            else:
                return None
            # NO TOCAR A MENOS QUE ESTE MUY SEGURO
                
        convolution *= dt
        
        return convolution

    def correlate_continuous(self, t, I, mode='fft', iterating=False):
        return self.convolve_continuous(t, I[::-1], mode=mode, iterating=iterating)[::-1]

    def convolve_discrete(self, t, s, A=None, shape=None):
        
        # Given a 1d-array t and a tuple of 1d-arrays s=(tjs, shape) containing timings in the
        # first 1darray of the tuple returns the convolution of the kernels with the timings
        # the convolution of the kernel with the timings. conv.ndim = s.ndim and
        # conv.shape = (len(t), max of the array(accross each dimension))
        # A is used as convolution weights. A=(A) with len(A)=len(s[0]).
        # Assumes kernel is only defined on t >= 0
        
        if type(s) is not tuple:
            s = (s,)
            
        if A is None:
            A = (1. for ii in range(s[0].size)) # Instead of creating the whole list/array in memory I use a generator
            #print([1. for ii in range(s[0].size)])
        #elif type(A) is not tuple:
            #A = (A,)

        if shape is None:
            # max(s[dim]) determines the size of each dimension
            shape = tuple([max(s[dim]) + 1 for dim in range(1, len(s))])
            
        #print(t.shape, s[0].shape)
        arg_s = searchsorted(t, s[0])
        #print(arg_s.shape)
        arg_s = np.atleast_1d(arg_s)
        #print(arg_s.shape, arg_s)

        convolution = np.zeros((len(t), ) + shape)

        for ii, (arg, A) in enumerate(zip(arg_s, A)):

            index = tuple([slice(arg, None)] + [s[dim][ii] for dim in range(1, len(s))])
            convolution[index] += A * self.interpolate(t[arg:] - t[arg])
                
        return convolution

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
        return cls.gaussian(np.sqrt(2.) * delta, 1. / np.sqrt(2. * np.pi * delta ** 2.) )

class KernelRect(Kernel):
    
    def __init__(self, tbins, coefs=None):
        self.nbasis = len(tbins) - 1
        self.tbins = np.array(tbins)
        self.support = np.array([tbins[0], tbins[-1]])
        self.coefs = np.array(coefs)
        
    def interpolate(self, t):

        t = np.atleast_1d(t)
        res = np.zeros(len(t))
        arg_bins = searchsorted(t, self.tbins)

        for ii, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
            res[arg0:argf] = self.coefs[ii]

        return res
    
    def area(self, dt=None):
        return np.sum(self.coefs * np.diff(self.tbins))
    
    def plot_basis(self, t, ax=None):
        
        if ax is None:
            fig, ax = plt.subplots()
        
        arg_bins = searchsorted(t, self.tbins)
        
        for k, (arg0, argf) in enumerate(zip( arg_bins[:-1], arg_bins[1:] )):
            vals = np.zeros( (len(t)) )
            vals[arg0:argf] = 1.
            ax.plot(t, vals, linewidth = 5. - 4.*k/(len(arg_bins)-1.) )

        return ax
    
    @classmethod
    def kistler_kernels(cls, delta, dt):
        kernel1 = cls(np.array([-delta, delta + dt]), [1.])
        kernel2 = cls(np.array([0., dt]), [1./dt])
        return kernel1, kernel2
    
    ########################################################################
    # DECONVOLVE CONTINUOUS SIGNAL
    ########################################################################

    def convolve_basis_continuous(self, t, I, method='fft'):
# TODO FIX THIS USING DECONVOLVE SO FITTING WORKS
        # Given a 1d-array t and an nd-array I with I.shape=(len(t),...) returns X,
        # the convolution matrix of each rectangular function of the base with axis 0 of I for all other axis values
        # so that X.shape = (I.shape, nbasis)
        # Discrete convolution can be achieved by using an I with 1/dt on the correct timing values

        dt = get_dt(t)

        arg_bins = searchsorted(t, self.tbins)

        X = np.zeros(I.shape + (self.nbasis, ))

        if method == 'fft':

            basis_shape = tuple([len(t)] + [1 for ii in range(I.ndim - 1)] + [self.nbasis])
            basis = np.zeros(basis_shape)

            for k, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
                basis[arg0:argf, ..., k] = 1.

            X = fftconvolve(basis, I[..., None])
            X = X[:len(t), ...] * dt

        return X
    
    def deconvolve_continuous(self, t, I, v, method='fft', mask=None):
        
        if mask is None:
            mask = np.ones(I.shape, dtype = bool)
        
        X = self.convolve_basis_continuous(t, I, method=method)
        X = X[mask,:]
        v = v[mask]
        
        self.coefs = np.linalg.solve(X, v)
   
    def convolve_basis_discrete(self, t, s, shape=None):
    
        if type(s) is np.ndarray:
            s = (s,)

        arg_s = searchsorted(t, s[0])
        arg_s = np.atleast_1d(arg_s)
        arg_bins = searchsorted(t, self.tbins)
        
        if shape is None:
            shape = tuple([len(t)] + [max(s[dim]) + 1 for dim in range(1, len(s))] + [self.nbasis])
        else:
            shape = shape + (self.nbasis, )
            
        X = np.zeros(shape)
        
        for ii, arg in enumerate(arg_s):
            for k, (arg0, argf) in enumerate(zip(arg_bins[:-1], arg_bins[1:])):
                
                indices = tuple([slice(arg+arg0, arg+argf)] + [s[dim][ii] for dim in range(1, len(s))] + [k])
                
                X[indices] += 1.
        
        return X
    
class KernelVals(Kernel):

    def __init__(self, values=None, support=None):
        self.values = values
        self.support = np.array(support)
