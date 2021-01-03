import unittest

import numpy as np
import matplotlib.pyplot as plt

from icglm.kernels.rect import KernelRect
from icglm.masks import shift_mask


class TestDeltaKernelRect(unittest.TestCase):

    def test_delta_kernel_is_KernelRect_with_symmetric_half_dt(self):

        dt = .1

        delta = KernelRect(tbins=np.array([-dt / 2, dt / 2]), coefs=[1. / dt])

        t = np.arange(-100., 100., dt)
        arg = int(len(t) / 2)

        y = delta.interpolate(t)

        self.assertAlmostEqual(y[arg], 1. / dt, msg='Delta is not 1/dt in t=0')
        self.assertTrue(np.allclose(np.concatenate((y[:arg], y[arg + 1:])), 0.), msg='Delta is not 0 away from t=0')

    def test_delta_kernel_is_KernelRect_with_0_dt(self):

        dt = .1

        delta = KernelRect(tbins=np.array([0., dt]), coefs=[1. / dt])

        t = np.arange(-100., 100., dt)
        arg = int(len(t) / 2)

        y = delta.interpolate(t)

        self.assertAlmostEqual(y[arg], 1. / dt, msg='Delta is not 1/dt in t=0')
        self.assertTrue(np.allclose(np.concatenate((y[:arg], y[arg + 1:])), 0.), msg='Delta is not 0 away from t=0')

    def test_delta_kernels_in_0_returns_original_signal(self):

        dt = .1
        t = np.arange(0, 100, dt)
        signal = np.random.randn(len(t), 10)

        delta0 = KernelRect(tbins=np.array([-dt / 2, dt / 2]), coefs=[1. / dt])
        convolution0 = delta0.convolve_continuous(t, signal)

        delta0_2 = KernelRect(tbins=np.array([0, dt]), coefs=[1. / dt])
        convolution0_2 = delta0_2.convolve_continuous(t, signal)

        try:
            self.assertTrue(np.allclose(convolution0, signal))
            self.assertTrue(np.allclose(convolution0_2, signal))
        except:
            plt.figure()
            plt.plot(t, signal[:, 0])
            plt.plot(t, convolution0[:, 0])
            plt.show()

    def test_delta_kernel_dt_2dt_shifts_signal_1_value_right(self):

        dt = .1
        t = np.arange(0, 100, dt)
        signal = np.random.randn(len(t), 10)

        delta_right = KernelRect(tbins=np.array([dt, 2. * dt]), coefs=[1. / dt])

        convolution_right = delta_right.convolve_continuous(t, signal)

        self.assertTrue(np.allclose(convolution_right[1:], signal[:-1]), msg='wrong right shift with delta kernel')

    def test_delta_kernel_minusdt_0_shifts_signal_1_value_left(self):

        dt = .1
        t = np.arange(0, 100, dt)
        signal = np.random.randn(len(t), 10)

        delta_left = KernelRect(tbins=np.array([-dt, 0.]), coefs=[1. / dt])

        convolution_left = delta_left.convolve_continuous(t, signal)

        self.assertTrue(np.allclose(convolution_left[:-1], signal[1:]), msg='wrong left shift with delta kernel')

    def test_delta_kernels_shift_signal(self):

        dt = .1
        t = np.arange(0, 100, dt)
        signal = np.random.randn(len(t), 10)

        delta_right = KernelRect(tbins=np.array([-dt / 2, dt / 2]) + 10., coefs=[1. / dt])
        delta_right_2 = KernelRect(tbins=np.array([0, dt]) + 10., coefs=[1. / dt])
        delta_left = KernelRect(tbins=np.array([-dt / 2, dt / 2]) - 10., coefs=[1. / dt])
        delta_left_2 = KernelRect(tbins=np.array([0, dt]) - 10., coefs=[1. / dt])

        arg = int(10. / dt)
        convolution_right = delta_right.convolve_continuous(t, signal)
        convolution_right_2 = delta_right_2.convolve_continuous(t, signal)
        convolution_left = delta_left.convolve_continuous(t, signal)
        convolution_left_2 = delta_left_2.convolve_continuous(t, signal)

        self.assertTrue(np.allclose(convolution_right[arg:], signal[:-arg]), msg='wrong right shift with delta kernel')
        self.assertTrue(np.allclose(convolution_left[:-arg], signal[arg:]), msg='wrong left shift with delta kernel')
        self.assertTrue(np.allclose(convolution_right_2[arg:], signal[:-arg]),
                        msg='wrong right shift with delta kernel')
        self.assertTrue(np.allclose(convolution_left_2[:-arg], signal[arg:]), msg='wrong left shift with delta kernel')


class TestHeavisideKernelRectDeltaSignal(unittest.TestCase):

    def test_kernel_0_2dt_delta_signal(self):

        dt = .1
        t = np.arange(0, 10, dt)
        signal = np.zeros((len(t), 1))
        signal[50, 0] = 1./dt

        kernel = KernelRect(tbins=np.array([0., 2. * dt]), coefs=[1. / dt])

        convolution = kernel.convolve_continuous(t, signal)

        signal_true = np.zeros((len(t), 1))
        signal_true[50:52, 0] = 1./dt

        self.assertTrue(np.allclose(convolution, signal_true))

    def test_kernel_dt_3dt_delta_signal(self):

        dt = .1
        t = np.arange(0, 10, dt)
        signal = np.zeros((len(t), 1))
        signal[50, 0] = 1./dt

        kernel = KernelRect(tbins=np.array([dt, 3. * dt]), coefs=[1. / dt])

        convolution = kernel.convolve_continuous(t, signal)

        signal_true = np.zeros((len(t), 1))
        signal_true[51:53, 0] = 1./dt

        self.assertTrue(np.allclose(convolution, signal_true))

    def test_kernel_minusdt_dt_delta_signal(self):

        dt = .1
        t = np.arange(0, 10, dt)
        signal = np.zeros((len(t), 1))
        signal[50, 0] = 1./dt

        kernel = KernelRect(tbins=np.array([-dt, dt]), coefs=[1. / dt])

        convolution = kernel.convolve_continuous(t, signal)

        signal_true = np.zeros((len(t), 1))
        signal_true[49:51, 0] = 1./dt

        self.assertTrue(np.allclose(convolution, signal_true))

    def test_kernel_minus2dt_0_delta_signal(self):

        dt = .1
        t = np.arange(0, 10, dt)
        signal = np.zeros((len(t), 1))
        signal[50, 0] = 1./dt

        kernel = KernelRect(tbins=np.array([-2. * dt, 0.]), coefs=[1. / dt])

        convolution = kernel.convolve_continuous(t, signal)

        signal_true = np.zeros((len(t), 1))
        signal_true[48:50, 0] = 1./dt

        self.assertTrue(np.allclose(convolution, signal_true))

    def test_kernel_minusdt_2dt_delta_signal(self):

        dt = .1
        t = np.arange(0, 10, dt)
        signal = np.zeros((len(t), 1))
        signal[50, 0] = 1./dt

        kernel = KernelRect(tbins=np.array([-dt, 2. * dt]), coefs=[1. / dt])

        convolution = kernel.convolve_continuous(t, signal)

        signal_true = np.zeros((len(t), 1))
        signal_true[49:52, 0] = 1./dt

        self.assertTrue(np.allclose(convolution, signal_true))

class TestShape(unittest.TestCase):

    def test_convolve_continuous_output_shape(self):

        tbins = np.arange(0, 160, 5)
        coefs = np.exp(-tbins[:-1] / 10.)
        ker = KernelRect(tbins, coefs)

        t = np.arange(0, 200, .1)
        signal = np.random.randn(len(t), 2, 2)

        conv = ker.convolve_continuous(t, signal)

        self.assertTrue(np.all([signal.shape[ii] == conv.shape[ii] for ii in range(signal.ndim)]),
                        msg="signal and output don't have the same shape")

        conv_true = np.zeros((len(t), 2, 2))
        for i in range(signal.shape[1]):
            for j in range(signal.shape[2]):
                conv_true[:, i, j] = ker.convolve_continuous(t, signal[:, i, j])

        self.assertTrue(np.allclose(conv_true, conv))

class TestInterpolate(unittest.TestCase):

    def test_return_coefs_for_tbins(self):

        tbins = np.arange(0, 160, 5)
        coefs = np.exp(-tbins[:-1] / 10.)
        ker = KernelRect(tbins, coefs)

        y1 = ker.interpolate(tbins, sorted_t=True)
        y2 = ker.interpolate(tbins, sorted_t=False)

        self.assertTrue(np.allclose(np.append(coefs, 0), y1))
        self.assertTrue(np.allclose(np.append(coefs, 0), y2))

    def test_interpolation_shifting(self):

        tbins = np.arange(0, 60, 5)
        coefs = np.exp(-tbins[:-1] / 10.)
        t = np.arange(-120, 120, 1)

        ker = KernelRect(tbins, coefs)
        y = ker.interpolate(t)

        tbinsl = tbins - 20
        ker = KernelRect(tbinsl, coefs)
        yl = ker.interpolate(t)

        tbinsr = tbins + 20
        ker = KernelRect(tbinsr, coefs)
        yr = ker.interpolate(t)

        self.assertTrue(np.allclose(y, shift_mask(yl, 20, fill_value=0)))
        self.assertTrue(np.allclose(y, shift_mask(yr, -20, fill_value=0)))

class TestConvolveDiscrete(unittest.TestCase):

    def test_convolve_discrete_is_convolve_continuous_with_deltas(self):
        tbins = np.arange(0, 100, 10)
        coefs = np.exp(-tbins[:-1] / 10)
        ker = KernelRect(tbins, coefs)

        dt = .5
        t = np.arange(0, 500, dt)
        argj = np.sort(np.random.choice(len(t), size=10))
        tj = t[argj]
        signal = np.zeros(len(t))
        signal[argj] = 1 / dt

        conv_cont = ker.convolve_continuous(t, signal)
        conv_disc = ker.convolve_discrete(t, tj)

        self.assertTrue(np.allclose(conv_disc, conv_cont))

# class TestConvolveDiscrete(unittest.TestCase):
#
#     def test_delta_kernel_on_exact_times(self):
#
#         kernel = KernelRect([0., .1], [1.])
#         t = np.arange(0, 200, .1)
#
#         s = np.array([0., 21.5, 30.8, 30.9, 50., 55., 60., 75.1, 88.4, 199.9])
#         # tj = (np.array([5., 15., 50., 60., 70., 80.]), np.array([0, 0, 1, 1, 2, 2]) )
#
#         conv = ker.convolve_discrete(t, tj)

if __name__ == '__main__':
    unittest.main()
