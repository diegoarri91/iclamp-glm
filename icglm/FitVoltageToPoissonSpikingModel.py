import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

from scipy.stats import kstest

from fun_signals import get_arg, extend_trues, shift_array, mask_args_away_from_maskinp, mask_thres_crossings

from IClamp import IClamp
from KernelRect import KernelRect
from SpikeTrain import SpikeTrain


class FitVoltageToPoissonSpikingModel:

    def __init__(self, ic=None):
        self.ic = ic
        if self.ic is not None:
            self.mask_spikes = np.zeros(self.ic.shape, dtype=bool)
        else:
            self.mask_spikes = None

        self.spike_kwargs = {}  #
        self.tf = self.ic.t[-1] + self.ic.dt  # if I fit a simulation I don't set_mask_spikes so I set it like this

        # subthreshold timings for fit
        self.th = None
        self.t0_subthr = None  # inferior bound for subthreshold fitting
        self.tf_subthr = None  # superior bound for subthreshold fitting
        self.tl_subthr = None  # subthreshold values have to be this value away from spikes in ms

        self.v_subthr = None
        self.v_subthr_R2 = None  #
        self.rmser = None  #
        self.rmser_kwargs = {}  #

        self.theta0 = None
        self.newton_kwargs = None
        self.logL = None  #
        self.logL_norm = None  #

        self.ks_stats = None
        self.r = None
        self.z = None
        self.psth_kwargs = None
        self.psth_exp = None
        self.psth_model = None
        self.Md = None

        self.mask_spikes_model = None

    @property
    def neuron_model(self):
        pass

    def save(self, folder, file, neuron, filename):
        # falta subthr fits o van en modelo?
        fit_params = {'neuron': neuron, 'filename': filename, 'neuron_model': self.neuron_model, 'tf': self.tf,
                      'spike_kwargs': self.spike_kwargs, 'th': self.th, 't0_subthr': self.t0_subthr,
                      'tf_subthr': self.tf_subthr, 'tl_subthr': self.tl_subthr, 'v_subthr_R2': self.v_subthr_R2,
                      'rmser': self.rmser, 'rmser_kwargs': self.rmser_kwargs, 'theta0': self.theta0,
                      'newton_kwargs': self.newton_kwargs, 'logL': self.logL, 'logL_norm': self.logL_norm,
                      'ks_stats': self.ks_stats, 'Md': self.Md, 'psth_kwargs': self.psth_kwargs}
        pickle.dump(fit_params, open(folder + file+'.pk', "wb" ) )

    @classmethod
    def load(cls, save_folder, file, folder, filename, psth=False, time_rescale_transform=False, simulate_v_subthr=False):  #, root_folder='/home/diego/Dropbox/data/patch/'):
        fit_params = pickle.load(open(save_folder + file+'.pk', "rb" ) )

        fit_model = cls(IClamp.from_file(folder, filename), fit_params['neuron_model'] )
        #fit_model.neuron_model = fit_params['neuron_model']
        fit_model.spike_kwargs = fit_params['spike_kwargs']
        fit_model.set_mask_spikes(**fit_model.spike_kwargs)

        fit_model.tf = fit_params['tf']
        fit_model.th = fit_params['th']
        fit_model.t0_subthr = fit_params['t0_subthr']
        fit_model.tf_subthr = fit_params['tf_subthr']
        fit_model.tl_subthr = fit_params['tl_subthr']

        if simulate_v_subthr:
            fit_model.simulate_v_subthr()
        fit_model.v_subthr_R2 = fit_params['v_subthr_R2']
        #fit_model.set_RMSER(**fit_params['rmser_kwargs'])

        fit_model.theta0 = fit_params['theta0']
        fit_model.newton_kwargs = fit_params['newton_kwargs']
        fit_model.logL = fit_params['logL']
        fit_model.logL_norm = fit_params['logL_norm']
        fit_model.ks_stats = fit_params['ks_stats']

        if time_rescale_transform:
            fit_model.time_rescale_transform()

        if psth:
            if fit_params['psth_kwargs'] is not None:
                fit_model.psth_kwargs = fit_params['psth_kwargs']
            else:
                fit_model.psth_kwargs = {'trials': 10, 'delta': 4.}

            argf = get_arg(fit_model.tf, fit_model.ic.dt)
            vh, Ih = fit_model.ic.get_vhIh(th=fit_model.th)
            fit_model.mask_spikes_model = np.zeros((fit_model.ic.shape[0], fit_model.psth_kwargs['trials']), dtype=bool)
            _, _, _, _, fit_model.mask_spikes_model[:argf] = fit_model.neuron_model.simulate(fit_model.ic.t[:argf],
                                                                                             np.stack([fit_model.ic.stim[:argf,
                                                                                                 0]] * fit_model.psth_kwargs['trials'],
                                                                                                      1), Ih=np.mean(Ih))

            st_exp = SpikeTrain(fit_model.ic.t, fit_model.mask_spikes)
            st_model = SpikeTrain(fit_model.ic.t, fit_model.mask_spikes_model)
            fit_model.psth_exp = st_exp.get_PSTH(fit_model.psth_kwargs['delta'])
            fit_model.psth_model = st_model.get_PSTH(fit_model.psth_kwargs['delta'])
        if fit_model.ic.nsweeps > 1:
            fit_model.Md = st_exp.Md(st_model, fit_model.psth_kwargs['delta'])

        return fit_model


    ####################################################################################################################
    # SPIKES
    ####################################################################################################################

    def set_mask_spikes(self, dvdt_thr=5., tl=6., tref=6., thr=-10., Dt=3., tf=None):
        '''
        Sets the mask of spikes that is used for the fitting
        Besides spiking parameters it can be bounded by self.tf so no spikes beyond self.tf are used for fitting
        It is not bounded by self.t0 because I can't ignore spikes that did occur even if I don't want to fit them
        :param dvdt_thr: Spike values are going to be the last time dvdt crossed dvdt_thr before each spike peak
        :param tl:
        :param tref:
        :param thr: voltage threshold above which spikes have to be
        :param Dt:
        :return:
        '''

        self.spike_kwargs = {'dvdt_thr': dvdt_thr, 'tl': tl, 'tref': tref, 'thr': thr, 'Dt': Dt}

        self.tf = tf if not (tf is None) else self.ic.t[-1] + self.ic.dt

        argf = get_arg(self.tf, self.ic.dt)
        maskrange = np.zeros(self.ic.shape, dtype=bool)
        maskrange[:argf, ...] = True

        argl = get_arg(tl, self.ic.dt)
        argref = get_arg(tref, self.ic.dt)

        mask_spk_peaks = self.ic.mask_spikes(thr=thr, Dt=Dt)
        mask_dvdt_thres_crossings = mask_thres_crossings(self.ic.get_dvdt(), dvdt_thr)

        # when they cross the threshold and come tl ms before a spike peak
        mask_spk = mask_dvdt_thres_crossings & extend_trues(mask_spk_peaks, argl, 0)
        # there can be no two spikes with separated by less than tref. I give back the last
        mask_spk = mask_spk & ~extend_trues(shift_array(mask_spk, -1, fill_value=False), argref, 0)

        self.mask_spikes = mask_spk & maskrange

    def get_mask_v_subthr(self, t0, tf, tl):
        '''

        used in fit_v_subthr, plot_v_subthr and set_v_subthr_R2 with tl=self.tl_subthr
        used in fit_supthr with potentially a different value than self.tl_subthr
        :param tl: time before spike to be considered subthreshold
        :return:
        '''
        arg0, argf = get_arg([t0, tf], self.ic.dt)
        maskrange = np.zeros(self.ic.shape, dtype=bool)
        maskrange[arg0:argf, ...] = True

        argl = get_arg(tl, self.ic.dt)
        arg_ref = get_arg(self.neuron_model.tref, self.ic.dt)
        mask_sub = ~extend_trues(self.mask_spikes, argl, arg_ref - 1) & maskrange

        return mask_sub

    def t_spikes(self):
        args = np.where(self.mask_spikes)
        t_spk = (self.ic.t[args[0]],) + args[1:]
        return t_spk

    def t_refractory_release(self):
        arg_ref = get_arg(self.neuron_model.tref, self.ic.dt)
        args = np.where(shift_array(self.mask_spikes, arg_ref, fill_value=False))
        t_ref = (self.ic.t[args[0]],) + args[1:]
        return t_ref

    ###################################################################################################
    # SUBTHRESHOLD FITTING FUNCTIONS
    ###################################################################################################

    def Xmatrix(self, mask):
        pass

    def _fit_vsubth(self, mask_sub, **fit_kwargs):
        pass

    def plot_v_subthr(self, axv=None, axs_kappa=None, axs_eta=None):

        mask_sub = self.get_mask_v_subthr(t0=self.t0_subthr, tf=self.tf_subthr, tl=self.tl_subthr)

        if axv is None and axs_kappa is None and axs_eta is None:
            fig = plt.figure(figsize=(10, 7.5))
            fig.tight_layout()
            axv = plt.subplot2grid((3, 4), (0, 0), colspan=3)
            # axresiduals = plt.subplot2grid((3, 4), (0, 3))
            axs_kappa = [plt.subplot2grid((3, 3), (1, col)) for col in range(3)]
            axs_eta = [plt.subplot2grid((3, 3), (2, col)) for col in range(3)]

        if axv is not None:
            R2 = np.round(self.v_subthr_R2, 3)
            axv.plot(self.ic.v.T[mask_sub.T])
            axv.plot(self.v_subthr.T[mask_sub.T], linewidth=.5)
            axv.text(.05, .8, 'R2=' + str(R2), transform=axv.transAxes)
            if self.rmser is not None:
                rmser = np.round(self.rmser, 3)
                axv.text(.05, .6, 'RMSER=' + str(rmser), transform=axv.transAxes)
            # axresiduals.hist(self.vsub[mask_sub] - self.icfit.v[mask_sub], bins = 100, orientation = 'vertical')

        if axs_kappa is not None:
            t_kappa = np.arange(0, 120., .1)
            self.neuron_model.kappa.plot_lin_log(t_kappa, axs=axs_kappa)

        if (self.neuron_model.eta is not None) and axs_eta is not None:
            t_eta = np.arange(0., self.neuron_model.eta.tbins[-1], .1)
            self.neuron_model.eta.plot_lin_log(t_eta, axs=axs_eta)

        return axv, axs_kappa, axs_eta

    def simulate_v_subthr(self):

        argf = get_arg(self.tf, self.ic.dt)  # I want it to go to self.tf because I'll need it for fitting supthr

        vh, Ih = self.ic.get_vhIh(th=self.th)

        self.v_subthr = np.zeros(self.ic.shape) * np.nan
        self.v_subthr[:argf, ...], _, _, _ = self.neuron_model.simulate_v_subthr(self.ic.t[:argf],
                                                                                 self.ic.stim[:argf, ...],
                                                                                 self.mask_spikes[:argf, ...], Ih=Ih)

    def set_v_subthr_R2(self):

        mask_sub = self.get_mask_v_subthr(t0=self.t0_subthr, tf=self.tf_subthr, tl=self.tl_subthr)

        R2 = np.zeros(self.ic.nsweeps) * np.nan

        for sw in range(self.ic.nsweeps):
            sum_square_error = np.sum((self.ic.v[mask_sub[:, sw], sw] - self.v_subthr[mask_sub[:, sw], sw]) ** 2., 0)
            sum_square_mean = np.sum((self.ic.v[mask_sub[:, sw], sw] - np.mean(self.ic.v[mask_sub[:, sw], sw])) ** 2.,
                                     0)
            R2[sw] = 1. - sum_square_error / sum_square_mean

        self.v_subthr_R2 = R2

    def set_RMSER(self, tl=20., tr=200.):

        self.rmser_kwargs = {'tl': tl, 'tr': tr}

        # I want maskrange to be the same maskrange that I use to fit subthreshold
        arg0, argf = get_arg([self.t0_subthr, self.tf_subthr], self.ic.dt)
        maskrange = np.zeros(self.ic.shape[0], dtype=bool)
        maskrange[arg0:argf] = True

        argl, argr = get_arg([tl, tr], self.ic.dt)

        mask_away_from_all_spikes = ~extend_trues(np.any(self.mask_spikes, 1), argl, argr) & maskrange

        T = np.sum(mask_away_from_all_spikes)

        rmse_exp = self.ic.get_RMSE(mask_away_from_all_spikes)

        rmse_fit = np.sqrt(1. / (T * self.ic.nsweeps) * np.sum(
            ((self.ic.data - self.v_subthr)[mask_away_from_all_spikes, :]) ** 2.))

        self.rmser = rmse_exp / rmse_fit

    ###################################################################################################
    # SUPTHRESHOLD FITTING FUNCTIONS
    ###################################################################################################

    def Ymatrix(self, mask_spk, mask_nrf):

        if not (self.neuron_model.gamma is None):
            n_gamma = self.neuron_model.gamma.nbasis
            Y_gamma = self.neuron_model.gamma.convolve_basis_discrete(self.ic.t, self.t_refractory_release())
            Y = np.zeros(self.ic.v.shape + (n_gamma + 2,))
            Y[:, :, 2:] = -Y_gamma
        else:
            Y = np.zeros(self.ic.shape + (2,))

        Y[:, :, 0] = self.v_subthr
        Y[:, :, 1] = -1.

        return Y[mask_spk, :], Y[mask_nrf, :]

    def likelihood(self, theta, Yspk, Ynrf):

        Yspk_theta = np.dot(Yspk, theta)
        Ynrf_theta = np.dot(Ynrf, theta)
        exp_Ynrftheta = np.exp(Ynrf_theta)

        # Compute loglikelihood defined in Eq. 20 Pozzorini et al. 2015
        logL = np.sum(Yspk_theta) - self.neuron_model.r0 * self.ic.dt * np.sum(exp_Ynrftheta)

        # Compute its gradient
        G = np.sum(Yspk, axis=0) - self.neuron_model.r0 * self.ic.dt * np.dot(Ynrf.T, exp_Ynrftheta)

        # Compute its Hessian
        H = - self.neuron_model.r0 * self.ic.dt * np.dot(Ynrf.T * exp_Ynrftheta, Ynrf)

        return logL, G, H

    def fit_supthr(self, theta0=None, tbins=None, newton_kwargs={}, prints=False, print_kwargs={}, time_rescale_transform=False, psth=True,
                   psth_kwargs={}, plot=False):

        self.theta0 = theta0
        self.newton_kwargs = newton_kwargs

        maxii = self.newton_kwargs.get('maxii', int(1e3))
        stop_cond = self.newton_kwargs.get('stop_cond', 1e-6)

        mask_nrf = self.get_mask_v_subthr(t0=0., tf=self.tf, tl=0.)

        if tbins is not None:  # if it is None I want it to fit without gamma
            self.neuron_model.gamma = KernelRect(tbins)

        Yspk, Ynrf = self.Ymatrix(self.mask_spikes, mask_nrf)

        n_spikes = np.sum(self.mask_spikes)
        T = np.sum(mask_nrf) * self.ic.dt
        logL_poisson = n_spikes * (np.log(n_spikes / T) - 1)

        theta = theta0
        old_logL = 1

        logL, G, H = 0., 0., 0.

        logLv = []
        nan_parameters = False
        t0 = time.time()
        for ii in range(maxii):

            learning_rate = newton_kwargs.get('learning_rate', 1e0)

            # In the first iterations using a small learning rate makes things somehow more stable
            if ii <= 10:
                learning_rate = 0.1

            if prints and ii%50:
                print(ii)

            logL, G, H = self.likelihood(theta, Yspk, Ynrf)
            logLv += [logL]
            try:
                theta = theta - learning_rate * np.dot(np.linalg.inv(H), G)
            except(np.linalg.LinAlgError):
                theta = theta*np.nan

            if ii > 0 and np.abs((logL - old_logL) / old_logL) < stop_cond:
                status = "\nConverged after %d iterations!\n" % (ii + 1)
                break
            if np.any(np.isnan(theta)):
                status = "\n There are nan parameters\n"
                nan_parameters = True
                break
            if ii == maxii - 1:
                status = '\nNot converged after %d iterations.\n' % (maxii)

            old_logL = logL

        fitting_time = (time.time() - t0)/60.

        if prints:
            print(status)
            print('\n Elapsed time: {} minutes \n'.format(fitting_time))

        self.neuron_model.dV = 1. / theta[0]
        self.neuron_model.vt0 = theta[1] / theta[0]
        if self.neuron_model.gamma is not None:
            self.neuron_model.gamma.coefs = theta[2:] / theta[0]
        self.logL = logLv
        self.logL_norm = (logL - logL_poisson) / np.log(2) / n_spikes

        if time_rescale_transform and not(nan_parameters):
            if prints:
                print('\n Doing time rescaling transform \n')
            self.time_rescale_transform()
            if prints():
                print('\n Time rescaling transform is done \n')

        if psth and not(nan_parameters):
            psth_trials = psth_kwargs.get('trials', 10)
            psth_delta = psth_kwargs.get('delta', 4.)
            self.psth_kwargs = {'trials': psth_trials, 'delta': psth_delta}

            argf = get_arg(self.tf, self.ic.dt)
            vh, Ih = self.ic.get_vhIh(th=self.th)
            self.mask_spikes_model = np.zeros((self.ic.shape[0], psth_trials), dtype=bool)
            _, _, _, _, self.mask_spikes_model[:argf] = self.neuron_model.simulate(self.ic.t[:argf],
                                                                                   np.stack([self.ic.stim[:argf,
                                                                                             0]] * psth_trials,
                                                                                            1), Ih=np.mean(Ih) )

            st_exp = SpikeTrain(self.ic.t, self.mask_spikes)
            st_model = SpikeTrain(self.ic.t, self.mask_spikes_model)
            self.psth_exp = st_exp.get_PSTH(psth_delta)
            self.psth_model = st_model.get_PSTH(psth_delta)
            if self.ic.nsweeps > 1:
                self.Md = st_exp.Md(st_model, psth_delta)

        if plot:
            return self.plot_supthr()

    def plot_supthr(self):

        fig = plt.figure(figsize=(10, 7.5))
        fig.tight_layout()
        axr = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        axlogL = plt.subplot2grid((3, 3), (0, 2))
        axs_time_rescale_transform = [plt.subplot2grid((3, 2), (1, col)) for col in range(2)]
        axsgamma = [plt.subplot2grid((3, 3), (2, col)) for col in range(3)]

        axr.plot(self.psth_exp)
        axr.plot(self.psth_model)

        axlogL.plot(range(1, len(self.logL) + 1), self.logL, 'C0-o')

        logL_norm = np.round(self.logL_norm, 0)
        axs_time_rescale_transform[0].text(.05, .9, 'logL_norm=' + str(logL_norm),
                                           transform=axs_time_rescale_transform[0].transAxes)

        bins = np.arange(0, 1.01, .01)

        # for sw in range(self.ic.nsweeps):
        #     values, bins = np.histogram(self.z[sw], bins=bins)
        #     cum = np.append(0., np.cumsum(values) / np.sum(values))
        #     axs_time_rescale_transform[0].plot(bins, cum, '-')

        # TODO poner en otro lado
        z = np.concatenate((self.z))
        stat, p = np.round(self.ks_stats[0], 4), np.round(self.ks_stats[1], 3)
        axs_time_rescale_transform[0].text(.05, .8, 'stat=' + str(stat) + '  p=' + str(p),
                                           transform=axs_time_rescale_transform[0].transAxes)
        values, bins = np.histogram(z, bins=bins)
        cum = np.append(0., np.cumsum(values) / np.sum(values))
        axs_time_rescale_transform[0].plot(bins, cum, '-')
        axs_time_rescale_transform[0].plot([0, 1], [0, 1], 'k--')

        for sw in range(self.ic.nsweeps):
            axs_time_rescale_transform[1].plot(self.z[sw][:-1], self.z[sw][1:], 'C0o')

        vt0 = np.round(self.neuron_model.vt0, 1)
        dV = np.round(self.neuron_model.dV, 3)
        axsgamma[0].text(.6, .9, 'vt0=' + str(vt0), transform=axsgamma[0].transAxes)
        axsgamma[0].text(.6, .8, 'dV=' + str(dV), transform=axsgamma[0].transAxes)
        if self.neuron_model.gamma is not None:
            t_gamma = np.arange(0., self.neuron_model.gamma.tbins[-1], .1)
            self.neuron_model.gamma.plot_lin_log(t_gamma, axs=axsgamma)

        return axr, axlogL, axs_time_rescale_transform, axsgamma

    def time_rescale_transform(self):
        vh, Ih = self.ic.get_vhIh(th=self.th)
        _, self.r, _, _ = self.neuron_model.simulate_v_subthr(self.ic.t, self.ic.stim, self.mask_spikes, Ih=Ih)
        integral_r = np.cumsum(self.r * self.ic.dt, axis=0)

        z = []

        for sw in range(self.ic.nsweeps):
            # TODO understand if I have to shift mask_spikes or not
            # Lambda = integral_r[self.mask_spikes[:, sw], sw]
            Lambda = integral_r[shift_array(self.mask_spikes, 1, fill_value=False)[:, sw], sw]
            z += [1. - np.exp(-(Lambda[1:] - Lambda[:-1]))]

        self.z = z
        z = np.concatenate((self.z))
        self.ks_stats = kstest(z, 'uniform', args=(0, 1))

    def fit(self, tref=8., tf=None, spikes_kwargs={}, th=500., t0_subthr=0., tf_subthr=None, tl_subthr=0., subthr_kwargs={}, rmser=True, rmser_kwargs={},
            plot_subthr=True, theta0=None, tbins_gamma=None, newton_kwargs={}, time_rescale_transform=True, psth=True, psth_kwargs={}, plot_supthr=True):

        self.neuron_model.tref = tref

#        if tf is not None:
#            self.tf = tf

        if spikes_kwargs is not None:
            self.set_mask_spikes(**spikes_kwargs, tf=tf)

        self.fit_v_subthr(th=th, t0=t0_subthr, tf=tf_subthr, tl=tl_subthr, fit_kwargs=subthr_kwargs, rmser=rmser, rmser_kwargs=rmser_kwargs, plot=plot_subthr)

        self.fit_supthr(theta0=theta0, tbins=tbins_gamma, newton_kwargs=newton_kwargs, time_rescale_transform=time_rescale_transform, psth=psth,
                   psth_kwargs=psth_kwargs, plot=plot_supthr)
