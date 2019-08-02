#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:19:26 2018

@author: diego
"""

import numpy as np
import matplotlib.pyplot as plt
#import pickle

from fun_signals import get_arg, extend_trues, shift_array, mask_args_away_from_maskinp, mask_thres_crossings

from KernelRect import KernelRect

#from fun_math import solve_Xtheta_y
#from fun_fit import fit_exp

class FitVoltageToPoissonSpikingModel:
    
    def __init__(self, neuron_model = None, ic = None, **kwargs):
        self.neuron_model = neuron_model
        self.ic = ic
        self.mask_spikes = np.zeros(self.ic.shape, dtype = bool)
        self.t0 = kwargs.get('t0', self.ic.t[0])
        self.tf = kwargs.get('tf', self.ic.t[-1] + self.ic.dt)
        self.vsubth = None
        self.vsubth_R2 = None
    
    def save(self, folder, file):
        pass
    
    def load(self, folder, file, dic_files, root_folder = '/home/diego/Dropbox/data/patch/'):
        pass
        
    ####################################################################################################################
    # SPIKES
    ####################################################################################################################
    
    def set_mask_spikes(self, dvdt_thr = 9., tl = 2., tref = 3., thr = -10., Dt = 3.):
        
        argl = get_arg(tl, self.ic.dt)
        argref = get_arg(tref, self.ic.dt)
        
        mask_spk_peaks = self.ic.mask_spikes(t0 = self.t0, tf = self.tf, thr = thr, Dt = Dt)
        mask_dvdt_thres_crossings = mask_thres_crossings(self.ic.get_dvdt(), dvdt_thr)
        
        # when they cross the threshold and come tl ms before a spike peak
        mask_spk = mask_dvdt_thres_crossings & extend_trues(mask_spk_peaks, argl, 0)
        # there can be no two spikes with separated by less than tref. I give back the last
        mask_spk = mask_spk & ~extend_trues(shift_array(mask_spk, -1, fill_value = False), argref, 0)
        
        self.mask_spikes = mask_spk
        
    def get_mask_vsubth(self, tl=0.):
        
        arg0, argf = get_arg([self.t0, self.tf], self.ic.dt)
        maskrange = np.zeros(self.ic.shape, dtype = bool)
        maskrange[arg0:argf,...] = True
        
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
        args = np.where(shift_array(self.mask_spikes, arg_ref, fill_value = False) )
        t_ref = (self.ic.t[args[0]],) + args[1:]
        return t_ref
    
    ###################################################################################################
    # SUBTHRESHOLD FITTING FUNCTIONS
    ###################################################################################################
    
    def Xmatrix(self, mask):
        pass
    
    def fit_vsubth(self, tl=0, **kwargs):
        
        R2 = kwargs.get('R2', False)
        plot_vsubth = kwargs.get('plot_vsubth', False)
        
        mask_sub = self.get_mask_vsubth(tl=tl)
        
        self._fit_vsubth(mask_sub, **kwargs)
        
        if R2:
            self.vsubth_R2 = self.get_vsubth_R2()
        
        if plot_vsubth:
            return self.plot_vsubth(tl=tl)
            
    def plot_vsubth(self, tl=0, **kwargs):
        
        axv = kwargs.get('axv', None)
        axeta = kwargs.get('axeta', None)
        
        mask_sub = self.get_mask_vsubth(tl=tl)
        
        if self.vsubth is None:
            self.simulate_vsubth()

        if (axv is None) and (axeta is None):
            fig = plt.figure(figsize = (10, 7.5))
            fig.tight_layout()
            axv = plt.subplot2grid((3, 4), (0, 0), colspan = 3);
            axresiduals = plt.subplot2grid((3, 4), (0, 3));
            axskappa = [plt.subplot2grid((3, 3), (1, col)) for col in range(3)]
            axseta = [plt.subplot2grid((3, 3), (2, col)) for col in range(3)]

        if axv is not None:
            rmser = np.round(self.get_RMSER(), 3)
            R2 = np.round(self.get_vsubth_R2(tl=tl), 3)
            #axv.plot(self.ic.t[mask_sub[:,0]], self.ic.v[mask_sub])
            #axv.plot(self.ic.t[mask_sub[:,0]], self.v_subthr[mask_sub])
            axv.plot(self.ic.v.T[mask_sub.T])
            axv.plot(self.vsubth.T[mask_sub.T], linewidth = .5)
            axv.text(.05, .8, 'R2='+str(R2), transform = axv.transAxes)
            axv.text(.05, .6, 'RMSER='+str(rmser), transform = axv.transAxes)
            #axresiduals.hist(self.vsub[mask_sub] - self.icfit.v[mask_sub], bins = 100, orientation = 'vertical')
            
        if axskappa is not None:
            t_kappa = np.arange(0, 120., .1)
            self.neuron_model.kappa.plot_lin_log(t_kappa, axs = axskappa )
        
        if (self.neuron_model.eta is not None) and axseta is not None:
            t_eta = np.arange(0., self.neuron_model.eta.tbins[-1], .1)
            self.neuron_model.eta.plot_lin_log(t_eta, axs = axseta )

        return axv, axskappa, axseta
    
    def simulate_vsubth(self):
        
        argf = get_arg(self.tf, self.ic.dt)
        
        vh, Ih = self.ic.get_vhIh()
        
        self.vsubth = np.zeros(self.ic.shape) * np.nan
        self.vsubth[:argf,...], _ = self.neuron_model.simulate_v_subthr(self.ic.t[:argf], self.ic.stim[:argf, ...],\
                                                                      self.mask_spikes[:argf,...], Ih = Ih)
    
    def get_vsubth_R2(self,tl=0):
        
        if self.vsubth is None:
            self.simulate_vsubth()
        
        mask_sub = self.get_mask_vsubth(tl=tl)
        
        R2 = np.zeros(self.ic.nsweeps) * np.nan
        
        for sw in range(self.ic.nsweeps):
            
            sum_square_error = np.sum( (self.ic.v[mask_sub[:, sw], sw] - self.vsubth[mask_sub[:, sw], sw])**2., 0)
            sum_square_mean =  np.sum( (self.ic.v[mask_sub[:, sw], sw] - np.mean(self.ic.v[mask_sub[:, sw], sw]) )**2., 0)
            R2[sw] = 1. - sum_square_error/sum_square_mean
        
        return R2
    
    def get_RMSER(self, tl=20., tr=200.):
    
        arg0, argf = get_arg([self.t0, self.tf], self.ic.dt)
        maskrange = np.zeros(self.ic.shape[0], dtype = bool)
        maskrange[arg0:argf] = True
        
        argl, argr = get_arg([tl, tr], self.ic.dt)
        
        mask_away_from_all_spikes = mask_args_away_from_maskinp(np.any(self.mask_spikes, 1), argl, argr, arg0=arg0, argf=argf) & maskrange
        
        T = np.sum(mask_away_from_all_spikes)
        
        rmse_exp = self.ic.get_RMSE(mask_away_from_all_spikes)
        
        rmse_fit = 1./(T*self.ic.nsweeps)*np.sum(((self.ic.data - self.vsubth)[mask_away_from_all_spikes, :])**2.)
        print(rmse_fit)
        
        return rmse_exp/rmse_fit
        
    
    ###################################################################################################
    # SUPTHRESHOLD FITTING FUNCTIONS
    ###################################################################################################
    
    def Ymatrix(self, mask_spk, mask_nrf):
        
        tref = self.neuron_model.tref
    
        n_gamma = self.neuron_model.gamma.nbasis
        Y_gamma = self.neuron_model.gamma.convolve_basis_discrete(self.ic.t, self.t_refractory_release() )
        
        if np.sum(Y_gamma) != 0.:
            Y = np.zeros( self.ic.v.shape + (n_gamma + 2,) )
            Y[:,:,2:] = -Y_gamma
        else:
            Y = np.zeros( self.ic.shape + (2,) )

        self.vsubth = self.vsubth.reshape(self.ic.v.shape)

        Y[...,0] = self.vsubth
        Y[...,1] = -1.
    
        return Y[mask_spk, :], Y[mask_nrf, :]
    
    def likelihood(self, theta, Yspk, Ynrf):
        
        Yspk_theta = np.dot(Yspk, theta)
        Ynrf_theta = np.dot(Ynrf, theta)
        exp_Ynrftheta = np.exp( Ynrf_theta )

        # Compute loglikelihood defined in Eq. 20 Pozzorini et al. 2015
        L = np.sum(Yspk_theta) - self.neuron_model.r0 * self.ic.dt * np.sum(exp_Ynrftheta)
                                       
        # Compute its gradient
        G = np.sum(Yspk, axis = 0) - self.neuron_model.r0 * self.ic.dt * np.dot(Ynrf.T, exp_Ynrftheta)
        
        # Compute its Hessian
        H = - self.neuron_model.r0 * self.ic.dt * np.dot(Ynrf.T * exp_Ynrftheta, Ynrf)
        
        return L, G, H
    
    def fit_supth(self, theta0 = None, tbins = None, **kwargs):
        
        maxii = kwargs.get('maxii', int(1e3))
        stop_cond = kwargs.get('stop_cond', 1e-6)
        
        mask_sub = self.get_mask_vsubth()
        
        if tbins is not None:
            self.neuron_model.gamma = KernelRect( tbins )
        
        if self.vsubth is None:
            self.simulate_vsubth()
        
        Yspk, Ynrf = self.Ymatrix(self.mask_spikes, mask_sub)
        #logL_poisson = N_spikes_tot * (np.log(N_spikes_tot/T_tot)-1)
                        
        theta = theta0
        old_L = 1

        L, G, H =0., 0., 0.
        
        for ii in range(maxii) :
            
            learning_rate = 1.0
            
            # In the first iterations using a small learning rate makes things somehow more stable
            if ii<=10 :                      
                learning_rate = 0.1
            
            L, G, H = self.likelihood(theta, Yspk, Ynrf)
            theta = theta - learning_rate * np.dot( np.linalg.inv(H), G )
            
            if (ii>0 and np.abs( (L-old_L)/old_L ) < stop_cond):
                print("\nConverged after %d iterations!\n" % (ii+1))
                break
            if np.any(np.isnan(theta)):
                print("\n There are nan parameters\n")
                break
            if ii==maxii-1:
                print('\nNot converged after %d iterations.\n' % (maxIter) )
            
            old_L = L
        
        self.neuron_model.dV = 1./theta[0]
        self.neuron_model.vt0 = theta[1]/theta[0]
        self.neuron_model.gamma.coefs = theta[2:]/theta[0]
    
            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            #L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            #reprint(L_norm)
            
         #   if math.isnan(L_norm):
         #       print "Problem during gradient ascent. Optimizatino stopped."
         #       break
    
        