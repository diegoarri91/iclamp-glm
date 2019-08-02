#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  10 10:28:06 2018

@author: diego
"""

import numpy as np

from fun_signals import get_dt, get_arg
from fun_misc import are_equal


class SRM:

    def __init__(self, kappa=None, vr=None, eta=None, vt0=None, dV=None, gamma=None, r0=10. / 1000., tref=0.,
                 vpeak=None):
        self.kappa = kappa
        self.eta = eta
        self.gamma = gamma

        self.vr = vr
        self.tref = tref
        self.vt0, self.dV = vt0, dV
        self.r0 = r0
        self.vpeak = vpeak

    @property
    def R(self):
        return self.kappa.area()

    def simulate(self, t, I, Ih=0.):

        # Ignore overflow warning when calculating r[j+1] which can be very big if dV small
        np.seterr(over='ignore')

        vr = self.vr
        tref = self.tref
        vt0, dV = self.vt0, self.dV
        vpeak = self.vpeak
        r0 = self.r0

        dt = get_dt(t)

        tref = tref if tref >= dt else dt
        argref = get_arg(tref, dt)

        if I.ndim == 1:
            I = I.reshape(len(t), 1)
            shape = (len(t), 1)
        else:
            shape = I.shape

        v = vr + Ih * self.kappa.area() + self.kappa.convolve_continuous(t, I - Ih)
        r = np.zeros(shape) * np.nan
        r[:2, ...] = 0.
        eta = np.zeros(shape)
        gamma = np.zeros(shape)

        mask_spk = np.zeros(shape, dtype=bool)
        mask_ref = np.zeros(shape[1:], dtype=bool)
        t_refs = 2. * dt * np.ones(shape[1:])

        j = 0
        while j < len(t) - 1:

            r[j + 1, ...] = r0 * np.exp((v[j + 1, ...] - (vt0 + gamma[j + 1, ...])) / dV)
            r[j + 1, mask_ref] = 0.
            r[j + 1, r[j + 1, ...] > 1./dt] = 1./dt
            p_spk = 1. - np.exp(-r[j + 1, ...] * dt)
            aux = np.random.rand(*shape[1:])

            mask_spk[j + 1, ...] = p_spk > aux

            t_refs[mask_ref] += dt

            if vpeak is not None:
                v[j + 1, mask_spk[j + 1, ...]] = vpeak
                v[j + 1, mask_ref] = vr

            if np.any(mask_spk[j + 1, ...]) and j + 1 + argref < len(t):
                if self.eta is not None:
                    eta_ = self.eta.interpolate(t[j + 1 + argref:] - t[j + 1 + argref])[:, None]
                    eta[j + 1 + argref:, mask_spk[j + 1, ...]] += eta_
                    v[j + 1 + argref:, mask_spk[j + 1, ...]] -= eta_
                if self.gamma is not None:
                    gamma[j + 1 + argref:, mask_spk[j + 1, ...]] += self.gamma.interpolate(
                        t[j + 1 + argref:] - t[j + 1 + argref])[:, None]

            mask_ref[mask_spk[j + 1, ...]] = True
            mask_release_ref = (t_refs > tref)
            mask_ref[mask_release_ref] = False
            t_refs[mask_release_ref] = 2. * dt

            j += 1

        return v, r, eta, gamma, mask_spk

    def simulate_v_subthr(self, t, I, mask_spk, Ih=0.):

        # AT SPIKING VALUE (mask_spk) v has to be the result of normal integration (no nan no vpeak!!!) because it is used for fitting

        vr = self.vr
        vt0, dV = self.vt0, self.dV

        tref = self.tref
        r0 = self.r0

        dt = get_dt(t)

        tref = tref if tref >= dt else dt
        argref = get_arg(tref, dt)

        if I.ndim == 1:
            I = I.reshape(len(t), 1)
            shape = (len(t), 1)
        else:
            shape = I.shape

        v = vr + Ih * self.kappa.area() + self.kappa.convolve_continuous(t, I - Ih)
        r = np.zeros(shape) * np.nan
        r[0, :] = 0.
        eta = np.zeros(shape)
        gamma = np.zeros(shape)

        mask_ref = np.zeros(shape[1:], dtype=bool)
        t_refs = 2. * dt * np.ones(shape[1:])

        j = 0

        while j < len(t) - 1:

            if vt0 is not None and dV is not None:
                r[j + 1, ...] = r0 * np.exp((v[j + 1, ...] - (vt0 + gamma[j + 1, ...])) / dV)
                r[j + 1, mask_ref] = 0.
                r[j + 1, r[j + 1, ...] > 1./dt] = 1./dt

            t_refs[mask_ref] += dt

            v[j + 1, mask_ref] = vr

            if np.any(mask_spk[j + 1, ...]) and j + 1 + argref < len(t):
                if self.eta is not None:
                    eta_ = self.eta.interpolate(t[j + 1 + argref:] - t[j + 1 + argref])[:, None]
                    eta[j + 1 + argref:, mask_spk[j + 1, ...]] += eta_
                    v[j + 1 + argref:, mask_spk[j + 1, ...]] -= eta_
                if self.gamma is not None:
                    gamma[j + 1 + argref:, mask_spk[j + 1, ...]] += self.gamma.interpolate(
                                                                       t[j + 1 + argref:] - t[j + 1 + argref] )[:, None]

            mask_ref[mask_spk[j + 1, ...]] = True
            mask_release_ref = (t_refs > tref)
            mask_ref[mask_release_ref] = False
            t_refs[mask_release_ref] = 2. * dt

            j += 1

        return v, r, eta, gamma
