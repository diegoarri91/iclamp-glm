import sys
sys.path.append("/home/diego/Dropbox/hold_noise/iclamp-glm/")

import matplotlib.pyplot as plt
import numpy as np

from icglm.glm import GLM
from icglm.kernels import KernelRect
from icglm.processes import WienerPhaseProcess

t = np.arange(0, 1000, 1)
mu, sd = 60, 0.1
wpp = WienerPhaseProcess(w=10e-2, sd=2e-2)
stim = wpp.sample(t, seed=0) * sd + mu
kappa = KernelRect.exponential(tf=240, dt=8, tau=40, A=1e-2)
eta = KernelRect.exponential(tf=450, dt=25, tau=75, A=7)
u0 = 16
glm_true = GLM(u0=u0, kappa=kappa, eta=eta)
np.random.seed(0)
r, kappa_conv, eta_conv, mask_spk_true = glm_true.simulate(t, stim)

r, kappa_conv, eta_conv, mask_spk_true = glm_true.simulate(t, stim)
fig, (ax1, ax2) = plt.subplots(figsize=(16, 5), nrows=2)
ax1.plot(t, mask_spk_true)
ax2.plot(t, stim)

newton_kwargs = dict(max_iterations=30, stop_cond=5e-5, learning_rate=5e-1, initial_learning_rate=5e-10, warm_up_iterations=5)

stim_true = stim.copy()
# stim0 = np.zeros(len(t))# + mu
stim0 = 1e-1 * t + np.pi
ax2.plot(t, stim0 * sd + mu)
stim_dec, optimizer = glm_true.decode(t, mask_spk_true, stim0=stim0, mu_I=mu, sd_I=sd, stim_h=mu, prior=wpp, newton_kwargs=newton_kwargs, verbose=True)
# fig, ax = plt.subplots(figsize=(16, 5))
ax2.plot(t, sd * stim_dec + mu)

fig, (ax1, ax2) = plt.subplots(figsize=(10, 5), ncols=2)
ax1.plot(optimizer.log_prior_iterations)
ax2.plot(optimizer.log_posterior_iterations, '-o')