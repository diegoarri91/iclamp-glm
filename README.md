# iclamp-glm
iclamp-glm is a Python module for the analysis of electrophysiological patch clamp data by using encoding and decoding neural models. It is built on top of SciPy.

## Example
### Loading and preprocessing patch clamp data
```python
import matplotlib.pyplot as plt
import numpy as np
from icglm.iclamp import IClamp

ic = IClamp.load_from_abf(path='./18o03049.abf') # load 9 repetitions of voltage response to noisy stimulation
ic = ic.subtract_Ih(th=1000.1).restrict(t0=1000.1, tf=11000.1) # subtract holding current and restrict data
ic.set_mask_spikes(thr=0, t_before_spike_peak=0.8) # set spikes
ic = ic.subsample(1) # subsample data

fig, axs = ic.plot(sweeps=[0], spikes=True)
```
<p align="center">
  <img src=/examples/ic_plot.png>
</p>

### Fitting encoding GLM to data
```python
from icglm.models.glm import GLM
from icglm.kernels import KernelRect
from icglm.plot.fit import GLMPlotter

# define time bins and initial parameters for GLM fit
tbins_kappa = np.arange(0, 220, 5)
coefs_kappa0 = 1e-2 * np.exp(-tbins_kappa[:-1] / 50)
tbins_eta = np.arange(0, 480, 10)
coefs_eta0 = 6 * np.exp(-tbins_eta[:-1] / 50)
glm = GLM(u0=15, kappa=KernelRect(tbins_kappa, coefs_kappa0), eta=KernelRect(tbins_eta, coefs_eta0))

newton_kwargs = dict(max_iterations=30, stop_cond=5e-4, learning_rate=5e-1, initial_learning_rate=5e-2, warm_up_iterations=3)
optimizer = glm.fit(ic.t, ic.stim, ic.mask_spikes, newton_kwargs=newton_kwargs)

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(15, 5), ncols=3)
glm_plotter = GLMPlotter(ic=ic, model=glm, optimizer=optimizer) # plotting class
glm_plotter.plot_filters(axs=(ax1, ax2))
glm_plotter.plot_posterior_iterations(ax=ax3)
```

<p align="center">
  <img src=examples/filters.png>
</p>

```python
from icglm.kernels import KernelFun
from icglm.spiketrain import SpikeTrain

psth_kernel = KernelFun.gaussian_delta(delta=40)
v, r, mask_spikes = glm.sample(ic.t, ic.stim) # simulating 9 repetitions of fit GLM
glm_plotter.mask_spikes_model = mask_spikes

# compute psths
glm_plotter.psth_model = SpikeTrain(ic.t, mask_spikes).get_psth(psth_kernel) 
glm_plotter.psth_exp = SpikeTrain(ic.t, ic.mask_spikes).get_psth(psth_kernel)

fig, axs = glm_plotter.plot_raster()
```

<p align="center">
  <img src=examples/raster_plot.png>
</p>

### Decoding stimulus from spike data and GLM
```python
from icglm.processes import OUProcess
from icglm.plot.decoding import DecodingPlotter

mu, sd = np.mean(ic.stim), np.mean(np.std(ic.stim, 0))
ou = OUProcess(mu=mu, sd=sd, tau=3) # define prior for decoding
stim_true = ic.stim[:, 0]
stim0 = np.zeros(len(ic.t)) + mu # define initial stimulus

newton_kwargs = dict(learning_rate=1e-1, stop_cond=2e-6, max_iterations=20)
stim_dec, optimizer = glm.decode(ic.t, ic.mask_spikes, stim0=stim0, prior=ou, newton_kwargs=newton_kwargs)

dec_plotter = DecodingPlotter(t=ic.t, model=[glm], optimizer=optimizer, mask_spikes=[ic.mask_spikes], stim_dec=stim_dec, stim_true=stim_true) # plotting class
fig, axs = dec_plotter.plot_decoded_stimulus(t0=4000, tf=6500)
```

<p align="center">
  <img src=examples/decoding_plot.png>
</p>
