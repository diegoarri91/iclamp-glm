# iclamp-glm
iclamp-glm is a Python module for the analysis of electrophysiological patch clamp data by using encoding and decoding neural models. It is built on top of SciPy.

## Example
### Loading and preprocessing patch clamp data
```python
import numpy as np
from icglm.iclamp import IClamp
ic = IClamp.load_from_abf(path='./18o03049.abf') # load 9 repetitions of voltage response to noisy stimulation
ic = ic.subtract_Ih(th=1000.1).restrict(t0=1000.1, tf=11000.1) # subtract holding current and restrict data
ic.plot(sweeps=[0]) # plot first sweep
```
<p align="center">
  <img src=/examples/ic_plot.png>
</p>

### Fitting encoding GLM to data
```python
from icglm.glm_fitting import GLMFitter
glm_fit = GLMFitter(ic).set_mask_spikes(thr=-13).subsample(10) # find spikes and subsample signal
tbins_kappa = np.arange(0, 220, 5) # define time bins for stimulus filter
tbins_eta = np.arange(0, 480, 10) # define time bins for post-spike filter
glm_fit.fit(tbins_kappa=tbins_kappa, tbins_eta=tbins_eta)
glm_fit.plot_filters()
```

<p align="center">
  <img src=examples/filters.png>
</p>

```python
from icglm.kernels import KernelFun
glm_fit.set_mask_spikes_model(trials=9) # simulating 9 trials of fit GLM
glm_fit.psth(psth_kernel=KernelFun.gaussian_delta(delta=40)) # computing psth with gaussian kernel
glm_fit.plot_raster()
```

<p align="center">
  <img src=examples/raster_plot.png>
</p>

### Decoding stimulus from spike data and GLM
```python
from icglm.glm_decoding import GLMDecoder
Imu, Isd = np.mean(glm_fit.stim), np.mean(np.std(glm_fit.stim, 0))
I0 = np.zeros(len(glm_fit.t))
decoder = GLMDecoder(glms=[glm_fit.glm], t=glm_fit.t, mask_spk=[glm_fit.mask_spikes], 
                     tau=3, Imu=[Imu], Isd=[Isd], I_true=(glm_fit.stim - Imu) / Isd)
decoder.estimate_MAP_I(I0, prior='OU')
decoder.plot_decoded_stimulus(t0=4000, tf=6500)
```

<p align="center">
  <img src=examples/decoding_plot.png>
</p>
