# iclamp-glm
iclamp-glm is a Python module for the analysis of electrophysiological patch clamp data by using encoding and decoding neural models. It is built on top of SciPy.

## Example
### Loading and preprocessing patch clamp data
```python
from icglm.iclamp import IClamp
ic = IClamp.load_from_abf(path='./18o03049.abf') # load 9 repetitions of voltage response to noisy stimulation
ic = ic.subtract_Ih(th=1000.1).restrict(t0=1000.1, tf=11000.1) # subtract holding current and restrict data
ic.plot(sweeps=[0]) # plot first sweep
```
<p align="center">
  <img src=/examples/ic_plot.png>
</p>

### Simulating GLM spike train
```python
from icglm.glm_fitting import GLMFitter
glm_fit = GLMFitter(ic).set_mask_spikes(thr=-13).subsample(10) # find spikes and subsample signal
tbins_kappa = np.arange(0., 220, 5.) # define time bins for stimulus filter
tbins_eta = np.arange(0., 475., 10.) # define time bins for post-spike filter
glm_fit.fit(tbins_kappa=tbins_kappa, tbins_eta=tbins_eta)
glm_fit.plot_filters()
```

<p align="center">
  <img src=examples/filters.png>
</p>

```python
glm_fit.set_mask_spikes_model(trials=9)
glm_fit.psth(psth_kernel=KernelFun.gaussian_delta(delta=40))
glm_fit.plot_raster()
```
<p align="center">
  <img src=examples/raster_plot.png>
</p>
