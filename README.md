# iclamp-glm

## Example
### Loading and preprocessing patch clamp data
```python
from icglm.iclamp import IClamp
ic = IClamp.load_from_abf(path='./18o03049.abf') # load 9 repetitions of voltage response to noisy stimulation
ic = ic.subtract_Ih(th=1000.1).restrict(t0=1000.1, tf=11000.1) # subtract holding current and restrict data
ic.plot(sweeps=[0]) # plot first sweep
```
![](/examples/ic_plot.png)

### Simulating GLM spike train

## Documenation
* **[holi](/doc/)**
