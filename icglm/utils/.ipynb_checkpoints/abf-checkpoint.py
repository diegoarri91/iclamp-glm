import pyabf
import numpy as np


def load_data(path, sweeps=None, channels=None):
    """
    Given a current_folder and a filename it returns an array t, a float dt and 
    an array data=[datapoint, sweep, channel]
    """

    abf = pyabf.ABF(path)

    if channels is None:
        channels = abf.channelList

    if sweeps is None:
        sweeps = abf.sweepList

    n_channels = len(channels)
    n_sweeps = len(sweeps)
    n_points = len(abf.sweepX)

    t = abf.sweepX
    t = (t - t[0]) * 1000
    dt = 1 / abf.dataRate * 1000

    data = np.zeros((n_points, n_sweeps, n_channels)) * np.nan

    for c, ch in enumerate(channels):

        for s, sw in enumerate(sweeps):
            abf.setSweep(sweepNumber=sw, channel=ch)

            data[:, s, c] = abf.sweepY

    return t, dt, data


def load_protocol(path, sweeps=None):
    abf = pyabf.ABF(path)

    if sweeps is None:
        sweeps = abf.sweepList

    n_sweeps = len(sweeps)
    n_points = len(abf.sweepX)

    stim = np.zeros((n_points, n_sweeps)) * np.nan

    for s, sw in enumerate(sweeps):
        abf.setSweep(sweepNumber=sw)

        stim[:, s] = abf.sweepC

    return stim
