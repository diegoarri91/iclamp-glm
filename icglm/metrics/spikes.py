import numpy as np
from scipy.stats import kstest


def log_likelihood_normed(dt, mask_spikes, u, r):
    lent = mask_spikes.shape[0]
    n_spikes = np.sum(mask_spikes, 0)
    n_spikes = n_spikes[n_spikes > 0]
    log_likelihood = np.sum(u[mask_spikes]) - dt * np.sum(r) + np.sum(n_spikes) * np.log(dt)
    log_likelihood_poisson = np.sum(n_spikes * (np.log(n_spikes / lent) - 1))
    log_like_normed = (log_likelihood - log_likelihood_poisson) / np.log(2) / np.sum(n_spikes)
    return log_like_normed


def time_rescale_transform(dt, mask_spikes, r):
    integral_r = np.cumsum(r * dt, axis=0)

    z = []
    for sw in range(mask_spikes.shape[1]):
        integral_r_spikes = integral_r[mask_spikes[:, sw], sw]  # I think there is no need for shifting the mask
        z += [1. - np.exp(-(integral_r_spikes[1:] - integral_r_spikes[:-1]))]

    ks_stats = kstest(np.concatenate(z), 'uniform', args=(0, 1))

    return z, ks_stats
