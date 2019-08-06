import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt

from .signals import get_arg, get_dt


class SpikeTrain:

    def __init__(self, t, mask):
        self.t = t
        self.mask = mask
        self.dt = get_dt(t)
        self.nsweeps = mask.shape[1]

    def save(self, path):
        dic = {'arg0': get_arg(self.t[0], self.dt), 'dt': self.dt, 'arg_spikes': np.where(self.mask),
               'shape': self.mask.shape}

        with open(path, 'wb') as pk_f:
            pickle.dump(dic, pk_f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as pk_f:
            dic = pickle.load(pk_f)

        arg0 = dic.get('arg0', 0)
        t = np.arange(arg0, arg0 + dic['shape'][0], 1) * dic['dt']
        mask = np.zeros(dic['shape'], dtype=bool)
        mask[dic['arg_spikes']] = True
        return cls(t, mask)

    def t_spikes(self):
        args = np.where(self.mask_)
        t_spk = (self.t[args[0]],) + args[1:]
        return t_spk

    def plot(self, ax=None, **kwargs):

        sweeps = kwargs.get('sweeps', range(self.nsweeps))
        color = kwargs.get('color', ['C' + str(ii % 10) for ii in range(len(sweeps))])
        if not (type(color) is list):
            color = [color] * len(sweeps)
        t0 = kwargs.get('t0', 0.)
        tf = kwargs.get('tf', self.t[-1])
        linewidth = kwargs.get('linewidth', None) if not (kwargs.get('linewidth', None) is None) else kwargs.get('lw',
                                                                                                                 1.)
        delta = kwargs.get('delta', .8)

        if ax is None:
            figsize = kwargs.get('figsize', (8, 5))
            fig, ax = plt.subplots(figsize=figsize)

        for ii, sw in enumerate(sweeps):
            t_spikes_ = self.t[self.mask[:, sw]]
            t_spikes_ = t_spikes_[(t_spikes_ >= t0) & (t_spikes_ < tf)]
            t_spikes_ = np.array([t_spikes_] * 2)

            bars = np.zeros((2, t_spikes_.shape[1])) * np.nan
            bars[0, :] = -ii + delta / 2.
            bars[1, :] = -ii - delta / 2.

            ax.plot(t_spikes_, bars, '-', linewidth=linewidth, color=color[ii])

        extra_range = (self.t[-1] - self.t[0]) * .005
        ax.set_xlim(self.t[0] - extra_range, self.t[-1] + extra_range)
        ax.set_ylim(-ii - (delta / 2. + (1 - delta)), delta / 2. + (1 - delta))

        return ax

    def dot(self, st, kernel1, kernel2):
        '''
        :param st: spike train
        :param kernel:
        :return: dot product between self and st. If there is more than one spike train dot product is taken between
        corresponding pairs
        '''

        argf = min(len(self.t), len(st.t))

        self_conv = kernel1.convolve_continuous(self.t[:argf], self.mask[:argf] / self.dt)
        st_conv = kernel2.convolve_continuous(st.t[:argf], st.mask[:argf] / self.dt)

        return np.sum(self_conv * st_conv, 0) * self.dt

    def norm(self, kernel1, kernel2):
        return np.sqrt(self.dot(self, kernel1, kernel2))

    def norm_squared(self, kernel1, kernel2):
        return self.dot(self, kernel1, kernel2)

    def L(self, kernel1, kernel2):
        return np.mean(self.norm_squared(kernel1, kernel2))

    def var(self, kernel1, kernel2):
        return self.nsweeps / (self.nsweeps - 1) * (
                    self.L(kernel1, kernel2) - self.population_norm(kernel1, kernel2, biased=True) ** 2)

    def average_spike_train(self):
        return SpikeTrain(self.t, np.sum(self.mask, 1)[:, None] / self.nsweeps)

    def average_dot(self, st, kernel1, kernel2):

        average_spike_train_self = self.average_spike_train()
        average_spike_train_st = st.average_spike_train()

        return average_spike_train_self.dot(average_spike_train_st, kernel1, kernel2)[0]

    def population_norm(self, kernel1, kernel2, biased=True):

        average_spike_train = self.average_spike_train()

        if biased or self.nsweeps == 1:
            population_norm = average_spike_train.norm(kernel1, kernel2)[0]

        else:
            dot_sum_all = self.nsweeps ** 2. * average_spike_train.dot(average_spike_train, kernel1, kernel2)[
                0]  # ij & ji
            dot_sum_ii = np.sum(self.dot(self, kernel1, kernel2))
            population_norm = np.sqrt((dot_sum_all - dot_sum_ii) / (self.nsweeps * (self.nsweeps - 1.)))

        return population_norm

    def cosine(self, st, kernel1, kernel2):

        self_conv1 = kernel1.convolve_continuous(self.t, self.mask / self.dt)
        self_conv2 = kernel2.convolve_continuous(self.t, self.mask / self.dt)
        st_conv1 = kernel1.convolve_continuous(self.t, st.mask / self.dt)
        st_conv2 = kernel2.convolve_continuous(self.t, st.mask / self.dt)

        self_norm = np.sqrt(np.sum(self_conv1 * self_conv2, 0) * self.dt)
        st_norm = np.sqrt(np.sum(st_conv1 * st_conv2, 0) * self.dt)
        dot = np.sum(self_conv1 * st_conv2, 0) * self.dt

        return dot / (self_norm * st_norm)

    def convolve(self, kernel):
        return kernel.convolve_continuous(self.t, self.mask / self.dt)

    def get_PSTH(self, kernel, average_sweeps=True):

        if average_sweeps:
            average_spike_train = self.average_spike_train()
            psth = average_spike_train.convolve(kernel)

        else:
            psth = self.convolve(kernel)

        return psth

    def dot_product_matrix(self, st, kernel1, kernel2):

        # OJO SI TIENEN DIFERENTE dt self e ic
        # N1<=N2
        # dot_product_matrix[i, j] = <Si, Sj>

        Nself = self.nsweeps
        Nst = st.nsweeps

        if Nself <= Nst:
            N1, N2 = Nself, Nst
            mask_spk1, mask_spk2 = self.mask, st.mask
        else:
            N2, N1 = Nself, Nst
            mask_spk2, mask_spk1 = self.mask, st.mask

        index1 = np.arange(0, N1)
        dot_product_matrix = np.zeros((N1, N2)) * np.nan

        for ii in range(N2):
            index2 = (index1 + ii) % N2

            st_self = SpikeTrain(self.t, mask_spk1)
            st2 = SpikeTrain(self.t, np.roll(mask_spk2, -ii, axis=1)[:, :N1])
            dot_product_matrix[(index1, index2)] = st_self.dot(st2, kernel1, kernel2)

        if Nself > Nst:
            dot_product_matrix = dot_product_matrix.T

        return dot_product_matrix

    def cosine_matrix(self, st, kernel1, kernel2):

        dot_product_matrix = self.dot_product_matrix(st, kernel1, kernel2)

        self_norm, st_norm = self.norm(kernel1, kernel2), st.norm(kernel1, kernel2)
        norm_product_matrix = np.outer(self_norm, st_norm)

        return dot_product_matrix / norm_product_matrix

    def Ma(self, st, kernel1, kernel2, biased=True):

        average_dot_self_st = self.average_dot(st, kernel1, kernel2)
        population_norm_self = self.population_norm(kernel1, kernel2, biased=biased)
        population_norm_st = st.population_norm(kernel1, kernel2, biased=biased)

        return average_dot_self_st / (population_norm_self * population_norm_st)

    def Md(self, st, kernel1, kernel2, biased=True):

        average_dot_self_st = self.average_dot(st, kernel1, kernel2)
        population_norm_self = self.population_norm(kernel1, kernel2, biased=biased)
        population_norm_st = st.population_norm(kernel1, kernel2, biased=biased)

        return 2. * average_dot_self_st / (population_norm_self ** 2. + population_norm_st ** 2.)

    def reliability(self, kernel1, kernel2):

        average_spike_train = self.average_spike_train()

        dot_sum_all = self.nsweeps ** 2. * average_spike_train.dot(average_spike_train, kernel1, kernel2)[0]  # ij & ji
        dot_ii = self.dot(self, kernel1, kernel2)
        population_norm_squared = (dot_sum_all - np.sum(dot_ii)) / (self.nsweeps * (self.nsweeps - 1.))

        return population_norm_squared / np.mean(dot_ii)

    def reliability2(self, kernel1, kernel2):
        # Naud et al 2011 equation 2.15
        N = self.nsweeps

        dot_product_matrix = self.dot_product_matrix(self, kernel1, kernel2)
        mean_dot_ij = 2. / (N * (N - 1.)) * np.sum(dot_product_matrix[np.triu_indices(N, k=1)])

        L = np.mean(np.diagonal(dot_product_matrix))  # average of the trials norms squared 1/nsweeps*(sum(norm**2. ) )

        return mean_dot_ij / L

    def multiple_correlation_matrix(self, sts, delta):

        # n_neurons = self.nsweeps + np.sum(st.nsweeps for st in sts)

        # correlation_matrix = np.zeros( (n_neurons, n_neurons) )

        sts = [self] + sts

        dic = {}
        for (ii, st1), (jj, st2) in itertools.combinations_with_replacement(enumerate(sts), 2):
            # print(ii, jj)
            dic[ii, jj] = st1.dot_product_matrix(st2, delta)
            dic[jj, ii] = dic[ii, jj]

        return np.block([[dic[ii, jj] for jj in range(len(sts))] for ii in range(len(sts))])

        # for ii, st in enumerate(sts):
        # correlation_matrix[ii*st.nsweeps:(ii+1)*st.nsweeps, ii*st.nsweeps:(ii+1)*st.nsweeps] = st.correlation_matrix(st, delta)

    def get_spike_count(self, bins, average_sweeps=False):
        # 08/09/2018
        # Given arbitrary time bins computes the spike counts in those bins for each sweep
        # unless average=True
        
        spk_count = np.zeros( (len(bins) - 1, self.nsweeps) )
        
        mask_spk = self.mask
        
        for sw in range(self.nsweeps):
            
            t_spk = self.t[mask_spk[:, sw]]
            
            spk_count[:,sw], _ = np.histogram(t_spk, bins=bins)
        
        if average_sweeps:
            spk_count = np.mean(spk_count, 1)
            if spk_count.size==1:
                return spk_count[0]
            else:
                return spk_count
        if len(bins) == 2:
            return spk_count
        else:
            return np.squeeze(spk_count)
        
    def fano_factor(self, bins):
        
        spk_count = self.get_spike_count(bins, average_sweeps=False)
        
        return np.var(spk_count, 1) / np.mean(spk_count, 1)
        
        