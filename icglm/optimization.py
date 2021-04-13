import time

import numpy as np
from scipy.linalg import solveh_banded


class NewtonMethod:

    def __init__(self, theta0=None, g_log_prior=None, h_log_prior=None, gh_log_prior=None, gh_log_likelihood=None,
                 banded_h=False, theta_independent_h=False, learning_rate=1e-1, initial_learning_rate=1e-2,
                 max_iterations=200, stop_cond=5e-4, learning_rate_scaling=0.2, warm_up_iterations=2, verbose=False):

        self.theta0 = theta0

        self.g_log_prior = g_log_prior
        self.h_log_prior = h_log_prior
        self.gh_log_prior = gh_log_prior
        self.gh_log_likelihood = gh_log_likelihood

        self.use_prior = True if self.gh_log_prior is not None or self.g_log_prior is not None else False
        self.banded_h = banded_h
        self.theta_independent_h = theta_independent_h

        self.learning_rate = learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.stop_cond = stop_cond
        self.learning_rate_scaling = learning_rate_scaling
        self.warm_up_iterations = warm_up_iterations
        self.verbose = verbose

        self.theta_iterations = []
        self.log_prior_iterations = []
        self.log_posterior_iterations = []
        self.g_log_posterior = None
        self.h_log_posterior = None
        self.fit_status = None

    def optimize(self):

        log_prior = np.nan
        theta = self.theta0
        last_iteration_monotonic = baseline_lr = True
        status = ''
        converged = nan_parameters = False
        if self.theta_independent_h:
            h_log_prior = self.h_log_prior(theta)
            max_prior_band = h_log_prior.shape[0]

        if self.verbose:
            print('Starting gradient ascent... \n')

        t0 = time.time()

        for ii in range(self.max_iterations):

            log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(theta)
            max_likelihood_band = h_log_likelihood.shape[0]

            if self.use_prior:
                if self.theta_independent_h:
                    log_prior, g_log_prior = self.g_log_prior(theta)
                else:
                    log_prior, g_log_prior, h_log_prior = self.gh_log_prior(theta)
                log_posterior = log_likelihood + log_prior
                g_log_posterior = g_log_likelihood + g_log_prior
                if self.banded_h:
                    if max_likelihood_band >= max_prior_band:
                        h_log_posterior = h_log_likelihood
                        h_log_posterior[:max_prior_band, :] += h_log_prior
                    else:
                        h_log_posterior = h_log_prior
                        h_log_posterior[:max_likelihood_band, :] += h_log_likelihood
                else:
                    h_log_posterior = h_log_likelihood + h_log_prior
            else:
                log_posterior = log_likelihood
                g_log_posterior = g_log_likelihood
                h_log_posterior = h_log_likelihood

            if self.verbose:
                print('\r', 'Iteration {} of {}'.format(ii, self.max_iterations), '|',
                      'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), '|',
                      'log_prior={}'.format(np.round(log_prior, 2)), '|',
                      'log_posterior={}'.format(np.round(log_posterior, 2)), '\t', end='')

            self.log_prior_iterations += [log_prior]
            self.log_posterior_iterations += [log_posterior]
            self.theta_iterations += [theta]

            old_log_posterior = np.nan if len(self.log_posterior_iterations) < 2 else self.log_posterior_iterations[-2]
            diff_log_posterior = log_posterior - old_log_posterior
#             print(np.abs(diff_log_posterior / old_log_posterior))
            if ii > self.warm_up_iterations and baseline_lr and np.abs(diff_log_posterior / old_log_posterior) < self.stop_cond:
                status += '\n Iteration {} of {} | Converged | '.format(ii, self.max_iterations)
                converged = True
                nan_parameters = False
                n_iterations = ii + 1
                break
            elif np.any(np.isnan(theta)):
                status += "\n There are nan parameters. "
                nan_parameters = True
                converged = False
                n_iterations = ii + 1
                break
            elif ii == self.max_iterations - 1:
                status += '\n Not converged after {} iterations. '.format(self.max_iterations)
                converged = False
                nan_parameters = False
                n_iterations = ii + 1
            # print('\n', theta[0], theta[1], log_posterior, np.min(g_log_posterior), np.max(g_log_posterior))
            if len(self.log_posterior_iterations) == 1 or diff_log_posterior > 0:

                if last_iteration_monotonic:
                    if ii >= self.warm_up_iterations:
                        learning_rate = self.learning_rate
                        baseline_lr = True
                    else:
                        learning_rate = self.initial_learning_rate
                old_theta = np.copy(theta)
                old_g_log_posterior = np.copy(g_log_posterior)
                old_h_log_posterior = np.copy(h_log_posterior)
                if self.banded_h:
                    # the minus in h_log_posterior comes from the need of it being positive-definite.
                    # h_log_posterior isnt -h_log_posterior is
                    try:
                        theta = theta + learning_rate * solveh_banded(-h_log_posterior, g_log_posterior, lower=True)
                    except np.linalg.LinAlgError as e:
                        theta = np.zeros(len(theta)) * np.nan

                else:
                    theta = theta - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
                    # print('\n', theta[1])
                    # theta = theta + learning_rate * g_log_posterior
                last_iteration_monotonic = True
            else:

                self.log_prior_iterations = self.log_prior_iterations[:-1] + [self.log_prior_iterations[-2]]
                self.log_posterior_iterations = self.log_posterior_iterations[:-1] + [self.log_posterior_iterations[-2]]
                self.theta_iterations = self.theta_iterations[:-1] + [self.theta_iterations[-2]]
                learning_rate = learning_rate * self.learning_rate_scaling
                baseline_lr = False
                if self.banded_h:
                    try:
                        theta = old_theta + learning_rate * solveh_banded(-old_h_log_posterior, old_g_log_posterior, lower=True)
                    except np.linalg.LinAlgError as e:
                        theta = np.zeros(len(theta)) * np.nan
                else:
                    theta = old_theta - learning_rate * np.linalg.solve(old_h_log_posterior, old_g_log_posterior)
                    # theta = old_theta + learning_rate * g_log_posterior
                last_iteration_monotonic = False

        fitting_time = (time.time() - t0) / 60.

        status += 'Elapsed time: {} minutes | '.format(np.round(fitting_time, 4))

        if nan_parameters:
            log_posterior_monotonic = None
        elif np.any(np.diff(self.log_posterior_iterations) < 0.):
            status += 'Log posterior is not monotonic \n'
            log_posterior_monotonic = False
        else:
            status += 'Log posterior is monotonic \n'
            log_posterior_monotonic = True

        if self.verbose:
            print('\n', status)

        self.fit_status = {'n_iterations': n_iterations, 'converged': converged, 'nan_parameters': nan_parameters,
                           'fitting_time': fitting_time, 'log_posterior_monotonic': log_posterior_monotonic,
                           'status': status}

        self.theta_iterations = np.stack(self.theta_iterations, 1)
        self.log_posterior_iterations = np.array(self.log_posterior_iterations)
        self.log_prior_iterations = np.array(self.log_prior_iterations)
        self.g_log_posterior = g_log_posterior
        self.h_log_posterior = h_log_posterior

        return self

    
class NewtonRaphson:

    def __init__(self, theta0=None, g_log_prior=None, h_log_prior=None, gh_log_prior=None, gh_log_likelihood=None,
                 banded_h=False, theta_independent_h=False, learning_rate=1e-1, initial_learning_rate=1e-2,
                 max_iterations=200, stop_cond=5e-4, learning_rate_scaling=0.2, warm_up_iterations=2, verbose=False):

        self.theta0 = theta0

        self.g_log_prior = g_log_prior
        self.h_log_prior = h_log_prior
        self.gh_log_prior = gh_log_prior
        self.gh_log_likelihood = gh_log_likelihood

        self.use_prior = True if self.gh_log_prior is not None or self.g_log_prior is not None else False
        self.banded_h = banded_h
        self.theta_independent_h = theta_independent_h

        self.learning_rate = learning_rate
        self.initial_learning_rate = initial_learning_rate
        self.max_iterations = max_iterations
        self.stop_cond = stop_cond
        self.learning_rate_scaling = learning_rate_scaling
        self.warm_up_iterations = warm_up_iterations
        self.verbose = verbose

        self.theta_iterations = []
        self.log_prior_iterations = []
        self.log_posterior_iterations = []
        self.g_log_posterior = None
        self.h_log_posterior = None
        self.fit_status = None

    def optimize(self):

        log_prior = np.nan
        theta = self.theta0
        last_iteration_normal = baseline_lr = True
        status = ''
        converged = nan_parameters = False
        if self.theta_independent_h:
            h_log_prior = self.h_log_prior(theta)
            max_prior_band = h_log_prior.shape[0]

        if self.verbose:
            print('Starting gradient ascent... \n')

        t0 = time.time()

        for ii in range(self.max_iterations):

            log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(theta)
            max_likelihood_band = h_log_likelihood.shape[0]

            if self.use_prior:
                if self.theta_independent_h:
                    log_prior, g_log_prior = self.g_log_prior(theta)
                else:
                    log_prior, g_log_prior, h_log_prior = self.gh_log_prior(theta)
                log_posterior = log_likelihood + log_prior
                g_log_posterior = g_log_likelihood + g_log_prior
                if self.banded_h:
                    if max_likelihood_band >= max_prior_band:
                        h_log_posterior = h_log_likelihood
                        h_log_posterior[:max_prior_band, :] += h_log_prior
                    else:
                        h_log_posterior = h_log_prior
                        h_log_posterior[:max_likelihood_band, :] += h_log_likelihood
                else:
                    h_log_posterior = h_log_likelihood + h_log_prior
            else:
                log_posterior = log_likelihood
                g_log_posterior = g_log_likelihood
                h_log_posterior = h_log_likelihood

            if self.verbose:
                print('\r', 'Iteration {} of {}'.format(ii, self.max_iterations), '|',
                      'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), '|',
                      'log_prior={}'.format(np.round(log_prior, 2)), '|',
                      'log_posterior={}'.format(np.round(log_posterior, 2)), end='')

            self.log_prior_iterations += [log_prior]
            self.log_posterior_iterations += [log_posterior]
            self.theta_iterations += [theta]

            old_log_posterior = np.nan if len(self.log_posterior_iterations) < 2 else self.log_posterior_iterations[-2]
            diff_log_posterior = log_posterior - old_log_posterior
#             print(np.abs(diff_log_posterior / old_log_posterior))
#             cond = np.mean(np.abs(np.diff(self.log_posterior_iterations[-4:]) / np.array(self.log_posterior_iterations[-4:-1])))
            cond = np.abs((np.mean(self.log_posterior_iterations[-2:]) - np.mean(self.log_posterior_iterations[-4:-2])) / np.mean(self.log_posterior_iterations[-4:-2]))
#             print(cond, self.stop_cond)
            if ii > self.warm_up_iterations and baseline_lr and cond < self.stop_cond:
                status += '\n Iteration {} of {} | Converged | '.format(ii, self.max_iterations)
                converged = True
                nan_parameters = False
                n_iterations = ii + 1
                break
            elif np.any(np.isnan(theta)):
                status += "\n There are nan parameters. "
                nan_parameters = True
                converged = False
                n_iterations = ii + 1
                break
            elif ii == self.max_iterations - 1:
                status += '\n Not converged after {} iterations. '.format(self.max_iterations)
                converged = False
                nan_parameters = False
                n_iterations = ii + 1
            # print('\n', theta[0], theta[1], log_posterior, np.min(g_log_posterior), np.max(g_log_posterior))
            if len(self.log_posterior_iterations) == 1 or np.abs(diff_log_posterior / old_log_posterior) < 0.1:

                if last_iteration_normal:
                    if ii >= self.warm_up_iterations:
                        learning_rate = self.learning_rate
                        baseline_lr = True
                    else:
                        learning_rate = self.initial_learning_rate
                old_theta = np.copy(theta)
                old_g_log_posterior = np.copy(g_log_posterior)
                old_h_log_posterior = np.copy(h_log_posterior)
                if self.banded_h:
                    # the minus in h_log_posterior comes from the need of it being positive-definite.
                    # h_log_posterior isnt -h_log_posterior is
                    try:
                        theta = theta + learning_rate * solveh_banded(-h_log_posterior, g_log_posterior, lower=True)
                    except np.linalg.LinAlgError as e:
                        theta = np.zeros(len(theta)) * np.nan

                else:
                    theta = theta - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
                    # print('\n', theta[1])
                    # theta = theta + learning_rate * g_log_posterior
                last_iteration_normal = True
            else:
                self.log_prior_iterations = self.log_prior_iterations[:-1] + [self.log_prior_iterations[-2]]
                self.log_posterior_iterations = self.log_posterior_iterations[:-1] + [self.log_posterior_iterations[-2]]
                self.theta_iterations = self.theta_iterations[:-1] + [self.theta_iterations[-2]]
                learning_rate = learning_rate * self.learning_rate_scaling
                baseline_lr = False
                if self.banded_h:
                    try:
                        theta = old_theta + learning_rate * solveh_banded(-old_h_log_posterior, old_g_log_posterior, lower=True)
                    except np.linalg.LinAlgError as e:
                        theta = np.zeros(len(theta)) * np.nan
                else:
                    theta = old_theta - learning_rate * np.linalg.solve(old_h_log_posterior, old_g_log_posterior)
                    # theta = old_theta + learning_rate * g_log_posterior
                last_iteration_normal = False

        fitting_time = (time.time() - t0) / 60.

        status += 'Elapsed time: {} minutes | '.format(np.round(fitting_time, 4))

        if nan_parameters:
            log_posterior_monotonic = None
        elif np.any(np.diff(self.log_posterior_iterations) < 0.):
            status += 'Log posterior is not monotonic \n'
            log_posterior_monotonic = False
        else:
            status += 'Log posterior is monotonic \n'
            log_posterior_monotonic = True

        if self.verbose:
            print('\n', status)

        self.fit_status = {'n_iterations': n_iterations, 'converged': converged, 'nan_parameters': nan_parameters,
                           'fitting_time': fitting_time, 'log_posterior_monotonic': log_posterior_monotonic,
                           'status': status}

        self.theta_iterations = np.stack(self.theta_iterations, 1)
        self.log_posterior_iterations = np.array(self.log_posterior_iterations)
        self.log_prior_iterations = np.array(self.log_prior_iterations)
        self.g_log_posterior = g_log_posterior
        self.h_log_posterior = h_log_posterior

        return self

    

# class NewtonRaphson:

#     def __init__(self, theta0=None, g_log_prior=None, h_log_prior=None, gh_log_prior=None, gh_log_likelihood=None,
#                  banded_h=False, theta_independent_h=False, learning_rate=1e-1, initial_learning_rate=1e-2,
#                  max_iterations=200, stop_cond=5e-4, learning_rate_scaling=0.2, warm_up_iterations=2, verbose=False):

#         self.theta0 = theta0

#         self.g_log_prior = g_log_prior
#         self.h_log_prior = h_log_prior
#         self.gh_log_prior = gh_log_prior
#         self.gh_log_likelihood = gh_log_likelihood

#         self.use_prior = True if self.gh_log_prior is not None or self.g_log_prior is not None else False
#         self.banded_h = banded_h
#         self.theta_independent_h = theta_independent_h

#         self.learning_rate = learning_rate
#         self.initial_learning_rate = initial_learning_rate
#         self.max_iterations = max_iterations
#         self.stop_cond = stop_cond
#         self.learning_rate_scaling = learning_rate_scaling
#         self.warm_up_iterations = warm_up_iterations
#         self.verbose = verbose

#         self.theta_iterations = []
#         self.log_prior_iterations = []
#         self.log_posterior_iterations = []
#         self.g_log_posterior = None
#         self.h_log_posterior = None
#         self.fit_status = None

#     def optimize(self):

#         log_prior = np.nan
#         theta = self.theta0
#         last_iteration_monotonic = baseline_lr = True
#         status = ''
#         converged = nan_parameters = False
#         if self.theta_independent_h:
#             h_log_prior = self.h_log_prior(theta)
#             max_prior_band = h_log_prior.shape[0]

#         if self.verbose:
#             print('Starting gradient ascent... \n')

#         t0 = time.time()

#         for ii in range(self.max_iterations):

#             log_likelihood, g_log_likelihood, h_log_likelihood = self.gh_log_likelihood(theta)
#             max_likelihood_band = h_log_likelihood.shape[0]

#             if self.use_prior:
#                 if self.theta_independent_h:
#                     log_prior, g_log_prior = self.g_log_prior(theta)
#                 else:
#                     log_prior, g_log_prior, h_log_prior = self.gh_log_prior(theta)
#                 log_posterior = log_likelihood + log_prior
#                 g_log_posterior = g_log_likelihood + g_log_prior
#                 if self.banded_h:
#                     if max_likelihood_band >= max_prior_band:
#                         h_log_posterior = h_log_likelihood
#                         h_log_posterior[:max_prior_band, :] += h_log_prior
#                     else:
#                         h_log_posterior = h_log_prior
#                         h_log_posterior[:max_likelihood_band, :] += h_log_likelihood
#                 else:
#                     h_log_posterior = h_log_likelihood + h_log_prior
#             else:
#                 log_posterior = log_likelihood
#                 g_log_posterior = g_log_likelihood
#                 h_log_posterior = h_log_likelihood

#             if self.verbose:
#                 print('\r', 'Iteration {} of {}'.format(ii, self.max_iterations), '|',
#                       'Elapsed time: {} seconds'.format(np.round(time.time() - t0, 2)), '|',
#                       'log_prior={}'.format(np.round(log_prior, 2)), '|',
#                       'log_posterior={}'.format(np.round(log_posterior, 2)), end='')

#             self.log_prior_iterations += [log_prior]
#             self.log_posterior_iterations += [log_posterior]
#             self.theta_iterations += [theta]

#             old_log_posterior = np.nan if len(self.log_posterior_iterations) < 2 else self.log_posterior_iterations[-2]
#             diff_log_posterior = log_posterior - old_log_posterior
# #             print(np.abs(diff_log_posterior / old_log_posterior), self.stop_cond)
#             cond = np.mean(np.abs(np.diff(self.log_posterior_iterations[-4:]) / np.array(self.log_posterior_iterations[-4:-1])))
# #             if ii > self.warm_up_iterations and baseline_lr and np.abs(diff_log_posterior / old_log_posterior) < self.stop_cond:
#             if ii > self.warm_up_iterations and baseline_lr and cond < self.stop_cond:
#                 status += '\n Iteration {} of {} | Converged | '.format(ii, self.max_iterations)
#                 converged = True
#                 nan_parameters = False
#                 n_iterations = ii + 1
#                 break
#             elif ii == self.max_iterations - 1:
#                 status += '\n Not converged after {} iterations. '.format(self.max_iterations)
#                 converged = False
#                 nan_parameters = False
#                 n_iterations = ii + 1
            
#             if len(self.log_posterior_iterations) == 1 or not(np.any(np.isnan(theta))):
                
#                 if ii >= self.warm_up_iterations:
#                     learning_rate = self.learning_rate
#                 else:
#                     learning_rate = self.initial_learning_rate
                        
#                 old_theta = np.copy(theta)
#                 old_g_log_posterior = np.copy(g_log_posterior)
#                 old_h_log_posterior = np.copy(h_log_posterior)
#                 if self.banded_h:
#                     try:
#                         theta = theta + learning_rate * solveh_banded(-h_log_posterior, g_log_posterior, lower=True)
#                     except np.linalg.LinAlgError as e:
#                         theta = np.zeros(len(theta)) * np.nan

#                 else:
#                     theta = theta - learning_rate * np.linalg.solve(h_log_posterior, g_log_posterior)
#             else:
#                 self.log_prior_iterations = self.log_prior_iterations[:-1] + [self.log_prior_iterations[-2]]
#                 self.log_posterior_iterations = self.log_posterior_iterations[:-1] + [self.log_posterior_iterations[-2]]
#                 self.theta_iterations = self.theta_iterations[:-1] + [self.theta_iterations[-2]]
#                 learning_rate = learning_rate * self.learning_rate_scaling
#                 if self.banded_h:
#                     try:
#                         theta = old_theta + learning_rate * solveh_banded(-old_h_log_posterior, old_g_log_posterior, lower=True)
#                     except np.linalg.LinAlgError as e:
#                         theta = np.zeros(len(theta)) * np.nan
#                 else:
#                     theta = old_theta - learning_rate * np.linalg.solve(old_h_log_posterior, old_g_log_posterior)
#                     # theta = old_theta + learning_rate * g_log_posterior
                

#         fitting_time = (time.time() - t0) / 60.

#         if np.any(np.isnan(theta)):
#             status += "\n There are nan parameters. "
#             nan_parameters = True
#             converged = False
#             n_iterations = ii + 1
        
#         status += 'Elapsed time: {} minutes | '.format(np.round(fitting_time, 4))
        
#         if nan_parameters:
#             log_posterior_monotonic = None
#         elif np.any(np.diff(self.log_posterior_iterations) < 0.):
#             status += 'Log posterior is not monotonic \n'
#             log_posterior_monotonic = False
#         else:
#             status += 'Log posterior is monotonic \n'
#             log_posterior_monotonic = True

#         if self.verbose:
#             print('\n', status)

#         self.fit_status = {'n_iterations': n_iterations, 'converged': converged, 'nan_parameters': nan_parameters,
#                            'fitting_time': fitting_time, 'log_posterior_monotonic': log_posterior_monotonic,
#                            'status': status}

#         self.theta_iterations = np.stack(self.theta_iterations, 1)
#         self.log_posterior_iterations = np.array(self.log_posterior_iterations)
#         self.log_prior_iterations = np.array(self.log_prior_iterations)
#         self.g_log_posterior = g_log_posterior
#         self.h_log_posterior = h_log_posterior

#         return self
