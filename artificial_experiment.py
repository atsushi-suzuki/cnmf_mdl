# -*- coding: utf-8 -*-
import sys
import os.path
import numpy as np
from numpy.random import *
import argparse
import cnmf

parser = argparse.ArgumentParser(description='Apply CNMF for artificial data.')
# argv = sys.argv
# argc = len(argv)
parser.add_argument('-s', '--seed_number', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=0, \
                    type=int, \
                    choices=None, \
                    help='seed_number', \
                    metavar=None)
parser.add_argument('-d', '--npy_dir', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default='../npy', \
                    type=str, \
                    choices=None, \
                    help='directory where data will be stored', \
                    metavar=None)
parser.add_argument('-f', '--n_features', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=0, \
                    type=int, \
                    choices=None, \
                    help='the number of features', \
                    metavar=None)
parser.add_argument('-nc', '--n_components', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=3, \
                    type=int, \
                    choices=None, \
                    help='the true number of components', \
                    metavar=None)
parser.add_argument('-w', '--convolution_width', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=3, \
                    type=int, \
                    choices=None, \
                    help='convolution width', \
                    metavar=None)
parser.add_argument('-mr', '--missing_rate', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=0.8, \
                    type=float, \
                    choices=None, \
                    help='convolution width', \
                    metavar=None)
parser.add_argument('-nt', '--n_trials', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=50, \
                    type=int, \
                    choices=None, \
                    help='the number of trials', \
                    metavar=None)
parser.add_argument('-l', '--loop_max', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=1000, \
                    type=int, \
                    choices=None, \
                    help='loop max', \
                    metavar=None)
parser.add_argument('-th', '--convergence_threshold', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=0.0001, \
                    type=float, \
                    choices=None, \
                    help='convergence_threshold', \
                    metavar=None)
parser.add_argument('-tf', '--code_len_transition_save_flag', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=0, \
                    type=int, \
                    choices=None, \
                    help='if 1, the transition of code_len is saved (very large size).', \
                    metavar=None)
parser.add_argument('-cm', '--component_max', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=12, \
                    type=int, \
                    choices=None, \
                    help='upper limit of the number of component', \
                    metavar=None)
parser.add_argument('-wm', '--convolution_max', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=12, \
                    type=int, \
                    choices=None, \
                    help='upper limit of convolution width', \
                    metavar=None)
parser.add_argument('-b', '--base_max', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=10.0, \
                    type=float, \
                    choices=None, \
                    help='base_max', \
                    metavar=None)
args = parser.parse_args()
npy_dir = args.npy_dir
n_features = args.n_features
n_components = args.n_components
convolution_width = args.convolution_width
missing_rate = args.missing_rate
n_trials = args.n_trials
loop_max = args.loop_max
convergence_threshold = args.convergence_threshold
seed_number = args.seed_number
code_len_transition_save_flag = args.code_len_transition_save_flag
component_max = args.component_max
convolution_max = args.convolution_max
base_max = args.base_max
seed(args.seed_number)

# npy_dir = "../npy/"
os.mkdir(npy_dir)
n_methods = 5
proposed = 0
aic1 = 1
bic1 = 2
aic2 = 3
bic2 = 4
# if argc != 12:
#     error()
# n_features = int(argv[1]) # 8
# print("n_features", n_features)
# n_components = int(argv[2]) # 3
# print("n_components", n_components)
# convolution_width = int(argv[3]) # 3
# missing_rate = float(argv[4]) # 0.8
# n_trials = int(argv[5]) # 50
# loop_max = int(argv[6]) # 10000
# convergence_threshold = float(argv[7]) # 0.0001
# seed_number = int(argv[8]) # 0
# code_len_transition_save_flag = int(argv[9]) # 0
# component_max = int(argv[10])
# print("component_max", component_max)
# convolution_max = int(argv[11])
# seed(seed_number)
# base_max = 10.0
n_samples_array = np.array([40, 80, 120, 160, 200, 240, 280, 320, 360, 400])
# n_samples_array = np.array([80, 160])
code_len_result = np.zeros([n_samples_array.shape[0], n_trials,
                            convolution_max + 1, component_max + 1, loop_max])
criterion_result = np.zeros([n_samples_array.shape[0],
                             n_trials, n_methods,
                             convolution_max + 1, component_max + 1])
completion_result = np.zeros([n_samples_array.shape[0],
                              n_trials, n_methods,
                              convolution_max + 1, component_max + 1])
estimate = np.zeros([n_samples_array.shape[0], n_trials, n_methods, 2])
estimate_given_width = np.zeros([n_samples_array.shape[0], n_trials, n_methods])
best_completion = np.zeros([n_samples_array.shape[0],
                                     n_trials, n_methods])
best_completion_given_width = np.zeros([n_samples_array.shape[0],
                                                 n_trials, n_methods])
for i_n_samples in range(0, n_samples_array.shape[0]):
    n_samples = n_samples_array[i_n_samples]
    for i_trial in range(0, n_trials):
        base = np.random.uniform(0.0, base_max,
                                 [convolution_width, n_components, n_features])
        gamma_shape = 2.0
        gamma_scale = 2.0
        actvt = np.random.gamma(gamma_shape, gamma_scale,
                                [n_samples, n_components])
        X_noiseless = cnmf.CNMF.convolute(actvt, base)
        X = np.random.poisson(X_noiseless).astype(float)
        X[X==0.0] = np.finfo(float).eps
        print((X==0).sum())
        filtre = np.random.binomial(1, missing_rate, X.shape)
        factorizer = cnmf.CNMF(None, convolution_width,
                               convolution_max,
                               gamma_shape, gamma_scale,
                               convergence_threshold,
                               loop_max, base_max, component_max)
        factorizer.fit(X, None, filtre)
        if code_len_transition_save_flag == 1:
            np.save(npy_dir + '/factorizer'\
                    + str(i_n_samples) + '_' + str(i_trial) + '.npy',
                    factorizer.code_len_result)
            code_len_result[i_n_samples, i_trial, :, :, :]\
                = factorizer.code_len_result
        criterion_result[i_n_samples, i_trial, :, :, :]\
            = factorizer.criterion_result
        completion_result[i_n_samples, i_trial, :, :]\
            = factorizer.completion_result
        estimate[i_n_samples, i_trial, :, :] = factorizer.estimate
        estimate_given_width[i_n_samples, i_trial, :]\
            = factorizer.estimate_given_width
        best_completion[i_n_samples, i_trial, :]\
            = factorizer.best_completion
        best_completion_given_width[i_n_samples, i_trial, :]\
            = factorizer.best_completion_given_width
if code_len_transition_save_flag == 1:
    np.save(npy_dir + '/code_len_result.npy', code_len_result)
np.save(npy_dir + '/criterion_result.npy', criterion_result)
np.save(npy_dir + '/completion_result.npy', completion_result)
np.save(npy_dir + '/estimate.npy', estimate)
np.save(npy_dir + '/estimate_given_width.npy', estimate_given_width)
np.save(npy_dir + '/best_completion.npy', best_completion)
np.save(npy_dir + '/best_completion_given_width.npy', best_completion_given_width)
