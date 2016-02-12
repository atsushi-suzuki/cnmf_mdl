import os.path
import numpy as np
from numpy.random import *
import cnmf
import plotter

npy_dir = "../npy/"
os.mkdir(npy_dir)
seed(1)
n_methods = 5
proposed = 0
aic1 = 1
bic1 = 2
aic2 = 3
bic2 = 4
n_features = 8
convolution_width = 3
convolution_max = 5
n_components = 3
loop_max = 100
convergence_threshold = 1
base_max = 10.0
n_samples = 40
base = np.random.uniform(0.0, base_max,
                         [convolution_width, n_components, n_features])
gamma_shape = 2.0
gamma_scale = 2.0
p = 0.8
actvt = np.random.gamma(gamma_shape, gamma_scale,
                        [n_samples, n_components])
X_noiseless = cnmf.CNMF.convolute(actvt, base)
X = np.random.poisson(X_noiseless)
filtre = np.random.binomial(1, p, X.shape)
factorizer = cnmf.CNMF(None, None,
                       convolution_max, gamma_shape, gamma_scale,
                       convergence_threshold, loop_max, base_max)
factorizer.fit(X, None, filtre)
np.save(npy_dir + 'code_len.npy', factorizer.code_len_result)
np.save(npy_dir + 'criterion.npy', factorizer.criterion_result)
np.save(npy_dir + 'completion.npy', factorizer.completion_result)
print(X_noiseless)
print(X)
print(filtre)
