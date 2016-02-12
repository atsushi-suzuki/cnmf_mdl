# -*- coding: utf-8 -*-
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros
from matplotlib import pylab as plt
import xml.etree.ElementTree as et
import stft
import cnmf
import stat_parser

arranged_price_mat = np.load('stat.npy')
ylabels = ['food', 'residence', 'utilities', 'household', 'clothing', 'health', 'traffic', 'education', 'hobby', 'others']
month_mean_price_mat = np.zeros(arranged_price_mat.shape)
for month in range(12):
    month_mean_price_mat[:, month::12]\
    = np.nanmean(arranged_price_mat[:, month::12], axis = 1)[:,np.newaxis].dot(np.ones([1, len(month_mean_price_mat[0, month::12])]))
price_ratio_mat = arranged_price_mat / month_mean_price_mat
category_idx_list = [0, 266, 295, 323, 402, 491, 527, 576, 632, 733, 780]
category_mat = np.zeros([len(category_idx_list) - 1, arranged_price_mat.shape[1]])
for category_idx in range(len(category_idx_list) - 1):
    category_mat[category_idx, :] = np.nanmean(price_ratio_mat[category_idx_list[category_idx]:category_idx_list[category_idx + 1], :], axis = 0)
X = abs(category_mat[:,1:] - category_mat[:,:-1]).T * 1000
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_yticklabels(ylabels)
plt.imshow(X.T, aspect = "auto", origin = "lower", interpolation='none')
plt.yticks(range(10))
plt.savefig("mat.png")
convolution_width = 12
convolution_max = 12
gamma_shape = 2.0
gamma_scale = 2.0
convergence_threshold = 0.00000001
loop_max = 1000
base_max = 100
component_max = 10
filtre = np.ones(X.shape)
print(X)
factorizer = cnmf.CNMF(None, convolution_width,
                       convolution_max,
                       gamma_shape, gamma_scale,
                       convergence_threshold,
                       loop_max, base_max, component_max)
factorizer.fit(X, None, filtre)
# print(factorizer.completion_result)
# print(factorizer.estimate_given_width)
print(factorizer.best_actvt_given_width[0])
print(factorizer.best_base_given_width[0])
completed_mat = factorizer.convolute(factorizer.best_actvt_given_width[0], factorizer.best_base_given_width[0])
print(completed_mat)
print(factorizer.best_completion_given_width)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_yticklabels(ylabels)
plt.imshow(completed_mat.T, aspect = "auto", origin = "lower", interpolation='none')
plt.yticks(range(10))
plt.savefig("completed_mat.png")
for i_base in range(factorizer.best_base_given_width[0].shape[1]):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yticklabels(ylabels)
    plt.imshow(factorizer.best_base_given_width[0][:,i_base,:].T, aspect = "auto", origin = "lower", interpolation='none')
    plt.yticks(range(10))
    plt.savefig("base_"+str(i_base)+".png")
