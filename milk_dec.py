# -*- coding: utf-8 -*-
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros
from matplotlib import pylab as plt
import xml.etree.ElementTree as et
import stft
import cnmf
import stat_parser

dat_dir = '../dat/milk/'
arranged_price_mat = np.load('stat.npy')
ylabels = ['milk (delivery)', 'milk (store)', 'milk powder', 'butter', 'cheese (Japanese)', 'cheese (imported)', 'yogurt']
mean_price_mat = np.nanmean(arranged_price_mat, axis = 1)[:,np.newaxis].dot(np.ones([1, arranged_price_mat.shape[1]]))
# month_mean_price_mat = np.zeros(arranged_price_mat.shape)
# for month in range(12):
#     month_mean_price_mat[:, month::12]\
#     = np.nanmean(arranged_price_mat[:, month::12], axis = 1)[:,np.newaxis].dot(np.ones([1, len(month_mean_price_mat[0, month::12])]))
price_ratio_mat = arranged_price_mat / mean_price_mat
X = np.log(abs(price_ratio_mat[67:74,1:] - price_ratio_mat[67:74,:-1]).T + 1) * 1000
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_yticklabels(ylabels)
im = plt.imshow(X.T, aspect = "auto", origin = "lower", interpolation='none')
plt.colorbar(im)
plt.yticks(range(7))
plt.savefig(dat_dir + "mat.png")
convolution_width = 1
convolution_max = 1
gamma_shape = 1.0001
gamma_scale = 2.0
convergence_threshold = 0.0001
loop_max = 1000
base_max = np.max(X)
component_max = X.shape[1]
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
im = plt.imshow(completed_mat.T, aspect = "auto", origin = "lower", interpolation='none')
plt.colorbar(im)
plt.yticks(range(7))
plt.savefig(dat_dir + "completed_mat.png")
for i_base in range(factorizer.best_base_given_width[0].shape[1]):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_yticklabels(ylabels)
    im = plt.imshow(factorizer.best_base_given_width[0][:,i_base,:].T, aspect = 1, origin = "lower", interpolation='none', clim=(0, np.max(factorizer.best_base_given_width[0])))
    plt.colorbar(im)
    plt.yticks(range(7))
    plt.savefig(dat_dir + "base_"+str(i_base)+".png")
np.save(dat_dir + 'base.npy', factorizer.best_base_given_width[0])
