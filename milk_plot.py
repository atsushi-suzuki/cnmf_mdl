# -*- coding: utf-8 -*-
import sys
import os.path
import math
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros
from matplotlib import pylab as plt
import matplotlib
import argparse

parser = argparse.ArgumentParser(description='plot the result of the experiment of CNMF for artificial data.')
parser.add_argument('-d', '--dat_dir', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default='./npy', \
                    type=str, \
                    choices=None, \
                    help='directory where data are stored', \
                    metavar=None)
parser.add_argument('-r', '--n_rows', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=2, \
                    type=int, \
                    choices=None, \
                    help='the number of rows', \
                    metavar=None)
parser.add_argument('-w', '--width', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=28, \
                    type=int, \
                    choices=None, \
                    help='width', \
                    metavar=None)
parser.add_argument('-he', '--height', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=10, \
                    type=int, \
                    choices=None, \
                    help='height', \
                    metavar=None)
parser.add_argument('-f', '--font_size', \
                    action='store', \
                    nargs='?', \
                    const=None, \
                    default=24, \
                    type=int, \
                    choices=None, \
                    help='font size', \
                    metavar=None)
args = parser.parse_args()
# argv = sys.argv
# argc = len(argv)
# if argc != 6:
#     error()
dat_dir = args.dat_dir
n_rows = args.n_rows
width = args.width
height = args.height
font_size = args.font_size
best_base = np.load(dat_dir + '/base.npy')
print(best_base.shape)
n_cols = math.ceil(best_base.shape[1] / n_rows)
ylabels = ['milk (delivery)', 'milk (store)', 'milk powder', 'butter', 'cheese (Japanese)', 'cheese (imported)', 'yogurt']
# fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
# for i_base in range(best_base.shape[1]):
#     ax = axes.flat[i_base]
#     plt.xticks(range(best_base.shape[0]))
#     plt.yticks(range(7))
#     ax.set_xticklabels(range(-1, best_base.shape[0]))
#     ax.set_yticklabels(ylabels)
#     im = ax.imshow(best_base[:,i_base,:].T, aspect = 1, origin = 0, interpolation='none', clim=(0, np.max(best_base)))
# cax,kw = matplotlib.colorbar.make_axes([ax for ax in axes.flat])
# plt.colorbar(im, cax=cax, **kw)
fig = plt.figure(figsize=(width, height))
for i_base in range(best_base.shape[1]):
    ax = fig.add_subplot(n_rows, n_cols, i_base + 1)
    if i_base % n_cols == 0:
        plt.yticks(range(7))
        ax.set_yticklabels(ylabels, fontsize = font_size)
        ax.set_ylabel('item', fontsize = font_size)
    else:
        plt.yticks(())
    plt.xticks(range(best_base.shape[0]))
    ax.set_xticklabels(range(best_base.shape[0]), fontsize = font_size)
    ax.set_xlabel('time / month', fontsize = font_size)
    im = ax.imshow(best_base[:,i_base,:].T, aspect = 1, origin = 0, interpolation='none', clim=(0, np.max(best_base)), cmap=plt.cm.binary)
cax,kw = matplotlib.colorbar.make_axes(fig.axes)
cbar = fig.colorbar(im, cax=cax, **kw)
cbar.ax.tick_params(labelsize=font_size)
plt.savefig(dat_dir+'/base.png')
