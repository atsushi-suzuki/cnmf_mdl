# -*- coding: utf-8 -*-
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros
from matplotlib import pylab as plt
import xml.etree.ElementTree as et
import stft
import cnmf
import stat_parser

tree_list = []
for i in range(85):
    tree_list.append(et.parse('../xml/price'+str(i)+'.xml'))
price_data = stat_parser.STAT_PARSER(tree_list)
price_mat = price_data.table[0,:,:,::-1]
# masked_price_mat = np.ma.masked_array(price_mat, np.isnan(price_mat))
# arranged_price_mat = masked_price_mat.mean(axis = 1)
arranged_price_mat = np.nanmean(price_mat, axis = 1)
np.save('stat.npy', arranged_price_mat)
plt.imshow(arranged_price_mat[:,:], aspect = "auto", origin = "lower")
plt.savefig("mat.png")
