# -*- coding: utf-8 -*-
import sys
import os.path
import numpy as np
from scipy import ceil, complex64, float64, hamming, zeros
from numpy.random import *
from matplotlib import pylab as plt
import stft
import cnmf

argv = sys.argv
argc = len(argv)
if argc != 3:
    error()
wav_file_name = argv[1]
seed_number = int(argv[2])
npy_dir = "../npy/"
os.mkdir(npy_dir)
seed(seed_number)

window_len = 512 # 512
band_cnt = int(window_len / 2 + 1)
step = int(window_len / 2)
transformer = stft.STFT(hamming(window_len), step)
raw_wav = np.mean(transformer.read_wav(wav_file_name), axis = 1)
# raw_wav = transformer.read_wav(wav_file_name)
raw_wav_len = raw_wav.shape[0]
test_wav_len = 44100 * 5 # 44100 * 10
test_start = np.random.randint(0, raw_wav_len - test_wav_len - 1)
test_end = test_start + test_wav_len
wav = (raw_wav[test_start:test_end])
print(wav.shape)
wav_len = wav.shape[0]
batch_cnt = int(ceil(float(wav_len - window_len + step) / step))
missed_len = int(batch_cnt / 4)
missed_start = np.random.randint(0, batch_cnt - missed_len - 1)
missed_power_spec = np.zeros([batch_cnt, band_cnt])
filtre = np.zeros([batch_cnt, band_cnt])
true_power_spec = transformer.get_power_spec(wav)
first_end_point = step * (missed_start - 1) + window_len
missed_start_point = step * missed_start
missed_end_point = missed_start_point + step * (missed_len - 1) + window_len
recovery_start_point = step * (missed_start + missed_len)
missed_power_spec[:missed_start, :]\
    = transformer.get_power_spec(wav[:first_end_point])
filtre[:missed_start, :] = 1
transformer = stft.STFT(hamming(window_len/2), step/2)
missed_power_spec[missed_start:missed_start + missed_len,\
                  :window_len / 4 + 1]\
                  = 2 * transformer.get_power_spec((wav[missed_start_point:missed_end_point])[::2])
filtre[missed_start:missed_start + missed_len, :window_len / 4 + 1]\
    = 1
transformer = stft.STFT(hamming(window_len), step)
missed_power_spec[missed_start + missed_len:, :]\
    = transformer.get_power_spec(wav[recovery_start_point:])
filtre[missed_start + missed_len:, :] = 1
# plt.matshow(filtre.T)
# plt.savefig("filtre.png")
# plt.matshow(np.log(missed_power_spec.T + 1))
# plt.savefig("missed_power_spec.png")
# plt.matshow(np.log(true_power_spec.T + 1))
# plt.savefig("true_power_spec.png")

n_methods = 5
convolution_width = 1
convolution_max = 1
gamma_shape = 1.00001
gamma_scale = 2.0
convergence_threshold = 0.00000001
loop_max = 200
base_max = 1000000000
component_max = 40 # 20
conventional_factorizer = cnmf.CNMF(None, convolution_width,
                       convolution_max,
                       gamma_shape, gamma_scale,
                       convergence_threshold,
                       loop_max, base_max, component_max)
conventional_factorizer.fit(missed_power_spec, None, filtre)
conventional_base_cnt = []
conventional_actvt = []
conventional_base = []
conventional_divergence = []
conventional_recovered_power_spec = []
for i_method in range(n_methods):
    conventional_base_cnt.append(conventional_factorizer.estimate_given_width[i_method])
    conventional_actvt.append(conventional_factorizer.best_actvt_given_width[i_method])
    conventional_base.append(conventional_factorizer.best_base_given_width[i_method])
    conventional_recovered_power_spec.append(conventional_factorizer.convolute(conventional_factorizer.best_actvt_given_width[i_method], conventional_factorizer.best_base_given_width[i_method]))
    conventional_divergence.append(conventional_factorizer.divergence(true_power_spec, np.ones(true_power_spec.shape), conventional_factorizer.best_actvt_given_width[i_method], conventional_factorizer.best_base_given_width[i_method]))

convolution_width = 100 # 100
convolution_max = 100 # 100
convolutive_factorizer = cnmf.CNMF(None, convolution_width,
                       convolution_max,
                       gamma_shape, gamma_scale,
                       convergence_threshold,
                       loop_max, base_max, component_max)
convolutive_factorizer.fit(missed_power_spec, None, filtre)
convolutive_base_cnt = []
convolutive_actvt = []
convolutive_base = []
convolutive_divergence = []
convolutive_recovered_power_spec = []
for i_method in range(n_methods):
    convolutive_base_cnt.append(convolutive_factorizer.estimate_given_width[i_method])
    convolutive_actvt.append(convolutive_factorizer.best_actvt_given_width[i_method])
    convolutive_base.append(convolutive_factorizer.best_base_given_width[i_method])
    convolutive_recovered_power_spec.append(convolutive_factorizer.convolute(convolutive_factorizer.best_actvt_given_width[i_method], convolutive_factorizer.best_base_given_width[i_method]))
    convolutive_divergence.append(convolutive_factorizer.divergence(true_power_spec, np.ones(true_power_spec.shape), convolutive_factorizer.best_actvt_given_width[i_method], convolutive_factorizer.best_base_given_width[i_method]))

np.save(npy_dir + 'seed_number.npy', seed_number)
np.save(npy_dir + 'true_power_spec.npy', true_power_spec)
np.save(npy_dir + 'conventional_base_cnt.npy', conventional_base_cnt)
np.save(npy_dir + 'conventional_actvt.npy', conventional_actvt)
np.save(npy_dir + 'conventional_base.npy', conventional_base)
np.save(npy_dir + 'conventional_recovered_power_spec.npy', conventional_recovered_power_spec)
np.save(npy_dir + 'conventional_divergence.npy', conventional_divergence)
np.save(npy_dir + 'convolutive_base_cnt.npy', convolutive_base_cnt)
np.save(npy_dir + 'convolutive_actvt.npy', convolutive_actvt)
np.save(npy_dir + 'convolutive_base.npy', convolutive_base)
np.save(npy_dir + 'convolutive_recovered_power_spec.npy', convolutive_recovered_power_spec)
np.save(npy_dir + 'convolutive_divergence.npy', convolutive_divergence)
print(convolutive_base_cnt)
print(convolutive_divergence)
print(convolutive_factorizer.criterion_result)
plt.matshow(np.log(convolutive_recovered_power_spec[0].T + 1))
plt.savefig("recovered_power_spec.png")
plt.matshow(np.log(true_power_spec.T + 1))
plt.savefig("true_power_spec.png")
