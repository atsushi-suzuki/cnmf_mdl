# -*- coding: utf-8 -*-
# ==================================
#
#    Short Time Fourier Trasform
#
# ==================================
from scipy import ceil, complex64, float64, hamming, zeros
from scipy.fftpack import fft# , ifft
from scipy import ifft # こっちじゃないとエラー出るときあった気がする
from scipy.io.wavfile import read
from matplotlib import pylab as pl

class STFT:
    def __init__(self, win = hamming(1024), step = 512):
        self.win = win
        self.step = step

    def read_wav(self, wav_file):
        self.fs, self.wav = read(wav_file)
        return self.wav

    def get_power_spec(self, wav):
        spec = self.stft(wav)
        power_spec = abs(spec) * abs(spec)
        win = self.win
        N = len(win)
        return power_spec[:, :(N/2)+1]

    def stft(self, x):
        l = len(x) # 入力信号の長さ
        win = self.win
        N = len(win) # 窓幅、つまり切り出す幅
        step = self.step
        M = int(ceil(float(l - N + step) / step)) # スペクトログラムの時間フレーム数
        new_x = zeros(N + ((M - 1) * step), dtype = float64)
        new_x[: l] = x # 信号をいい感じの長さにする
        # print(M)
        # print(N)
        X = zeros([M, N], dtype = complex64) # スペクトログラムの初期化(複素数型)
        for m in range(M):
            start = step * m
            X[m, :] = fft(new_x[start : start + N] * win)
        return X

    def istft(self, X):
        M, N = X.shape
        win = self.win
        assert (len(win) == N), "FFT length and window length are different."
        step = self.step
        l = (M - 1) * step + N
        x = zeros(l, dtype = float64)
        wsum = zeros(l, dtype = float64)
        for m in range(M):
            start = step * m
            ### 滑らかな接続
            x[start : start + N] = x[start : start + N] + ifft(X[m, :]).real * win
            wsum[start : start + N] += win ** 2

        pos = (wsum != 0)
        x_pre = x.copy()
        ### 窓分のスケール合わせ
        x[pos] /= wsum[pos]
        return x

    def plot_wav(self):
        fig = pl.figure()
        pl.plot(self.wav)
        pl.xlim([0, len(self.wav)])
        pl.title("Input signal", fontsize = 20)
        pl.savefig("wav.png")

    def plot_spec(self, power_spec):
        fig = pl.figure()
        pl.imshow(power_spec.T, aspect = "auto", origin = "lower")
        pl.savefig("spec.png")
