import os
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, n_samples_array, n_components):
        self.n_samples_array = n_samples_array
        self.n_components = n_components
        self.proposed = 0
        self.aic1 = 1
        self.bic1 = 2
        self.aic2 = 3
        self.bic2 = 4
        self.npy_dir = "../npy"
        self.img_dir = "../img"
        self.legend_loc = "upper left"
        self.font_size = 24
        os.mkdir(self.img_dir)
        matplotlib.use('Agg')

    def plot_accuracy(self):
        estimate_given_width = np.load(self.npy_dir\
                                       + "/estimate_given_width.npy")
        n_trials = estimate_given_width.shape[1]
        accuracy = np.transpose(
            np.sum((1.0 / n_trials)\
                   * (estimate_given_width == self.n_components),
                   axis = 1),
            [1, 0])
        n_samples_array = self.n_samples_array
        plt.plot(n_samples_array, accuracy[self.aic1], '--ob', label = "AIC1")
        plt.plot(n_samples_array, accuracy[self.bic1], '--og', label = "BIC1")
        plt.plot(n_samples_array, accuracy[self.aic2], '--or', label = "AIC2")
        plt.plot(n_samples_array, accuracy[self.bic2], '--oy', label = "BIC2")
        plt.plot(n_samples_array, accuracy[self.proposed], '-om',
                 label = "Proposed")
        plt.legend(loc=self.legend_loc)
        plt.title("Accuracy Rate in Estimating # Base", fontsize=self.font_size)
        plt.xlabel("# Samples", fontsize=self.font_size)
        plt.ylabel("Accuracy Rate", fontsize=self.font_size)
        plt.savefig(self.img_dir + "/accuracy.png")
        plt.close()

    def plot_benefit(self):
        estimate_given_width = np.load(self.npy_dir\
                                       + "/estimate_given_width.npy")
        n_trials = estimate_given_width.shape[1]
        benefit = np.transpose(
            np.sum((1.0 / n_trials)
                   * (1 - np.maximum(np.zeros(
                       estimate_given_width.shape),
                                     np.abs(estimate_given_width
                                            - self.n_components) / 2.0)),
                   axis = 1),
            [1, 0])
        n_samples_array = self.n_samples_array
        plt.plot(n_samples_array, benefit[self.aic1], '--ob', label = "AIC1")
        plt.plot(n_samples_array, benefit[self.bic1], '--og', label = "BIC1")
        plt.plot(n_samples_array, benefit[self.aic2], '--or', label = "AIC2")
        plt.plot(n_samples_array, benefit[self.bic2], '--oy', label = "BIC2")
        plt.plot(n_samples_array, benefit[self.proposed], '-om',
                 label = "Proposed")
        plt.legend(loc=self.legend_loc)
        plt.title("Benefit Rate in Estimating # Base", fontsize=self.font_size)
        plt.xlabel("# Samples", fontsize=self.font_size)
        plt.ylabel("Benefit", fontsize=self.font_size)
        plt.savefig(self.img_dir + "/benefit.png")
        plt.close()

    def plot_n_bases(self):
        estimate_given_width = np.load(self.npy_dir\
                                       + "/estimate_given_width.npy")
        n_trials = estimate_given_width.shape[1]
        mean = np.transpose(
            estimate_given_width.mean(axis=1),
            [1, 0])
        sd = np.transpose(
            estimate_given_width.std(axis=1),
            [1, 0])
        n_samples_array = self.n_samples_array
        self.__mean_sd_plot(n_samples_array, mean, sd)
        plt.legend(loc=self.legend_loc)
        plt.title("Estimated # Base", fontsize=self.font_size)
        plt.xlabel("# Samples", fontsize=self.font_size)
        plt.ylabel("# Base", fontsize=self.font_size)
        plt.savefig(self.img_dir + "/n_of_bases.png")
        plt.close()

    def plot_completion(self):
        completion_given_width = np.load(self.npy_dir\
                                       + "/best_completion_given_width.npy")
        n_trials = completion_given_width.shape[1]
        mean = np.transpose(
            completion_given_width.mean(axis=1),
            [1, 0])
        sd = np.transpose(
            completion_given_width.std(axis=1),
            [1, 0])
        n_samples_array = self.n_samples_array
        self.__mean_sd_plot(n_samples_array, mean, sd)
        plt.legend(loc=self.legend_loc)
        plt.title("Completion Error", fontsize=self.font_size)
        plt.xlabel("# Samples", fontsize=self.font_size)
        plt.ylabel("# Completion Error", fontsize=self.font_size)
        plt.savefig(self.img_dir + "/completion_error_given_width.png")
        plt.close()

    def __mean_sd_plot(self, n_samples_array, mean, sd):
        plt.errorbar(n_samples_array, mean[self.aic1], fmt='--ob',
                     yerr=sd[self.aic1], label = "AIC1")
        plt.errorbar(n_samples_array, mean[self.bic1], fmt='--og',
                     yerr=sd[self.bic1], label = "BIC1")
        plt.errorbar(n_samples_array, mean[self.aic2], fmt='--or',
                     yerr=sd[self.aic2], label = "AIC2")
        plt.errorbar(n_samples_array, mean[self.bic2], fmt='--oy',
                     yerr=sd[self.bic2], label = "BIC2")
        plt.errorbar(n_samples_array, mean[self.proposed], fmt='-om',
                     yerr=sd[self.proposed], label = "Proposed")
