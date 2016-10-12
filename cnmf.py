import numpy as np
from scipy import special
from scipy.misc import logsumexp

class CNMF:
    def __init__(self,
                 n_components = None, true_width = None, convolution_max = 6,
                 gamma_shape = 0.5, gamma_scale = 2.0,
                 convergence_threshold = 0.0001, loop_max = 1000,
                 base_max = 10.0, component_max = None):
        self.n_components = n_components
        self.true_width = true_width
        self.convolution_max = convolution_max
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.loop_max = loop_max
        self.convergence_threshold = convergence_threshold
        self.base_max = base_max
        self.component_max = component_max
        self.n_methods = 5
        self.proposed = 0
        self.aic1 = 1
        self.bic1 = 2
        self.aic2 = 3
        self.bic2 = 4
        self.actvt_result = []
        self.base_result = []
        self.estimate = np.zeros([self.n_methods, 2], dtype=np.int)
        self.estimate_given_width = np.zeros(self.n_methods, dtype=np.int)
        self.best_actvt = [None for i_method in range(self.n_methods)]
        self.best_actvt_given_width = [None for i_method in range(self.n_methods)]
        self.best_base = [None for i_method in range(self.n_methods)]
        self.best_base_given_width = [None for i_method in range(self.n_methods)]
        self.best_completion = np.zeros(self.n_methods)
        self.best_completion_given_width = np.zeros(self.n_methods)

    def fit(self, X, y = None, filtre = None):
        self.X = X
        if filtre == None:
            self.filtre = np.ones(X.shape)
        else:
            self.filtre = filtre
        filtre = self.filtre
        (n_samples, n_features) = X.shape
        if self.component_max == None:
            self.component_max = n_features
        convolution_max = self.convolution_max
        n_components = self.n_components
        self.code_len_result = np.float("nan")\
                               * np.ones([convolution_max + 1,
                                          self.component_max + 1,
                                          self.loop_max])
        self.loop_cnt_result = np.float("nan")\
                        * np.ones([convolution_max + 1,
                                   self.component_max + 1])
        self.criterion_result = np.float("inf")\
                                * np.ones([self.n_methods,
                                           convolution_max + 1,
                                           self.component_max + 1])
        self.completion_result = np.float("nan")\
                               * np.ones([convolution_max + 1,
                                          self.component_max + 1])
        self.actvt_result = [[None for col
                              in range(self.component_max + 1)]
                             for row in range(convolution_max + 1)]
        self.base_result = [[None for col
                              in range(self.component_max + 1)]
                             for row in range(convolution_max + 1)]
        convolution_range = []
        if self.true_width == None:
            convolution_range = range(1, self.convolution_max + 1)
        else:
            convolution_range = [self.true_width]
        component_range = []
        if self.n_components == None:
            component_range = range(1, self.component_max + 1)
        else:
            component_range = [self.n_components]
        print("convolution_range", convolution_range)
        for convolution_width in convolution_range:
            log_integral_term = self.__log_integral_term(convolution_width,
                                                         n_samples, n_features,
                                                         self.gamma_shape,
                                                         self.gamma_scale)
            print('log_integral_term', log_integral_term)
            for n_components in component_range:
                print("n_components", n_components)
                (actvt, base, code_len_transition, loop_cnt)\
                    = self.__factorize(X, filtre, n_components,
                                       convolution_width,
                                       self.gamma_shape, self.gamma_scale,
                                       self.convergence_threshold)
                self.actvt_result[convolution_width][n_components] = actvt
                self.base_result[convolution_width][n_components] = base
                self.code_len_result[convolution_width, n_components, :]\
                    = code_len_transition
                self.loop_cnt_result[convolution_width, n_components]\
                    = loop_cnt
                self.criterion_result[:, convolution_width, n_components]\
                = self.__compute_criterion(X, filtre,
                                           actvt, base,
                                           self.gamma_shape, self.gamma_scale,
                                           log_integral_term)
                self.completion_result[convolution_width, n_components]\
                = self.__evaluate_completion(X, actvt, base,
                                             self.gamma_shape, self.gamma_scale)
        self.__store_estimate()

    def __store_estimate(self):
        for i_method in range(0, self.n_methods):
            self.estimate[i_method, :]\
                = np.unravel_index(
                    np.nanargmin(self.criterion_result[i_method, :, :]),
                    self.criterion_result[i_method, :, :].shape)
            self.best_actvt[i_method]\
                = self.actvt_result\
                [self.estimate[i_method, 0]][self.estimate[i_method, 1]]
            self.best_base[i_method]\
                = self.base_result\
                [self.estimate[i_method, 0]][self.estimate[i_method, 1]]
            self.best_completion[i_method]\
                = self.completion_result\
                [self.estimate[i_method, 0]][self.estimate[i_method, 1]]
        if not (self.true_width == None):
            for i_method in range(0, self.n_methods):
                self.estimate_given_width[i_method]\
                    = np.nanargmin(self.criterion_result[i_method, self.true_width, :])
                self.best_actvt_given_width[i_method]\
                    = self.actvt_result\
                    [self.true_width][self.estimate_given_width[i_method]]
                self.best_base_given_width[i_method]\
                    = self.base_result\
                    [self.true_width][self.estimate_given_width[i_method]]
                self.best_completion_given_width[i_method]\
                    = self.completion_result\
                    [self.true_width][self.estimate_given_width[i_method]]

    def __factorize(self, X, filtre, n_components, convolution_width,
                    gamma_shape, gamma_scale, convergence_threshold):
        (n_samples, n_features) = X.shape
        base = np.random.chisquare(1.0, [convolution_width,
                                         n_components, n_features])
        actvt = np.random.gamma(gamma_shape, gamma_scale,
                                [n_samples, n_components])
        new_actvt = actvt
        new_base = base
        code_len_transition = np.nan * np.ones(self.loop_max)
        loop_cnt = 0
        for loop_idx in range(0, self.loop_max):
            new_actvt = self.__actvt_numerator(X, filtre,
                                               actvt, base, gamma_shape)\
                        / (self.__actvt_denominator(filtre, base, gamma_scale) + np.finfo(float).eps)
            # print('actvt: ', self.__code_len(X, filtre, actvt, base,
            #                                  gamma_shape, gamma_scale))
            for i_convolution in range(0, convolution_width):
                new_base[i_convolution, :, :]\
                    = (base[i_convolution, :, :]\
                       * ((self.time_shift(actvt, i_convolution).T)\
                       .dot(filtre * X / (self.convolute(actvt, base) + np.finfo(float).eps)))\
                       / ((self.time_shift(actvt, i_convolution).T)\
                       .dot(filtre) + np.finfo(float).eps))
            # print('base: ', self.__code_len(X, filtre, actvt, base,
            #                                 gamma_shape, gamma_scale))
            code_len_transition[loop_idx]\
                = self.__code_len(X, filtre, actvt, base,
                                  gamma_shape, gamma_scale)
            if (loop_idx >= 1\
                and (code_len_transition[loop_idx - 1]\
                     - code_len_transition[loop_idx])\
                / code_len_transition[loop_idx] < convergence_threshold\
                and loop_idx > 0.10 * self.loop_max)\
                or loop_idx == self.loop_max - 1:
                loop_cnt = loop_idx
                print('loop_cnt', loop_cnt)
                break;
            base = new_base
            actvt = new_actvt
        return (new_actvt, new_base, code_len_transition, loop_cnt)

    def __actvt_numerator(self, X, filtre, actvt, base, gamma_shape):
        n_samples = actvt.shape[0]
        (convolution_width, n_components, n_features) = base.shape
        ans = np.zeros([n_samples, n_components])
        for i_convolution in range(0, convolution_width):
            ans += (self.inv_time_shift(filtre * X / (self.convolute(actvt, base) + np.finfo(float).eps), i_convolution)).dot(base[i_convolution, :, :].T)
        ans = actvt * ans
        ans = (gamma_shape - 1) + ans
        return ans

    def __actvt_denominator(self, filtre, base, gamma_scale):
        n_samples = filtre.shape[0]
        (convolution_width, n_components, n_features) = base.shape
        ans = (1 / gamma_scale) * np.ones([n_samples, n_components])
        for i_convolution in range(0, convolution_width):
            ans += self.inv_time_shift(filtre, i_convolution).dot((base[i_convolution, :, :]).T)
        return ans

    def __compute_criterion(self, X, filtre, actvt, base,
                            gamma_shape, gamma_scale, log_integral_term):
        (n_samples, n_features) = X.shape
        (convolution_width, n_components, n_features) = base.shape
        criterion_value = np.float("inf") * np.ones(self.n_methods)
        code_len = self.__code_len(X, filtre, actvt, base,\
                              self.gamma_shape, self.gamma_scale)
        criterion_value[self.proposed]\
            = code_len\
                  + (convolution_width * n_components * n_features / 2)\
                  + np.log(n_samples * self.base_max / np.pi)\
                  + n_components * log_integral_term
        divergence = self.__divergence(X, filtre, actvt, base)
        criterion_value[self.aic1]\
            = divergence\
            + convolution_width * n_components * (n_features + 1)
        criterion_value[self.bic1]\
            = divergence\
            + 0.5 * convolution_width * n_components * (n_features + 1)\
            * np.log(n_samples)
        criterion_value[self.aic2]\
            = code_len\
            + convolution_width * n_components * (n_features + 1)
        criterion_value[self.bic2]\
            = code_len\
            + 0.5 * convolution_width * n_components * (n_features + 1)\
            * np.log(n_samples)
        return criterion_value

    def __evaluate_completion(self, X, actvt, base, gamma_shape, gamma_scale):
        return self.__divergence(X, np.ones(X.shape), actvt, base)

    def __is_converged(self, actvt, base, new_actvt, new_base, convergence_threshold):
        return ((self.convolute(new_actvt, new_base) - self.convolute(actvt, base)) * (self.convolute(new_actvt, new_base) - self.convolute(actvt, base))).sum() < convergence_threshold

    def __divergence(self, X, filtre, actvt, base):
        L = self.convolute(actvt, base)
        X = X + np.finfo(float).eps
        L = L + np.finfo(float).eps
        return (filtre * (X * (np.log(X) - np.log(L)) - X + L)).sum()

    def __code_len(self, X, filtre, actvt, base, gamma_shape, gamma_scale):
        return self.__divergence(X, filtre, actvt, base)\
            - ((gamma_shape - 1) * np.log(actvt)).sum()\
            + ((1 / gamma_scale) * actvt).sum()\
            + gamma_shape * np.log(gamma_scale) + special.gammaln(gamma_shape)

    def __square_error(self, X, actvt, base, gamma_shape, gamma_scale):
        L = self.convolute(actvt, base)
        return ((X - L) * (X - L)).sum()

    def __log_integral_term(self, convolution_width, n_samples, n_features,
                      gamma_shape, gamma_scale):
        n = 1000
        sample = np.zeros(n)
        for i in range(0, n):
            z = np.random.gamma(gamma_shape, gamma_scale, n_samples)
            tmp_array = np.zeros(convolution_width)
            for i_convolution in range(0, convolution_width):
                # prod *= np.power(sum(z[0:-convolution_width]), n_features / 2.0)
                tmp_array[i_convolution] = np.log(sum(z[0:-convolution_width]))
            sample[i] = sum(tmp_array) * n_features / 2.0
        return logsumexp(sample) - np.log(n)

    def __likelihood(self, X, base, gamma_shape, gamma_scale):
        (n_samples, n_features) = X.shape
        (convolution_width, n_components, n_features) = base.shape
        ans = 0.0;
        n = 1000
        for i in range(0, n):
            actvt = np.random.gamma(gamma_shape, gamma_scale, [n_samples, n_components])
            ans += np.exp(- self.__divergence(X, actvt, base) + 0.8 * sum(sum(X)))
        return - np.log(ans / n) - 0.8 * sum(sum(X))

    @classmethod
    def time_shift(cls, mat, time):
        if time == 0:
            return mat
        else:
            return np.pad(mat, ((time,0),(0,0)), mode='constant')[:-time, :]

    @classmethod
    def inv_time_shift(cls, mat, time):
        if time == 0:
            return mat
        else:
            return np.pad(mat, ((0,time),(0,0)), mode='constant')[time:, :]

    @classmethod
    def convolute(cls, actvt, base):
        convolution_width = base.shape[0]
        ans = actvt.dot(base[0,:,:])
        for i_convolution in range(1, convolution_width):
            ans += cls.time_shift(actvt, i_convolution).dot(base[i_convolution, :, :])
        return ans

    @classmethod
    def divergence(cls, X, filtre, actvt, base):
        L = cls.convolute(actvt, base)
        X = X + np.finfo(float).eps
        L = L + np.finfo(float).eps
        return (filtre * (X * (np.log(X) - np.log(L)) - X + L)).sum()
