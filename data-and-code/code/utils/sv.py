import numpy as np

from optimize import l1_optimize, l2_optimize
import time

class SV:
    def __init__(self, X, Y, s, s_size, real_beta):
        self.X = X
        self.Y = Y
        self.s = s
        self.s_size = s_size
        self.real_beta = real_beta
        self.beta = None
        self.iter = 0
        self.__Y_tilde = np.ones(s_size)
        self.__f0 = None
        self.__is_fitted = False
        self.error_list = []

    def __get_bandwidth(self):
        t = self.iter
        s = self.s
        n = self.X.shape[0]
        m = self.s_size
        h = np.sqrt(s * np.log(n) / n) + s ** (-0.5) * (s**2 * np.log(n) / 10 / m) ** ((t + 1) / 2)
        return h

    @staticmethod
    def __I(y, x, beta):
        flag = y - x.dot(beta)
        return 1 if flag <= 0 else 0

    @staticmethod
    def __kernel(x):
        # if (x <= 1 and x >= -1):
        #     return -315/64*x**6 + 735/64*x**4 - 525/64*x**2 + 105/64
        # return 0 
        return np.exp(-(np.sqrt(2))*0.5*x**2)

    def __get_f0(self):
        h = self.__get_bandwidth()
        X, Y = self.__get_curXY()
        n = X.shape[0]
        beta = self.beta
        ret = 0
        for i in range(X.shape[0]):
            ret += self.__kernel((Y[i] - X[i].dot(beta))/h)
        self.__f0 = ret / (n * h)
        return ret / (n * h)

    def __get_Yi_tilde(self, Yi, Xi):
        f0 = self.__f0
        beta = self.beta
        Yi_tilde = Xi.dot(beta) - 1/f0 * (self.__I(Yi, Xi, beta) - 0.5)
        return Yi_tilde
    
    def __get_Y_tilde(self):
        X, Y = self.__get_curXY()
        Y_tilde = np.ones(Y.shape[0])
        for i in range(Y_tilde.shape[0]):
            Y_tilde[i] = self.__get_Yi_tilde(Y[i], X[i])
        self.__Y_tilde = Y_tilde

    def __get_curXY(self):
        t = self.iter
        size = self.s_size
        X = self.X
        Y = self.Y
        if (size*t) >= X.shape[0]:
            return (None, None)
        if (size*(t+1) > X.shape[0]):
            return (X[size*t:], Y[size*t:])
        else:
            return (X[size*t:size*(t+1)], Y[size*t:size*(t+1)])
    
    def __init_beta(self):
        X, Y = self.__get_curXY()
        # ret = l1_optimize(X, Y.T)
        ret = l2_optimize(X, Y.T)
        beta = ret.x
        self.beta = beta
        self.iter += 1
        return beta
    
    def __call_stop(self, new_beta):
        a = new_beta
        b = self.beta
        cosangle = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))  
        return np.arccos(cosangle) < 1e-3
        # return sum(np.square(a - b)) < 

    def fit(self):
        if self.__is_fitted:
            return
        self.__init_beta()
        start_time = time.time()
        while True:
            X, Y = self.__get_curXY()
            if (X is None): break
            # if self.iter >= 20:
            #     break

            self.__get_f0()
            self.__get_Y_tilde()
            new_beta = l2_optimize(X, self.__Y_tilde.T).x
            if self.real_beta is not None:
                self.error_list.append(sum(np.square(new_beta - self.real_beta)))
            if self.__call_stop(new_beta):
                break
            self.beta = new_beta
            self.iter += 1
        end_time = time.time()
        self.__is_fitted = True
        self.time = end_time - start_time

