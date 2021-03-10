import numpy as np
from time import time
import gen_low_rank_matrix as gm
import matplotlib.pyplot as plt
from sklearn import preprocessing

def centralize_data(ndarray):
    return preprocessing.scale(ndarray, axis=1, with_std=False) 

class PCA_L1:
    def __init__(self, X, m):
        self.X = X
        self.m = m
        self.W = []
        self.iter = 0
        self.__iter = 0
        self.__fitted = False
        self.__curX = None
        self.__inits = None
    
    @staticmethod
    def l2_norm(vector):
        return np.linalg.norm(vector)

    def __get_curX(self):
        if (self.__iter == 0):
            self.__curX = self.X
        else:
            X = self.__curX
            curX = []
            last_w = self.W[-1]
            for x_i in X:
                curX.append(x_i - last_w.dot(last_w.dot(x_i)))
            self.__curX = np.asarray(curX)
    
    def gen_w0(self):
        # ret = np.ones(self.X.shape[1])
        # return ret / self.l2_norm(ret)
        return self.__inits[self.__iter]
    
    def get_w(self):
        self.__get_curX()
        curX = self.__curX
        w = self.gen_w0()
        while True:
            self.iter += 1
            p = np.ones(curX.shape[0])
            for i in range(len(p)):
                p[i] = -1 if (w.dot(curX[i]) < 0) else 1
            w_new = sum([p[i]*curX[i] for i in range(curX.shape[0])])
            w_new = w_new / self.l2_norm(w_new)

            if (not (w_new == w).all()):
                w = w_new 
                for i in range(len(p)):
                    if (w.dot(curX[i]) == 0):
                        delta_w = np.random.random(len(w)) / 20
                        break
                continue
            else:
                break
        self.W.append(w)
        self.__iter += 1
        return w

    def fit(self):
        X = self.X
        m = self.m
        sigma = X.dot(X.T) / X.shape[0]
        e_values, e_vectors = np.linalg.eig(sigma)
        A = e_vectors.T[:m]
        self.__inits = A
        if (self.__fitted): return 
        start_time = time()

        for i in range(self.m):
            self.get_w()

        self.W = np.asarray(self.W)
        end_time = time()
        self.time = end_time - start_time
        self.__fitted = True


if __name__ == '__main__':
    m, n = 3,80
    X = gm.gen_low_rank_matrix(m, n)
    X = gm.disturb_matrix(X, 10, 0)
    X = centralize_data(X)
    pca_l1 = PCA_L1(X, m)
    pca_l1.fit()
    W = pca_l1.W
    errors = (X - W.T.dot(W.dot(X))).flatten()
    errors = np.square(errors)
    plt.hist(errors, density=False, histtype='stepfilled', bins=
    np.linspace(0, 100, 50), color='dimgrey', label='l1')

    plt.legend()
    plt.show()
    sigma = X.dot(X.T) / X.shape[0]
    e_values, e_vectors = np.linalg.eig(sigma)
    A = e_vectors.T[:m]
    l2_errors = (X - A.T.dot(A.dot(X))).flatten()
    l2_errors = np.square(l2_errors)

    plt.hist(l2_errors, density=False, histtype='stepfilled', bins=
    np.linspace(0, 100, 50), color='dimgrey', label='l2')
    plt.legend()
    plt.show()