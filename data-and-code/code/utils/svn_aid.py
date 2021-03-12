import numpy as np
from optimize import l1_optimize, l2_optimize
import time


class SVN_AID:
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