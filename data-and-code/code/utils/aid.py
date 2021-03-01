'''
    Implementation of the aggregate and disaggregate algorithm
'''
import numpy as np
import time
from optimize import l1_optimize, l2_optimize
from sv import SV
import  matplotlib.pyplot as plt

def gen_data(p, n, s: int, dist='normal'):
    '''
        generate simulation data.
    '''
    if not p:
        p = 100
    if not n:
        n = 10000
    if not s:
        s = p
    beta = np.zeros(p)
    for i in range(s):
        beta[i] = 10*i/s
    
    sigma = np.ones((p, p))
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            sigma[i][j] = abs(i - j) * 0.5
    
    # X = np.random.multivariate_normal(
    #     np.zeros(p),
    #     np.ones((p, p)),
    #     n
    # )

    X = np.random.random((n, p))
    X.T[0] = np.ones(n)

    Y = np.asarray([x.dot(beta) + np.random.standard_cauchy()*0.01 for x in X])
    
    return (X, Y.T, beta)

def l2_error(beta):
    return sum(np.square(beta))

if __name__ == '__main__':
    n = 200_000 
    X, Y , beta= gen_data(30, 20000, 10)
    print(beta)
    print('-----------')

    sv = SV(X, Y, 10, 500, beta)
    sv.fit()

    plt.plot(sv.error_list[:50], label='l2-errors of SVN', color='gray')
    print(sv.time)
    start_time = time.time()
    res = l1_optimize(X, Y)
    end_time = time.time()
    print(end_time -start_time)
    a = sum(np.square(res.x - beta))
    plt.axhline(y=a, xmin=0, xmax=50, label='l2-error of LP', color='gray', marker='.')
    plt.legend()
    plt.title('Cauchy noise')
    plt.show()