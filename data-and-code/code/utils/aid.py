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

    Y = np.asarray([x.dot(beta) + np.random.random()*0.1 for x in X])
    
    return (X, Y.T, beta)

def l2_error(beta):
    return sum(np.square(beta))

if __name__ == '__main__':
    n = 200_000 
    X, Y , beta= gen_data(500, 20000, 100)
    print('-----------')
    for i in range(2000):
        a = np.random.choice(range(20000))
        Y[a] = Y[a] + np.random.standard_cauchy()
    
    s = 2
# for s in sv_list:
    sv = SV(X, Y, s, 500, beta)
    sv.fit()
    plt.plot(sv.error_list[:50], label='convergence of SVN', color='gray')
    start_time = time.time()
    res = l1_optimize(X, Y)
    end_time = time.time()
    res2 = l2_optimize(X, Y)
    # print(end_time -start_time)
    a = sum(np.square(res.x - beta))
    b = sum(np.square(sv.beta - beta))
    c = sum(np.square(res2.x - beta))
    print(f'-----s ==> {s}------')
    print('iter', sv.iter)
    print('l1_time',end_time - start_time)
    print('sv_time', sv.time)
    print('l1:', a)
    print('l2:', c)
    print('SV:', b)
    # plt.axhline(y=a, xmin=0, xmax=50, label='l2-error of LP', color='gray', marker='.')
        # plt.legend()
        # plt.title('Cauchy noise / beta initialized by LP')
        # plt.title(f's = {s}')
        # plt.show()