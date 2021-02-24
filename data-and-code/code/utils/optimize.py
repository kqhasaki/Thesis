import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
from os import path

DIR_PATH = path.abspath('../')

def fit_linear(X, params):
    '''
    Params:
        X: a numpy ndarray
        params: a numpy ndarray
    '''
    return X.dot(params)


def l2_cost_function(params, X, y):
    '''
    Params:
        X: a n row p column matrix.
        y: a n row array.
        params: a p row array.
    '''
    return np.sum(np.square(y - fit_linear(X, params)))


def l1_cost_function(params, X, y):
    '''
    Params:
        X: a n row p column matrix.
        y: a n row array.
        params: a p row array.
    '''
    return np.sum(np.abs(y - fit_linear(X, params))) 


def l1_optimize(X, y, x0=None):
    '''
    Params:
        X: a n row p column matrix.
        y: a n row array.
        x0: a initial guess of params.
    '''
    if (x0 == None):
        p = X.shape[1]
        x0 = np.ones(p)
    res = minimize(l1_cost_function, x0, args=(X, y))
    return res


def l2_optimize(X, y, x0=None):
    '''
    Params:
        X: a n row p column matrix.
        y: a n row array.
        x0: a initial guess of params.
    '''
    if (x0 == None):
        p = X.shape[1]
        x0 = np.ones(p)
    res = minimize(l2_cost_function, x0, args=(X, y))
    return res


def show_l1_l2_diff():
    x = np.random.randint(0, 20, 20)
    y = np.asarray([0.7 * var + np.random.random() for var in x])
    X = np.asarray([np.ones(x.shape), x]).T
    # y[3] = 30
    # y[12] = 60
    # y[8] = 0
    res_l1 = l1_optimize(X, y)
    res_l2 = l2_optimize(X, y)
    l1_y_hat = fit_linear(X, res_l1.x)
    l2_y_hat = fit_linear(X, res_l2.x)
    fig = plt.figure()
    ax = plt.axes()

    x_l = np.linspace(0, 20, 100)
    X_l = np.asarray([np.ones(x_l.shape), x_l]).T
    y_l_l1 = fit_linear(X_l, res_l1.x)
    y_l_l2 = fit_linear(X_l, res_l2.x)

    plt.scatter(x, y, marker='x', color='black')
    plt.plot(x_l, y_l_l1, color='blue', label='L1 norm')
    plt.plot(x_l, y_l_l2, color='red', label='L2 norm')
    plt.legend()
    # plt.show()

    plt.savefig(f'{DIR_PATH}/pics/l1-l2-diff-{time.ctime()}.pdf' )


if __name__ == '__main__':
    show_l1_l2_diff()