'''
    Implementation of the pooled REL(Robust Estimator with LASSO)
'''

def pooledReAl(X, Y):
    '''
        Pooled Re estimation algorithm without LASSO.
        Args: 
            X: pxn ndarray of observations (X1, X2, ...., Xp).
            Y: nx1 ndarray of observations Y.
        Returns:
            beta: the estimation of coefficient.
    '''

    # 1. init: beta_0 = arg min 1/n sum(Yi - X.T \beta)
    S = sample_data(X, Y)
    # beta = l1_optimize(S).X

    while True:
        pass

    # return beta

