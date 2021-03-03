import numpy as np

def gen_random_factors(m,n):
    a = np.random.random((m, n))
    return a

def gen_X(p, t, m):
    a = gen_random_factors(m, t).T

    X_means = np.asarray([np.random.random()*0 for i in range(p)])
    X_loadings = np.asarray([
       np.asarray([
           np.random.random() for i in range(m)
       ])  for i in range(p)])
    

    X = []
    for i in range(p):
        mean = X_means[i]
        X_i = np.asarray([
            X_means[i] + X_loadings[i].dot(a[i]) + np.random.random() for var in range(t)
        ])
        for k in range(int(0.1 * t)):
            X_i[k] = X_i[k] + np.random.random() * 4

        X.append(X_i)
    
    return np.asarray(X), a

# print(gen_X(120, 240, 8))