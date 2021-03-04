'''
    To solve the problem: X(pxn) = A(pxm) F(mxn), we adapt 3 different solutions.
    Algorithm Input: <-- X(pxn) is a numpy ndarray.
    Algorithm Output: ---> (A[pxm], F[mxn]) is a tuple of numpy ndarray.

    PCA solution: Calculate XX.T then compute eigen vectors and eigen values ...

    SVD solution: Use numpy.linalg.svd / sklearn.decomposition.pca ... 

    L2norm solution: An alternate convex programming using L2norm ...

    L1norm solution: An alternate convex programming using L1norm ...
'''
import numpy as np
import matplotlib.pyplot as plt
from low_rank_matrix import gen_low_rank_matrix, disturb_matrix
import visualize
from sklearn import preprocessing
import time
import optimize
from sv import SV


def is_close(a, b):
    '''
        Determine whether the two vectors are too close.
    '''
    cosangle = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))  
    return np.arccos(cosangle) < 0.05


def if_stop_A(A_new, A_old):
    '''
        Determine whether the iteration should be terminated.
    '''
    for i in range(A_new.shape[0]):
        a_new = A_new[i]
        a_old = A_old[i]
        if not is_close(a_new, a_old):
            return False
    return True


def if_stop_F(F_new, F_old):
    '''
        Determine whether the iteration should be terminated.
    '''
    for j in range(F_new.T.shape[0]):
        f_new = F_new.T[j]
        f_old = F_old.T[j]
        if not is_close(f_new, f_old):
            return False
    return True


def centralize_data(ndarray):
    return preprocessing.scale(ndarray, axis=1, with_std=False) 


def svd_solution(X, m: int):
    '''
        Arguments:
            X: a pxn numpy ndarray.
            m: arg used to define the lower dimension of output.
        Returns:
            (A, F): A, the factor loading which is a pxm numpy ndarray;
                    F, the factor scores which is a mxn numpy ndarray.
    '''
    U, S, Vh = np.linalg.svd(X)
    S = np.diag(S)
    A = U.dot(S[:,0:3])
    F = Vh[:3, :]
    return (A, F)


def l1_solution(X, m: int):
    '''
        Using alternate convex programming:
        1. initialization, A(0) = A(pca_solution), Sigma = diag(ones(1))
        iteration = 0

        2. while(True):
                F_new = optimize_F(X, A)
                
                A_new = optimize_A(X, F_new)

                if(if_stop(F_new, F) and if_stop(A_new, A)): break

                Na = diag(A_new.T A_new)
                Nf = diag(F_new.T F_new)
                F = F_new(Nf)**-1
                A = A_new(Na)**-1
                Sigma = Na Sigma Nf

        3. F = F Sigma**0.5, A = A Sigma**0.5

        return (A, F)

        Arguments:
            X: a pxn numpy ndarray.
            m: arg used to define the lower dimension of output.
        Returns:
            (A, F): A, the factor loading which is a pxm numpy ndarray;
                    F, the factor scores which is a mxn numpy ndarray.
    '''
    A, _ = pca_solution(X, m) # A pxm
    Sigma = np.diag(np.ones(m)) # Sigma mxm
    iteration = 0
    print(A.shape, X.shape)
    print('------begins!--------')

    while (True):
        F_new = optimize.l1_optimize_F(X, A)
        A_new = optimize.l1_optimize_A(X, F_new)
        print(F_new.shape, A_new.shape)
        print('------------')

        if (iteration > 0 and if_stop_F(F_new, F) and if_stop_A(A_new, A)):
            break
        
        if (iteration > 20):
            break

        F = F_new
        A = A_new

        # Na = np.diag(np.diag(A_new.T.dot(A_new)))
        # Nf = np.diag(np.diag(F_new.dot(F_new.T)))
        # F = (np.linalg.inv(Nf)).dot(F_new)
        # A = A_new.dot(np.linalg.inv(Na))
        # print(Nf.shape)
        # Sigma = Na.dot(Sigma).dot(Nf)
        print(f'---------iteration{iteration}------------')
        iteration += 1

    # A = A.dot(np.sqrt(Sigma))
    # F = np.sqrt(Sigma).dot(F)
    return (A, F)


def sv_optimize_F(X, A, m):
    F_columns = []
    for x_j in X.T:
        f_j = optimize.l1_optimize(A, x_j).x # f_j mx1
        F_columns.append(f_j)
    F = np.asarray(F_columns).T
    return F


def sv_optimize_A(X, F, m):
    A_columns = []
    for x_i in X:
        sv = SV(F.T, x_i, m, int(X.shape[0]/10), None)
        sv.fit()
        # a_i = l1_optimize(F.T, x_i).x
        a_i = sv.beta
        A_columns.append(a_i)
    A = np.asarray(A_columns)
    return A


def l2_solution(X, m: int):
    '''
        Arguments:
            X: a pxn numpy ndarray.
            m: arg used to define the lower dimension of output.
        Returns:
            (A, F): A, the factor loading which is a pxm numpy ndarray;
                    F, the factor scores which is a mxn numpy ndarray.
    '''
    A, _ = pca_solution(X, m) # A pxm
    Sigma = np.diag(np.ones(m)) # Sigma mxm
    iteration = 0
    print(A.shape, X.shape)
    print('------begins!--------')

    while (True):
        F_new = optimize.l2_optimize_F(X, A)
        A_new = optimize.l2_optimize_A(X, F_new)
        print(F_new.shape, A_new.shape)
        print('------------')

        if (iteration > 0 and if_stop_F(F_new, F) and if_stop_A(A_new, A)):
            break
        
        if (iteration > 20):
            break

        F = F_new
        A = A_new

        # Na = np.diag(np.diag(A_new.T.dot(A_new)))
        # Nf = np.diag(np.diag(F_new.dot(F_new.T)))
        # F = (np.linalg.inv(Nf)).dot(F_new)
        # A = A_new.dot(np.linalg.inv(Na))
        # print(Nf.shape)
        # Sigma = Na.dot(Sigma).dot(Nf)
        print(f'---------iteration{iteration}------------')
        iteration += 1

    # A = A.dot(np.sqrt(Sigma))
    # F = np.sqrt(Sigma).dot(F)
    return (A, F)


def pca_solution(X, m: int):
    '''
        Arguments:
            X: a pxn numpy ndarray.
            m: arg used to define the lower dimension of output.
        Returns:
            (A, F): A, the factor loading which is a pxm numpy ndarray;
                    F, the factor scores which is a mxn numpy ndarray.
    '''
    sigma = X.dot(X.T) / X.shape[0]
    e_values, e_vectors = np.linalg.eig(sigma)
    A = e_vectors.T[:m].T
    F = A.T.dot(X) # / X.shape[0]
    return (A, F)


def test_method(X, m, method):
    solution_map = {
        'SVD': pca_solution,
        'ICP-L2': svd_solution,
        'ICP-L1': l1_solution,
        'IRLS': l2_solution,
        # 'SV-ICP-L1': sv_l1_solution,
    }
    solution = solution_map.get(method)
    time1= time.time()
    A, F = solution(X, m)
    error = (X - A.dot(F)).flatten()
    error_array = np.abs(error)
    print(time.time() - time1)
    error_array = np.square(error_array)
    print(np.median(error_array))
    plt.hist(error_array, density=False, histtype='stepfilled', bins=
    np.linspace(0, 100, 50), color='dimgrey')
    plt.show()
    # visualize.display_error_matrix(X - A.dot(F), name=method )# , 'absolute')


def test():
    X = gen_low_rank_matrix(3, 100)
    X = disturb_matrix(X, 10, 0)
    X = centralize_data(X)
    # print(l1_solution(X, 3))
    # test_method(X, 3, 'SVD')
    # test_method(X, 3, 'IRP-L2')
    test_method(X, 3, 'ICP-L1')
    # test_method(X, 3, 'SV-ICP-L1')
    # test_method(X, 3, 'IRLS')

if __name__ == '__main__':
    test()