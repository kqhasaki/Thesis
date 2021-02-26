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
import optimize


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
        
        if (iteration > 30):
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
        
        if (iteration > 30):
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
        'PCA': pca_solution,
        'SVD': svd_solution,
        'L1': l1_solution,
        'Re-Weighted-L2': l2_solution
    }
    solution = solution_map.get(method)
    A, F = solution(X, m)
    error = (X - A.dot(F)).flatten()
    error = np.abs(error)
    visualize.display_error_matrix(X - A.dot(F), name=method )# , 'absolute')


def test():
    X = gen_low_rank_matrix(3, 80)
    X = disturb_matrix(X, 3, 0)
    X = centralize_data(X)
    # print(l1_solution(X, 3))
    # test_method(X, 3, 'SVD')
    # test_method(X, 3, 'PCA')
    # test_method(X, 3, 'L1')
    test_method(X, 3, 'Re-Weighted-L2')

if __name__ == '__main__':
    test()