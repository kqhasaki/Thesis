'''
    To solve the problem: X(pxn) = A(pxm) F(mxn), we adapt 3 different solutions.
    Algorithm Input: <-- X(pxn) is a numpy ndarray.
    Algorithm Output: ---> (A[pxm], F[mxn]) is a tuple of numpy ndarray.

    PCA solution: Calculate XX.T then compute eigen vectors and eigen values ...

    SVD solution: Use numpy.linalg.svd ... 

    L2norm solution: An alternate convex programming using L2norm ...

    L1norm solution: An alternate convex programming using L1norm ...
'''
import numpy as np


def pca_solution(X, m: int):
    '''
        Arguments:
            X: a pxn numpy ndarray.
            m: arg used to define the lower dimension of output.
        Returns:
            (A, F): A, the factor loading which is a pxm numpy ndarray;
                    F, the factor scores which is a mxn numpy ndarray.
    '''