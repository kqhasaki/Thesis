"""
Functions and classes used for labs and empirical-research
"""
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def gen_low_rank_matrix(rank: int = 3, dimension: int = 40):
    '''
     generate a random low rank matrix which you can ajust the rank, dimension and scale.

     Args:
        rank: the rank of the matrix, must be an integer.
        dimension: the dimension of the matrix, must be an integer.

     Returns:
        return a numpy ndarray.
    '''
    if (type(rank) != int or type(dimension) != int or rank >= dimension):
        raise Exception(
            f'{gen_low_rank_matrix.__name__} has got incorrect arguments.')

    a = np.random.randint(-100, 100, (dimension, dimension))
    # a = np.random.random((dimension, dimension)) * 100
    m, sigma, vh = np.linalg.svd(a)
    return m[:, 0:rank].dot(np.diag(sigma)[0:rank, 0:rank]).dot(vh[0:rank, :])


def disturb_matrix(input_matrix, outliers: int = 10, missing_data: int = 10):
    '''
    create some outliers and missing data to disturb the given matrix.

    Args:
        input_matrix: must be a numpy ndarray.
        outliers: ratio of outliers, must be an integer.
        missing_data: ratio of missing data, must be an integer.

    Returns:
        return a numpy ndarray.
    '''
    if (type(outliers) != int or type(missing_data) != int):
        raise Exception(
            f'{disturb_matrix.__name__} has got incorrect arguments.')
    if (outliers >= 100 or outliers <= 0 or missing_data >= 100 or missing_data < 0):
        raise Exception(
            f'{disturb_matrix.__name__}: outliers and missing_data must be less than 100 and larger than 0.')

    matrix = deepcopy(input_matrix)
    m, n = matrix.shape
    # gen missing data
    if missing_data != 0:
        missing = int(missing_data / 100 * min(m, n))
        for i in range(0, m):
            for j in range(0, n):
                x = m - 1 - i
                y = j
                if (x + y <= missing):
                    matrix[i][j] = np.nan
    # gen outliers
    num = int(outliers * matrix.size / 100)
    coordinate_set = set()
    while (num > 0):
        new_cor = (np.random.randint(m), np.random.randint(n))
        if (new_cor in coordinate_set):
            continue
        coordinate_set.add(new_cor)
        if (np.isnan(matrix[new_cor])):
            continue
        matrix[new_cor] = np.random.randint(-2000, 2000)
        num -= 1
    return matrix


def test():
    # test random matrix generation and visualization
    from visualize import display_nd_array
    display_nd_array(disturb_matrix(gen_low_rank_matrix(4, 30)))


if __name__ == '__main__':
    test()
