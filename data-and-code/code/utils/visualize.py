import matplotlib.pyplot as plt
import time
import numpy as np
from os import path
import seaborn as sns

DIR_PATH = path.abspath('../')


def display_nd_array(ndarray, name: str = ''):
    '''
    Display an ndarray using matplotlib while the display should show the scale difference among the elements.

    Args:
        ndarray: numpy ndarray.
        name: give this figure a name

    Returns:
        save matplotlib figure. 
    '''
    plt.matshow(ndarray, cmap=plt.get_cmap('Greys'))
    plt.savefig(
        f'{DIR_PATH}/pics/matrixfig-{time.ctime()}.pdf' if not name else f'{DIR_PATH}/pics/{name}.pdf',
        dpi=600)


def display_error_matrix(ndarray, method='squared', name: str = ''):
    '''
    Display a error matrix.
    Args:
        ndarray: the error matrix.
        method: default 'squared', can be set to 'absolute'
    '''
    error_array = ndarray.flatten()
    if method == 'absolute':
        error_array = np.abs(error_array)
    error_array = np.square(error_array)
    plt.hist(error_array, density=False, histtype='stepfilled', bins=
    np.linspace(0, 100, 50))
    plt.title(name)
    plt.savefig(
         f'{DIR_PATH}/pics/error-matrix-{time.ctime()}-{name}.pdf',
        dpi=600 
    )
    plt.clf()