import matplotlib.pyplot as plt
import time
from os import path

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
        f'{DIR_PATH}/pics/matrixfig-{time.ctime()}' if not name else f'{DIR_PATH}/pics/{name}')
