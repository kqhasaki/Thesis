from matrix_factorize import l1_solution
from simulate_factor import gen_X
from visualize import display_error_matrix
import numpy as np
import matplotlib.pyplot as plt
# 解决中文显示问题

from scipy import stats
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
X, a = gen_X(120, 400, 8)


a = [stats.kurtosis(x) for x in X]
plt.hist(a)
plt.show()

# error = (X - A.dot(F)).flatten()
# error = np.abs(error)
# display_error_matrix(X - A.dot(F))# , 'absolute'
