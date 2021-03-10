import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matrix_factorize import l1_solution
from matrix_factorize import pca_solution
from pca_l1 import PCA_L1
import gen_low_rank_matrix as gm
from statsmodels.multivariate.factor_rotation import rotate_factors


sns.set_theme()
np.random.seed(130)

X = gm.gen_low_rank_matrix(4, 15)
X = gm.disturb_matrix(X, 3, 0)

# A1, _ = l1_solution(X, 4)
# A1 = np.real(A1)
# A1 = np.abs(A1)
# A1, _ = rotate_factors(A1, 'varimax')
# ax1 = sns.heatmap(A1,
# xticklabels=False, yticklabels=False, cmap='Greys', cbar=False)
# ax1.set_title('L1-PCA')
# ax1.plot()
# plt.show()

# A2, _ = pca_solution(X, 4)
# A2 = np.real(A2)
# A2 = np.abs(A2)
# A2, _ = rotate_factors(A2, 'varimax')
# ax2 = sns.heatmap(A2,
# xticklabels=False, yticklabels=False, cmap='Greys', cbar=False)
# ax2.set_title('L2-PCA')
# ax2.plot()
# plt.show()

pca = PCA_L1(X, 4)
pca.fit()
A3 = np.abs(pca.W)
A3 = np.real(A3)
A3, _ = rotate_factors(A3, 'varimax')
ax3 = sns.heatmap(A3.T,
xticklabels=False, yticklabels=False, cmap='Greys', cbar=False)
ax3.set_title('PCA-L1')
ax3.plot()
plt.show()
