from simulate_factor import gen_X
from matrix_factorize import test_method

m = 10
a, _ = gen_X(50, 200, m)

test_method(a, m, 'SVD')
test_method(a, m, 'ICP-L1')
