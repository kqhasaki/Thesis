import sys
sys.path.append('..')

import utils.low_rank_matrix as lab
import utils.visualize as visualize

def task1():
    # 产生一个随机矩阵并且画图
    dimension = 30
    r = 4
    M = lab.gen_low_rank_matrix(r, dimension)
    # print(matrix)
    matrix = lab.disturb_matrix(M, 15, 30)
    visualize.display_nd_array(matrix)


def main():
    task1()

main()