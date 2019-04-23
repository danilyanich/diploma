from sklearn.datasets import fetch_20newsgroups_vectorized as dataset

import methods as mt
import utils as ut
import matplotlib.pyplot as plt
# from scipy.sparse import csr_matrix

RANK_K = 4
EPS = 10e-3


if __name__ == '__main__':
  A = dataset().data# [:5000, :10000]

  # A = csr_matrix([
  #   [5, 1, 3, 5, 2],
  #   [0, 0, 0, 2, 5],
  #   [4, 0, 3, 0, 1],
  #   [1, 5, 5, 3, 3]
  # ])

  W, H = ut.initialize_random(*A.shape, RANK_K)

  methods = {
    'alternating_least_squares_solve': mt.alternating_least_squares_solve(A, W, H),
    'alternating_least_squares_nnls': mt.alternating_least_squares_nnls(A, W, H),
    'alternating_least_squares_lstsq': mt.alternating_least_squares_lstsq(A, W, H),
    'multiplicative_update_rule': mt.multiplicative_update_rule(A, W, H)
  }

  for name, generator in methods.items():
    print('# {}'.format(name))
    _, errors = ut.iterate(generator, A, EPS)
    print()

    plt.plot(errors, label=name)

  plt.legend()
  plt.show()
  pass
