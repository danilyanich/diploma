from numpy.linalg import solve as lg_solve
from scipy.optimize import nnls, lsq_linear
from scipy.sparse.linalg import lsqr

import numpy as np


def __non_negative(matrix):
  return matrix.clip(0, None)


def __get_col(sparse_matrix, index):
  return np.asarray(sparse_matrix.getcol(index).todense()).flatten()


def __solve_by_columns(A, B, solve):
  _, m = B.shape
  X = [solve(A, __get_col(B, i)) for i in range(m)]
  return np.array(X).T


def __nnls(A, b):
  x, _ = nnls(A, b)
  return x


def __lsqr(A, b):
  info = lsqr(A, b, show=False)
  return info[0]


def alternating_least_squares(A, W, H, solve_steps):
  '''Abstract ALS solver'''

  W, H = W.copy(), H.copy()

  while True:
    yield solve_steps(A, W, H)


def __solve_norm(A, W, H):
  '''Solve with normal equations'''

  H = lg_solve(W.T @ W, W.T @ A)
  H = __non_negative(H)

  W = lg_solve(H @ H.T, H @ A.T).T
  W = __non_negative(W)

  return W, H


def __solve_nnls(A, W, H):
  '''Solve with nonnegative least squares'''

  H = __solve_by_columns(W, A, __nnls)
  W = __solve_by_columns(H.T, A.T, __nnls).T

  return W, H


def __solve_lsqr(A, W, H):
  '''Solve with least squares'''

  H = __solve_by_columns(W, A, __lsqr)
  H = __non_negative(H)

  W = __solve_by_columns(H.T, A.T, __lsqr).T
  W = __non_negative(W)

  return W, H


types = {
  'norm': __solve_norm,
  'nnls': __solve_nnls,
  'lsqr': __solve_lsqr,
}
