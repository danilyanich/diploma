from numpy.linalg import solve
from scipy.optimize import nnls
from scipy.linalg import lstsq

import numpy as np


def __non_negative(matrix):
  return matrix.clip(0, None)


def alternating_least_squares_solve(A, W, H):
  W = W.copy()

  while True:
    # W' W H = W' A
    H = solve(W.T @ W, W.T @ A)
    H = __non_negative(H)

    # H H' W' = H A'
    WT = solve(H @ H.T, H @ A.T)
    W = __non_negative(WT.T)

    yield W, H


# min_X ||AX - B||_2, X >= 0
def __nnls(A, B):
  def __nnls__map(b):
    b = np.asarray(b).flatten()

    x, _ = nnls(A, b)
    return x

  X = [__nnls__map(B[:, i]) for i in range(B.shape[1])]
  return np.array(X).T


def alternating_least_squares_nnls(A, W, H):
  W = W.copy()

  while True:
    H = __nnls(W.T @ W, W.T @ A)
    H = __non_negative(H)

    WT = __nnls(H @ H.T, H @ A.T)
    W = __non_negative(WT.T)

    yield W, H


def __lstsq(A, B):
  X, _, _, _ = lstsq(A, B)
  return X


def alternating_least_squares_lstsq(A, W, H):
  W = W.copy()

  while True:
    # W H = A
    H = __lstsq(W.T @ W, W.T @ A)
    H = __non_negative(H)

    # H' W' = A'
    WT = __lstsq(H @ H.T, H @ A.T)
    W = __non_negative(WT.T)

    yield W, H
