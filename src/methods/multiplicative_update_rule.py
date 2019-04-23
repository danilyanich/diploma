import numpy as np


EPS = 10e-9


def multiplicative_update_rule(A, W, H):
  W = W.copy()
  H = H.copy()

  while True:
    # H = H * (W' A) / ((W' W H) + 10e-9)
    expr1 = W.T @ A
    expr2 = (W.T @ W) @ H + EPS
    H = np.divide(np.multiply(H, expr1), expr2)

    # W = W * (A H') / ((W H H') + 10e-9)
    expr3 = A @ H.T
    expr4 = W @ (H @ H.T) + EPS
    W = np.divide(np.multiply(W, expr3), expr4)

    yield W, H
