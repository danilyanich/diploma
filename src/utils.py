import numpy as np
import scipy.sparse.linalg as slg
import multiprocessing as mp
import itertools as it
import time as tm


def initialize_random(m, n, RANK_K):
  W = np.random.rand(m, RANK_K)
  H = np.random.rand(RANK_K, n)

  return W, H


def __pool_map(zipped):
  i, j, v, W, H = zipped
  return abs(v - W[i, :] @ H[:, j])


def __relative_error(A, W, H):
  cx = A.tocoo()

  pool_count = mp.cpu_count() - 1
  with mp.Pool(pool_count) as pool:
    zipped = zip(cx.row, cx.col, cx.data, it.repeat(W), it.repeat(H))
    norm = sum(pool.map(__pool_map, zipped))

    return norm / slg.norm(A)
  pass


def iterate(generator, A, EPS):
  errors = []
  result = None

  all_start = tm.process_time()
  print('[iterations] start')

  while len(errors) < 2 or abs(errors[-2] - errors[-1]) >= EPS:
    result = next(generator)

    error = __relative_error(A, *result)
    errors.append(error)

    print('\r{}: error={:.5f}'.format(len(errors), error), end='')
    pass

  print('\r[iterations] took {:.2f}s, {} iterations, error={:.5f}'.format(tm.process_time() - all_start, len(errors), errors[-1]))
  return result, errors
