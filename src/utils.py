import numpy as np
import scipy.sparse.linalg as slg
import itertools as it
import time as tm
import re
import os


def initialize_random(m, n, RANK_K):
  '''Create random matrices'''

  W = np.random.rand(m, RANK_K)
  H = np.random.rand(RANK_K, n)

  return W, H


def __trace_of_product(A, B):
  '''Compute efficient trace of matrix multiplication'''

  n, _ = A.shape

  diag = (A[i,:] @ B[:,i] for i in range(n))

  return sum(diag)


def __relative_error(A, W, H, trace_AT_A):
  '''Calculate relative error over sparse A matrix'''

  norm = trace_AT_A - 2 * __trace_of_product(H.T, W.T @ A) + __trace_of_product(H.T, W.T @ W @ H)
  relative_norm = norm / slg.norm(A)

  return relative_norm


def iterate(generator, A, W, H, **kwargs):
  '''Iterate over method generator'''

  EPS = kwargs.get('precision')
  label = kwargs.get('method_name')

  trace_AT_A = (A.T @ A).diagonal().sum()

  errors = [__relative_error(A, W, H, trace_AT_A)]
  result = None

  progress_gen = generator(A, W, H)

  while len(errors) < 2 or abs(errors[-2] - errors[-1]) >= EPS:
    result = next(progress_gen)

    error = __relative_error(A, *result, trace_AT_A)

    errors.append(error)
    pass

  return result, errors


def replace_ext(filename, ext):
  return os.path.splitext(filename)[0]+'.'+ext


def get_generator_progress(generator, label):
  all_start = tm.process_time()
  print('> {}'.format(label))
  length = 0

  for idx, item in enumerate(generator):
    yield item
    length += 1
    took = tm.process_time() - all_start
    print('\r[progress] {} steps, took {:.2f}s'.format(idx + 1, took), end='')
