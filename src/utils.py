import numpy as np
import scipy.sparse.linalg as slg
import itertools as it
import time as tm
import re


def initialize_random(m, n, RANK_K):
  '''Create random matrices'''

  W = np.random.rand(m, RANK_K)
  H = np.random.rand(RANK_K, n)

  return W, H


def __relative_error(A, W, H):
  '''Calculate relative error over sparse A matrix'''

  cx = A.tocoo()

  zipped = zip(cx.row, cx.col, cx.data)
  norm = sum(abs(v - W[i, :] @ H[:, j]) for i, j, v in zipped)

  return norm / slg.norm(A)


def iterate(generator, **kwargs):
  '''Iterate over method generator'''

  A = kwargs.get('matrix')
  EPS = kwargs.get('precision')
  label = kwargs.get('method_name')

  errors = []
  result = None

  progress_gen = get_generator_progress(generator, label)

  while len(errors) < 2 or abs(errors[-2] - errors[-1]) >= EPS:
    result = next(progress_gen)

    error = __relative_error(A, *result)
    errors.append(error)
    pass

  print()
  print('error {:.5f}'.format(errors[-1]))
  print()

  return result, errors


def get_sentences(raw_text):
  '''Split text into sentences'''

  # Replace spaces, quotes and newlines
  text = re.sub('[\n" ]+', ' ', raw_text)

  # Split text by punctuation and remove leading whitespaces
  raw_sentences = [
    re.sub('^ +', '', s)
    for s in re.split('[!.?]+ +', text)
  ]

  # Filter empty sentences
  sentences = [s for s in raw_sentences if len(s)]

  return sentences


def get_list_progress(iterable, label):
  all_start = tm.process_time()
  print('> {}'.format(label))
  length = len(iterable)

  for idx, item in enumerate(iterable):
    yield item
    took = tm.process_time() - all_start
    print('\r[progress] took {:.2f}s, {} of {} done'.format(took, idx + 1, length), end='')


def get_generator_progress(generator, label):
  all_start = tm.process_time()
  print('> {}'.format(label))
  length = 0

  for idx, item in enumerate(generator):
    yield item
    length += 1
    took = tm.process_time() - all_start
    print('\r[progress] {} steps, took {:.2f}s'.format(idx + 1, took), end='')
