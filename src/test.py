import os
import glob as gl

import methods as mt
import utils as ut
import parser as ps


PLOT_FILE = open('./report/test_plot.tex', mode='w')

RANK_K = 25
EPS = 10e-3


def __test_on_file(test_file):
  '''Run all methods on provided file'''

  print('### Testing ###')
  print()
  print('Parsing file {}'.format(os.path.basename(test_file.name)))
  print('Rank {}'.format(RANK_K))
  print()

  sentences = ut.get_sentences(test_file.read())
  A, terms = ps.get_weighted_term_document_matrix(sentences)

  print('Matrix a size {}'.format(A.shape))
  print()
  print()

  W, H = ut.initialize_random(*A.shape, RANK_K)

  methods = {
    'Alternating least squares (solve)': mt.alternating_least_squares_solve(A, W, H),
    'Alternating least squares (nnls)': mt.alternating_least_squares_nnls(A, W, H),
    'Alternating least squares (lstsq)': mt.alternating_least_squares_lstsq(A, W, H),
    'Multiplicative update rule': mt.multiplicative_update_rule(A, W, H)
  }

  print('# Iterations')
  print()

  error = 10e10
  best_result = None
  results = {}

  for name, generator in methods.items():
    try:
      result, errors = ut.iterate(
        generator,
        matrix=A,
        precision=EPS,
        method_name=name
      )

      results[name] = result, errors

      if errors[-1] < error:
        error = errors[-1]
        best_result = name

    except Exception:
      print('method_failed')

  print('Best - {}'.format(best_result))
  print()
  print()

  return results


if __name__ == '__main__':
  files = gl.glob('./samples/*.txt')

  test_results = [__test_on_file(open(f, 'r')) for f in files]
