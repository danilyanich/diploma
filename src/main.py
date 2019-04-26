import methods as mt
import utils as ut
import matplotlib.pyplot as plt
import numpy as np
import parser as ps
import re
import sys


RANK_K = 1
EPS = 10e-5

np.set_printoptions(
  precision=3,
  suppress=True,
  formatter={'float': '{: 0.3f}'.format}
)


def __do_methods(methods):
  best_result = None
  error = 10e10

  for name, generator in methods.items():
    print('# {}'.format(name))
    result, errors = ut.iterate(generator, A, EPS)

    plt.plot(errors, label=name)
    print()

    if error > errors[-1]:
      best_result = result
      error = errors[-1]

  plt.legend()
  plt.show()

  return best_result


if __name__ == '__main__':
  text = re.sub('[\n" ]+', ' ', sys.stdin.read())
  sentences = [re.sub('^ +', '', s) for s in re.split('[!.?]+', text)]

  A = ps.get_weighted_term_document_matrix(sentences)
  W, H = ut.initialize_random(*A.shape, RANK_K)

  W, H = __do_methods({
    'alternating_least_squares_solve': mt.alternating_least_squares_solve(A, W, H),
    'alternating_least_squares_nnls': mt.alternating_least_squares_nnls(A, W, H),
    'alternating_least_squares_lstsq': mt.alternating_least_squares_lstsq(A, W, H),
    'multiplicative_update_rule': mt.multiplicative_update_rule(A, W, H)
  })

  data = sorted(zip(W.T[0], sentences), key=lambda t: t[0], reverse=True)[:5]

  for d in data:
    score, sentence = d
    print('{:.3f} {}'.format(score, sentence))
  pass
