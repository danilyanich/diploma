import methods as mt
import utils as ut
import matplotlib.pyplot as plt
import numpy as np
import parser as ps
import re
import sys
import argparse as ar


np.set_printoptions(
  precision=3,
  suppress=True,
  formatter={'float': '{: 0.3f}'.format}
)

parser = ar.ArgumentParser(description='Perform nonnegative matrix decomposition.')
parser.add_argument(
  '--rank',
  type=int,
  help='rank of decomposition',
  default=1,
)
parser.add_argument(
  '--in',
  dest='infile',
  type=str,
  help='read document from file',
  default=None,
)
parser.add_argument(
  '--out',
  type=str,
  help='store decomposition to file',
  default=None,
)
parser.add_argument(
  '--precision',
  type=float,
  help='precision of computation',
  default=10e-5,
)
parser.add_argument(
  '--plot',
  action='store_true',
  help='draw a plot of relative error',
)


def __do_methods(methods, plot):
  best_result = None
  error = 10e10

  for name, generator in methods.items():
    try:
      print('# {}'.format(name))
      result, errors = ut.iterate(generator, A, EPS)

      if plot:
        plt.plot(errors, label=name)
      print()

      if error > errors[-1]:
        best_result = result
        error = errors[-1]
    except:
      print('method_failed')

  if plot:
    plt.legend()
    plt.show()

  return best_result


if __name__ == '__main__':
  args = parser.parse_args()

  IN_FILE = open(args.infile, mode='r') if args.infile else sys.stdin
  OUT_FILE = open(args.out, mode='w') if args.out else sys.stderr
  RANK_K = args.rank
  EPS = args.precision
  PLOT = args.plot

  text = re.sub('[\n" ]+', ' ', IN_FILE.read())
  sentences = list(filter(bool, [re.sub('^ +', '', s) for s in re.split('[!.?]+', text)]))

  A, terms = ps.get_weighted_term_document_matrix(sentences)

  W, H = ut.initialize_random(*A.shape, RANK_K)

  print('# rank k = {}'.format(RANK_K))
  print()

  W, H = __do_methods({
    'alternating_least_squares_solve': mt.alternating_least_squares_solve(A, W, H),
    'alternating_least_squares_nnls': mt.alternating_least_squares_nnls(A, W, H),
    'alternating_least_squares_lstsq': mt.alternating_least_squares_lstsq(A, W, H),
    'multiplicative_update_rule': mt.multiplicative_update_rule(A, W, H)
  }, PLOT)

  OUT_FILE.write('# matrix W sorted\n')
  for score, sentence in sorted(zip(W, sentences), key=lambda t: sum(t[0]), reverse=True):
    OUT_FILE.write('{} {}\n'.format(score, sentence))
  OUT_FILE.write('\n')

  OUT_FILE.write('# matrix H transposed and sorted\n')
  for score, term in sorted(zip(H.T, terms), key=lambda t: sum(t[0]), reverse=True):
    OUT_FILE.writelines('{} {}\n'.format(score, term))
  OUT_FILE.write('\n')

  pass
