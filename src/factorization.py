import numpy as np
import pickle
import sys
import argparse

import methods as mt
import utils as ut


methods = {
  'MU': mt.multiplicative_update_rule,
  'ALS_solve': mt.alternating_least_squares_solve,
  'ALS_lstsq': mt.alternating_least_squares_lstsq,
  'ALS_nnls': mt.alternating_least_squares_nnls,
}


parser = argparse.ArgumentParser()

parser.add_argument('input', default=None)
parser.add_argument('--eps', default=10e-5, type=float)
parser.add_argument('--rank', default=1, type=int)
parser.add_argument('--method', default=list(methods.keys())[0], choices=methods.keys())
parser.add_argument('--out', default=None)


if __name__ == '__main__':
  args = parser.parse_args()

  in_file_path = args.input
  precision = args.eps
  decomposition_rank = args.rank
  method_name = args.method
  out_file_path = args.out

  in_file = open(in_file_path, mode='rb+') if in_file_path else sys.stdin.buffer
  out_file = open(out_file_path, mode='wb+') if out_file_path else sys.stdout.buffer

  data = pickle.load(in_file)

  matrix = data.get('matrix')
  method_generator = methods[method_name]

  W, H = ut.initialize_random(*matrix.shape, decomposition_rank)


  result, errors_trace = ut.iterate(
    method_generator,
    matrix, W, H,
    precision=precision,
    method_name=method_name,
  )

  W, H = result


  pickle.dump({
    'method_name': method_name,
    'precision': precision,
    'decomposition_rank': decomposition_rank,
    'errors_trace': errors_trace,
    'W': W,
    'H': H,
  }, out_file)

  pass
