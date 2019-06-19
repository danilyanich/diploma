import scipy.linalg as lg
import numpy as np
import utils as ut
import argparse
import pickle


np.set_printoptions(
  formatter={'float': '{: 0.3f}'.format}
)


def get_saliency_key_factors(W, H):
  '''Extract key factors based on saliency score'''

  n_terms, _ = W.shape
  _, n_sentences = H.shape

  _, column_permutations = zip(*sorted(
    zip(H.T, range(n_sentences)),
    key=lambda t: -sum(t[0])
  ))

  _, row_permutations = zip(*sorted(
    zip(W, range(n_terms)),
    key=lambda t: -sum(t[0])
  ))

  return column_permutations, row_permutations


def get_qr_key_factors(W, H):
  '''Extract key factors based on QR decomposition with column pivoting'''

  _, _, column_permutations = lg.qr(H, pivoting=True)
  _, _, row_permutations = lg.qr(W.T, pivoting=True)

  return column_permutations, row_permutations


methods = {
  'saliency': get_saliency_key_factors,
  'qr': get_qr_key_factors,
}


parser = argparse.ArgumentParser()

parser.add_argument('parsed')
parser.add_argument('factorized')
parser.add_argument('--method', default=list(methods.keys())[0], choices=methods.keys())
parser.add_argument('--out', default=None)


if __name__ == '__main__':
  args = parser.parse_args()

  parsed_file_path = args.parsed
  factorized_file_path = args.factorized
  method_name = args.method
  out_file_path = args.out if args.out else ut.replace_ext(parsed_file_path, '{}.extracted'.format(method_name))

  parsed_file = open(parsed_file_path, mode='rb+')
  factorized_file = open(factorized_file_path, mode='rb+')
  out_file = open(out_file_path, mode='w+')

  parsed_data = pickle.load(parsed_file)
  factorized_data = pickle.load(factorized_file)

  W = factorized_data.get('W')
  H = factorized_data.get('H')

  terms = parsed_data.get('terms')
  sentences = parsed_data.get('sentences')


  method_func = methods[method_name]
  key_sent, key_words = method_func(W, H)


  out_file.write('\n# Key sentences\n')
  for index in key_sent:
    out_file.write('{} - {}'.format(H.T[index], sentences[index]))
    out_file.write('\n')
