import scipy.linalg as lg
import numpy as np


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
