import scipy.linalg as lg
import numpy as np


def get_key_sentence_numbers(W, H):
  _, rank = W.shape
  _, _, permutations = lg.qr(H, pivoting=True)

  return permutations[:rank]


def get_key_word_numbers(W, H):
  _, rank = W.shape
  _, _, permutations = lg.qr(W.T, pivoting=True)

  return permutations[:rank]
