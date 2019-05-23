import numpy as np
import re
import sys

import methods as mt
import utils as ut
import parser as ps
import extraction as ex


np.set_printoptions(
  precision=3,
  suppress=True,
  formatter={'float': '{: 0.3f}'.format}
)


TEXT_FILE = open('./samples/ai_article.txt', mode='r')

RESULT_FILE = open('./report/result.txt', mode='w')

RANK_K = 10
EPS = 10e-5


if __name__ == '__main__':
  sentences = ut.get_sentences(TEXT_FILE.read())
  A, terms = ps.get_weighted_term_document_matrix(sentences)

  W, H = ut.initialize_random(*A.shape, RANK_K)

  print('# rank k = {}'.format(RANK_K))
  print()

  result, _ = ut.iterate(
    mt.alternating_least_squares_lstsq(A, W, H),
    matrix=A,
    precision=EPS,
    method_name='Alternating least squares (lstsq)'
  )

  W, H = result


  qr_key_sent, qr_key_words = ex.get_qr_key_factors(W, H)

  RESULT_FILE.write('# Key words\n')
  for index in qr_key_words:
    RESULT_FILE.write('{} - {}'.format(W[index], terms[index]))
    RESULT_FILE.write('\n')

  RESULT_FILE.write('\n# Key sentences\n')
  for index in qr_key_sent:
    RESULT_FILE.write('{} - {}'.format(H.T[index], sentences[index]))
    RESULT_FILE.write('\n')

  pass
