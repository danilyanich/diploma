from nltk.corpus import stopwords
from string import punctuation
from scipy.sparse import coo_matrix

import Stemmer as st
import re
import math as ma


stem = st.Stemmer('russian')
russian_stopwords = stopwords.words('russian')
flatten = lambda l: [item for sublist in l for item in sublist]


def __preprocess_text(text):
  words = re.split('\W+', text.lower())
  stemmed = stem.stemWords(words)

  tokens = [token for token in stemmed
    if token not in russian_stopwords
      and token != " "
      and token.strip() not in punctuation]

  return tokens


def __get_frequency_dictionary(text):
  tokens = __preprocess_text(text)
  dictionary = dict.fromkeys(tokens, 0)

  for token in tokens:
    dictionary[token] += 1

  return dictionary


def get_weighted_term_document_matrix(documents):
  documents_dictionaries = [
    __get_frequency_dictionary(document)
    for document in documents
  ]

  all_terms = list(set(flatten([
    document.keys()
    for document in documents_dictionaries
  ])))

  documents_count = len(documents)
  terms_count = len(all_terms)
  shape = documents_count, terms_count

  frequency_table = [
    (all_terms.index(term), document_index, count)
    for document_index, dictionary in enumerate(documents_dictionaries)
    for term, count in dictionary.items()
  ]

  inverse_document_frequency = [
    sum(
      dictionary.get(term, 0)
      for dictionary in documents_dictionaries
    )
    for term in all_terms
  ]

  table = [
    (i, j, count * ma.log(documents_count / inverse_document_frequency[i]))
    for i, j, count in frequency_table
  ]

  data, i, j = zip(*table)

  return coo_matrix((data, (i, j)), shape).tocsr()
