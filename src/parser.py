from nltk.corpus import stopwords as sw
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import coo_matrix

import re
import math as ma
import utils as ut
import collections as cl


LANGUAGE = 'russian'

stemmer = SnowballStemmer(LANGUAGE)
stopwords = sw.words(LANGUAGE)


def __flatten(list_of_lists):
  '''Unwrap list of list to flat list'''

  flat = [
    item
    for sublist in list_of_lists
    for item in sublist
  ]

  return flat


def __preprocess_text(raw_text):
  '''Split text into stemmed tokens'''

  # Split text by non letters
  raw_words = re.split('\W+', raw_text.lower())

  # Filter stopwords and empty words and then stem them
  stemmed = [
    stemmer.stem(word)
    for word in raw_words
    if len(word)
    and word not in stopwords
  ]

  return stemmed


def __get_frequency_dictionary(text):
  '''Compute term-frequency dictionary from text'''

  tokens = __preprocess_text(text)

  # Transform array into dictionary of occurrences
  dictionary = cl.Counter(tokens)

  return dictionary


def get_weighted_term_document_matrix(documents):
  '''Compute weighted term-document matrix'''

  print('# Generating weighted term-document matrix')
  print()

  # Compute all temr-count distionaries
  documents_dictionaries = [
    __get_frequency_dictionary(document)
    for document in ut.get_list_progress(documents, 'Building term-count dictinaries')
  ]

  print()
  print()

  # Gather all temrs from all documents
  all_terms = list(set(__flatten([
    document.keys()
    for document in ut.get_list_progress(documents_dictionaries, 'Gathering all terms')
  ])))

  print()
  print()

  documents_count = len(documents)
  terms_count = len(all_terms)
  shape = terms_count, documents_count

  # Merge document dictionaries and all terms into coo_matrix format
  frequency_table = [
    (all_terms.index(term), document_index, count)
    for document_index, dictionary in enumerate(ut.get_list_progress(documents_dictionaries, 'Merging frequency table'))
    for term, count in dictionary.items()
  ]

  print()
  print()

  # For each term count the number of documents in which it is contained
  inverse_document_frequency = [
    sum(
      dictionary.get(term, 0)
      for dictionary in documents_dictionaries
    )
    for term in ut.get_list_progress(all_terms, 'Computing document-term frequency')
  ]

  print()
  print()

  # Apply weights to generated table
  table = [
    (i, j, count * ma.log(documents_count / inverse_document_frequency[i]))
    for i, j, count in ut.get_list_progress(frequency_table, 'Generating weighted term-document matrix')
  ]

  print()
  print()

  # Pack all data into coo_matrix format
  i, j, data = zip(*table)
  matrix = coo_matrix((data, (i, j)), shape)

  return matrix, all_terms
