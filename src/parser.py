from nltk.corpus import stopwords as sw
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import coo_matrix

import re
import math as ma
import collections as cl
import argparse
import pickle
import sys
import utils as ut


def __flatten(list_of_lists):
  '''Unwrap list of list to flat list'''

  flat = [
    item
    for sublist in list_of_lists
    for item in sublist
  ]

  return flat


def tokenize_text(raw_text, preprocessing_info):
  '''Split text into stemmed tokens'''

  stemmer, stopwords = preprocessing_info

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


def get_frequency_dictionary(tokens):
  '''Compute term-frequency dictionary from text'''

  # Transform array into dictionary of occurrences
  dictionary = cl.Counter(tokens)

  return dictionary


def get_sentences(raw_text):
  '''Split text into sentences'''

  # Replace spaces, quotes and newlines
  text = re.sub('[\n" ]+', ' ', raw_text)

  # Split text by punctuation and remove leading whitespaces
  raw_sentences = [
    re.sub('^ +', '', s)
    for s in re.split('[!.?]+ +', text)
  ]

  # Filter empty sentences
  sentences = [s for s in raw_sentences if len(s)]

  return sentences


def get_weighted_term_document_matrix(documents_dictionaries):
  '''Compute weighted term-document matrix'''

  # Gather all temrs from all documents
  all_terms = list(set(__flatten([
    document.keys()
    for document in documents_dictionaries
  ])))

  documents_count = len(documents_dictionaries)
  terms_count = len(all_terms)
  shape = terms_count, documents_count

  # Merge document dictionaries and all terms into coo_matrix format
  frequency_table = [
    (all_terms.index(term), document_index, count)
    for document_index, dictionary in enumerate(documents_dictionaries)
    for term, count in dictionary.items()
  ]

  # For each term count the number of documents in which it is contained
  inverse_document_frequency = [
    sum(
      dictionary.get(term, 0)
      for dictionary in documents_dictionaries
    )
    for term in all_terms
  ]

  # Apply weights to generated table
  table = [
    (i, j, count * ma.log(documents_count / inverse_document_frequency[i]))
    for i, j, count in frequency_table
  ]

  # Pack all data into coo_matrix format
  i, j, data = zip(*table)
  matrix = coo_matrix((data, (i, j)), shape)

  return matrix, all_terms


parser = argparse.ArgumentParser()

parser.add_argument('input')
parser.add_argument('--lang', default='russian')
parser.add_argument('--out', default=None)


if __name__ == '__main__':
  args = parser.parse_args()

  in_file_path = args.input
  text_language = args.lang
  out_file_path = args.out if args.out else ut.replace_ext(in_file_path, 'parsed')

  in_file = open(in_file_path, mode='r+')
  out_file = open(out_file_path, mode='wb+')

  preprocessing_info = (SnowballStemmer(text_language), sw.words(text_language) )


  sentences = get_sentences(in_file.read())
  tokenized = [tokenize_text(sentence, preprocessing_info) for sentence in sentences]
  frequency_dictionaries = [get_frequency_dictionary(tokens) for tokens in tokenized]
  A, terms = get_weighted_term_document_matrix(frequency_dictionaries)


  pickle.dump({
    'matrix': A,
    'terms': terms,
    'sentences': sentences,
  }, out_file)

  pass
