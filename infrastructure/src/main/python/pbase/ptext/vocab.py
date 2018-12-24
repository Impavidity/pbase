from __future__ import division
import pickle

import six
import numpy as np
from tqdm import tqdm


def vocab_file_parser(filename):
  with open(filename, 'rb') as inp:
    vocab = pickle.load(inp)
  if type(vocab) != dict:
    TypeError("vocab object type {} is not supported".format(type(vocab)))
  return vocab, len(vocab)


def vocab_vector_file_parser(filename, lower):
  vocab_vector = {}
  with open(filename, 'rb') as handler:
    lines = [line for line in handler]
  print("Loading vectors from {}".format(filename))
  dim = None
  for line in tqdm(lines, total=len(lines)):
    entries = line.strip().split(b' ')
    word, entries = entries[0], entries[1:]
    if dim is None:
      if len(entries) == 1:
        dim = int(entries[0])
        continue
      dim = len(entries)
    try:
      if isinstance(word, six.binary_type):
        word = word.decode('utf-8')
    except UnicodeDecodeError:
      print('non-UTF8 token', repr(word), 'ignore')
      continue
    if lower:
      word = word.lower()
    vocab_vector[word] = np.array([float(x) for x in entries])
  return vocab_vector, len(vocab_vector), dim


class UniformInitializer(object):
  def __init__(self, low, high):
    self.low = low
    self.high = high

  def __call__(self, shape):
    return np.random.uniform(self.low, self.high, size=shape)


class Vocab(object):
  """Define a vocabulary object that will used for numericalize


  """

  def __init__(self,
               vocab_dict=None,
               vocab_file_path=None,
               vocab_file_parser=vocab_file_parser,
               min_freq=1,
               padding_token=None,
               unk_token=None,
               specials=None,
               use_embedding=False,
               embed_dim=None,
               vocab_vector_file_path=None,
               vocab_vector_token_lower=False,
               unk_init=UniformInitializer(low=-0.001, high=0.001),
               export_path=None):
    """

    :param vocab_file_path:
    :param min_freq:
    :param specials:
    :param vocab_vector_file_path:
    :param unk_init:
    """
    if vocab_dict is None:
      if vocab_file_path is not None:
        self.vocab_dict, vocab_list_size = vocab_file_parser(vocab_file_path)
      else:
        raise TypeError("`vocab_path` and `vocab_list` are both None")
    else:
      self.vocab_dict = vocab_dict
    min_freq = max(min_freq, 1)
    self.itos = []
    self.padding_token_idx = None
    self.unk_token_idx = None
    if padding_token:
      self.padding_token = padding_token
      self.itos.append(padding_token)
      self.padding_token_idx = self.itos.index(padding_token)
    if unk_token:
      self.unk_token = unk_token
      self.itos.append(unk_token)
      self.unk_token_idx = self.itos.index(unk_token)
    if specials:
      self.itos.extend(specials)
    words_and_frequencies = sorted(
        self.vocab_dict.items(), key=lambda tup: tup[0])
    words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)
    for word, freq in words_and_frequencies:
      if freq < min_freq:
        break
      self.itos.append(word)
    self.vocab_size = len(self.itos)
    self.stoi = {word: i for i, word in enumerate(self.itos)}
    if use_embedding:
      if embed_dim:
        self.embed_dim = embed_dim
      else:
        raise TypeError("embed_dim is None when use_embedding is True")
      self.vocab_vector = unk_init((self.vocab_size, self.embed_dim))
      if vocab_vector_file_path:
        self.vocab_vector_token_lower = vocab_vector_token_lower
        self.load_vectors(vocab_vector_file_path=vocab_vector_file_path)
    if export_path:
      with open(export_path, "w") as fout:
        for tid, token in enumerate(self.itos):
          fout.write("{} {}\n".format(tid, token.encode('utf-8')))
      print("Export Vocab to {}".format(export_path))

  def load_vectors(self,
                   vocab_vector_file_path,
                   vocab_vector_file_parser=vocab_vector_file_parser):
    full_vocab_vector, vocab_vector_size, dim = vocab_vector_file_parser(
        vocab_vector_file_path, self.vocab_vector_token_lower)
    hit_count = 0
    for idx, token in enumerate(self.itos):
      if token in full_vocab_vector:
        self.vocab_vector[idx, :] = full_vocab_vector[token]
        hit_count += 1
    del full_vocab_vector
    print("Total vocab size {}, hit {} of them. Coverage: {}".format(
        self.vocab_size, hit_count, hit_count / self.vocab_size))

  def batch_to_index(self, batch):
    batch = np.array(batch)
    shape = batch.shape
    index_array = np.array(
        [self.string_to_index(token) for token in batch.reshape(-1)])
    return index_array.reshape(shape)

  def string_to_index(self, token):
    index = self.stoi.get(token, self.unk_token_idx)
    if index is None:
      raise ValueError(
          "unk_token is not defined, token {} is not in {}".format(
              token, self.stoi))
    else:
      return index

  def index_to_string(self, index):
    return self.itos[index]

  def batch_to_string(self, batch):
    if type(batch) == list:
      batch = np.array(batch)
    elif type(batch) != np.ndarray:
      raise TypeError("batch type must be numpy.ndarray or list, "
                      "{} is not supported".format(type(batch)))

    shape = batch.shape
    string_array = np.array(
        [self.index_to_string(index) for index in batch.reshape(-1)])
    return string_array.reshape(shape)
