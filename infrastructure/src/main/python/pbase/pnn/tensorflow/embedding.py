import pickle

import six
import numpy as np
import tensorflow as tf
from tqdm import tqdm


def vocab_file_parser(filename):
  with open(filename, 'rb') as inp:
    vocab_list = pickle.load(inp)
  return vocab_list, len(vocab_list)


def vocab_vector_file_parser(filename):
  vocab_vector = {}
  with open(filename, 'rb') as handler:
    lines = [line for line in handler]
  print("Loading vectors from {}".format(filename))
  dim = None
  for line in tqdm(lines, total=len(lines)):
    entries = line.strip().split(b' ')
    word, entries = entries[0], entries[1:]
    if dim is None:
      dim = len(entries)
    try:
      if isinstance(word, six.binary_type):
        word = word.decode('utf-8')
    except UnicodeDecodeError:
      print('non-UTF8 token', repr(word), 'ignore')
      continue
    vocab_vector[word] = np.array([float(x) for x in entries])
  return vocab_vector, len(vocab_vector), dim


class Embedding(object):
  def __init__(self,
               embed_dim,
               placeholder_name,
               params_name,
               variable_scope,
               trainable=True,
               initializer=None,
               load_from_pretrain=False,
               padding_token=None,
               unk_token=None,
               vocab_path=None,
               vocab_file_parser=vocab_file_parser,
               vocab_vector_path=None,
               vocab_vector_file_parser=vocab_vector_file_parser,
               vocab_list=None,
               vocab_vector_dict=None):
    if vocab_list is None:
      if vocab_path is not None:
        self.vocab_list, vocab_list_size = vocab_file_parser(vocab_path)
      else:
        raise TypeError("`vocab_path` and `vocab_list` are both None")
    else:
      self.vocab_list = vocab_list
    if unk_token is not None:
      self.vocab_list = [unk_token] + self.vocab_list
    if padding_token is not None:
      self.vocab_list = [padding_token] + self.vocab_list
    self.vocab_size = len(self.vocab_list)
    self.stoi = {word: i for i, word in enumerate(self.vocab_list)}
    self.embed_dim = embed_dim
    self.load_from_pretrain = load_from_pretrain
    full_vocab_vector = {}
    if self.load_from_pretrain:
      if vocab_vector_dict is None:
        if vocab_vector_path is not None:
          full_vocab_vector, vocab_vector_size, dim = vocab_vector_file_parser(
            vocab_vector_path)
          assert(dim == self.embed_dim)
        else:
          raise TypeError("`vocab_vector_dict` and `vocab_dict` are both None")
      else:
        full_vocab_vector = vocab_vector_dict
    self.vocab_vector = np.random.uniform(-0.05, 0.05,
                                          size=(self.vocab_size, self.embed_dim))
    for idx, token in enumerate(self.vocab_list):
      if token in full_vocab_vector:
        self.vocab_vector[idx, :] = full_vocab_vector[token]
    del full_vocab_vector
    self.initialized = False
    with tf.variable_scope(variable_scope):
      self.embedding_params = tf.get_variable(name=params_name,
                                              shape=(self.vocab_size, self.embed_dim),
                                              dtype=tf.float32,
                                              trainable=trainable,
                                              initializer=initializer)
    self.embedding_placeholder = tf.placeholder(
      dtype=tf.float32,
      shape=(self.vocab_size, self.embed_dim),
      name=placeholder_name)
    self.embedding_init = self.embedding_params.assign(self.embedding_placeholder)

  def string_to_index(self, batch):
    batch = np.array(batch)
    shape = batch.shape
    index_array = np.array([self.stoi[token] for token in batch.reshape(-1)])
    return index_array.reshape(shape)

  def index_to_embedding(self):
    if self.initialized:
      return self.embedding_params
    else:
      raise Exception("Need to initialize the parameters")

  def init(self, sess):
    if self.load_from_pretrain:
      sess.run(self.embedding_init,
               feed_dict={self.embedding_placeholder: self.vocab_vector})
    self.initialized = True
    del self.vocab_vector

  def __call__(self, ids):
    return tf.nn.embedding_lookup(params=self.index_to_embedding(),
                                  ids=ids)
