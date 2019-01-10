from __future__ import division

import tensorflow as tf


class Embedding(object):
  def __init__(self,
               vocab,
               embed_dim,
               placeholder_name,
               params_name,
               variable_scope,
               trainable=True,
               initializer=None,
               load_from_pretrain=False):
    self.initialized = False
    self.load_from_pretrain = load_from_pretrain
    self.vocab = vocab
    self.vocab_size = vocab.vocab_size
    self.embed_dim = embed_dim
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

  def index_to_embedding(self):
    return self.embedding_params

  def init(self, sess):
    if self.load_from_pretrain:
      sess.run(self.embedding_init,
               feed_dict={self.embedding_placeholder: self.vocab.vocab_vector})
    self.initialized = True
    del self.vocab.vocab_vector

  def __call__(self, ids):
    return tf.nn.embedding_lookup(params=self.index_to_embedding(),
                                  ids=ids)
