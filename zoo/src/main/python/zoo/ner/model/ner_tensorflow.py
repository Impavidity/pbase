from pbase.pnn.tensorflow.embedding import Embedding
from pbase.ptext.vocab import Vocab

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import crf


NER_INPUT_CHAR_PLACEHOLDER_NAME = "ner_input_char_placeholder"
NER_INPUT_WORD_PLACEHOLDER_NAME = "ner_input_word_placeholder"
NER_LABEL_PLACEHOLDER_NAME = "ner_label_placeholder"
WORD_EMBEDDING_PLACEHOLDER_NAME = "word_embedding_placeholder"

WORD_EMBEDDING_PARAMS_NAME = "word_embedding_params"
WORD_INPUT_LENGTH_NAME = "word_input_length"
LSTM_OUTPUT_CONCAT_NAME = "lstm_output_concat"
WEIGHT_NAME = "weight"
BIAS_NAME = "bias"
LOGITS_NAME = "logits"
TRANSITION_PARAMS_NAME = "transition_params"
CRF_CONFIDENCE_SCORE_NAME = "crf_confidence_score"
SCORE_DISTRIBUTION_NAME = "score_distribution"
SOFTMAX_CONFIDENCE_SCORE_NAME = "softmax_confidence_score"

EMBEDDING_SCOPE = "embedding"
SEQUENCE_LABELING_SCOPE = "sequence_labeling"
FULLY_CONNECTED_LAYER_SCOPE = "fully_connected_layer"


class NERModel(object):
  def __init__(self, config):
    # Vocab Definition
    self.word_vocab = Vocab(
      vocab_file_path=config.word_vocab_path,
      padding_token="<pad>",
      unk_token="<unk>",
      use_embedding=True,
      embed_dim=config.word_embedding_dim,
      vocab_vector_file_path=config.word_vocab_vector_path,
      export_path=config.word_vocab_export_path)
    self.char_vocab = Vocab(
      vocab_file_path=config.char_vocab_path,
      padding_token="<pad>",
      unk_token="<unk>",
      use_embedding=True,
      embed_dim=config.char_embedding_dim,
      export_path=config.char_vocab_export_path)
    self.label_vocab = Vocab(
      vocab_file_path=config.label_vocab_path,
      padding_token="<pad>",
      export_path=config.label_vocab_export_path)
    config.label_size = self.label_vocab.vocab_size
    # Placeholder Definition
    self.char = tf.placeholder(
      dtype=tf.int32,
      shape=[None, None, None],
      name=NER_INPUT_CHAR_PLACEHOLDER_NAME)
    self.word = tf.placeholder(
      dtype=tf.int32,
      shape=[None, None],
      name=NER_INPUT_WORD_PLACEHOLDER_NAME)
    self.label = tf.placeholder(
      dtype=tf.int32,
      shape=[None, None],
      name=NER_LABEL_PLACEHOLDER_NAME)
    self.word_embedding = Embedding(
      vocab=self.word_vocab,
      embed_dim=config.word_embedding_dim,
      placeholder_name=WORD_EMBEDDING_PLACEHOLDER_NAME,
      params_name=WORD_EMBEDDING_PARAMS_NAME,
      variable_scope=EMBEDDING_SCOPE,
      trainable=True,
      load_from_pretrain=True)
    self.word_input = self.word_embedding(self.word)
    self.word_input_length = tf.reduce_sum(
      tf.cast(
        tf.not_equal(self.word, self.word_vocab.padding_token_idx),
        tf.int32),
      axis=1, name=WORD_INPUT_LENGTH_NAME)
    with tf.variable_scope(SEQUENCE_LABELING_SCOPE) as sequence_labeling:
      fw_cell = rnn.LSTMCell(num_units=config.lstm_num_units,
                             state_is_tuple=True)
      bw_cell = rnn.LSTMCell(num_units=config.lstm_num_units,
                             state_is_tuple=True)
      lstm_output, _ = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=fw_cell,
        cell_bw=bw_cell,
        inputs=self.word_input,
        sequence_length=self.word_input_length,
        dtype=tf.float32)

    self.lstm_output_concat = tf.concat(lstm_output, axis=2, name=LSTM_OUTPUT_CONCAT_NAME)
    max_sent_length = tf.shape(self.lstm_output_concat)[1]
    lstm_output_2d = tf.reshape(self.lstm_output_concat,
                                shape=(-1, config.lstm_num_units * 2))
    with tf.variable_scope(FULLY_CONNECTED_LAYER_SCOPE) as fully_connected_layer:
      self.fully_connected_layer_weight = tf.get_variable(name=WEIGHT_NAME,
                                                  shape=(2 * config.lstm_num_units, config.label_size))
      self.fully_connected_layer_bias = tf.get_variable(name=BIAS_NAME,
                                                        shape=(config.label_size))
      logits_2d = tf.matmul(lstm_output_2d, self.fully_connected_layer_weight) + self.fully_connected_layer_bias
      self.logits = tf.reshape(logits_2d,
                               shape=(-1, max_sent_length, config.label_size),
                               name=LOGITS_NAME)

    if config.loss_func == "crf":
      self.trans_params = tf.get_variable(name=TRANSITION_PARAMS_NAME,
                                          shape=(config.label_size, config.label_size))
      likelihood, _ = crf.crf_log_likelihood(inputs=self.logits,
                             tag_indices=self.label,
                             sequence_lengths=self.word_input_length,
                             transition_params=self.trans_params)
      self.loss = -tf.reduce_mean(likelihood)
      self.decode_sequence, self.score = crf.crf_decode(potentials=self.logits,
                                                        transition_params=self.trans_params,
                                                        sequence_length=self.word_input_length)
      self.norm = crf.crf_log_norm(inputs=self.logits,
                                   sequence_lengths=self.word_input_length,
                                   transition_params=self.trans_params)
      self.prob = tf.exp(self.score - self.norm, name=CRF_CONFIDENCE_SCORE_NAME)
    elif config.loss_func == "softmax":
      loss = tf.losses.sparse_softmax_cross_entropy(labels=self.label,
                                                    logits=self.logits)
      mask = tf.sequence_mask(self.word_input_length)
      loss = tf.boolean_mask(loss, mask)
      self.loss = tf.reduce_mean(loss)
      self.distribution = tf.nn.softmax(self.logits, dim=2, name="score_distribution")
      self.decode_sequence = tf.cast(tf.argmax(self.distribution, axis=2), tf.int32)
      self.prob = tf.exp(tf.reduce_mean(tf.reduce_max(self.distribution, axis=2), axis=1),
                         name=SOFTMAX_CONFIDENCE_SCORE_NAME)

    if config.optim == "adam":
      optim = tf.train.AdamOptimizer(config.lr)
    elif config.optim == "adagrad":
      optim = tf.train.AdagradOptimizer(config.lr)
    elif config.optim == "sgd":
      optim = tf.train.GradientDescentOptimizer(config.lr)
    elif config.optim == "rmsprop":
      optim = tf.train.RMSPropOptimizer(config.lr)
    else:
      raise NotImplementedError("Unknown optimization method {}".format(config.optim))

    if config.clip > 0:
      tvars = tf.trainable_variables()
      grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm=config.clip)
      self.train_op = optim.apply_gradients(zip(grads, tvars))
    else:
      self.train_op = optim.minimize(self.loss)

  def train(self, batch, sess):
    word_batch_index = self.word_vocab.batch_to_index(batch.word[0])
    char_batch_index = self.char_vocab.batch_to_index(batch.char[0])
    label_batch_index = self.label_vocab.batch_to_index(batch.label[0])
    feed_dict = {
      self.word: word_batch_index,
      self.char: char_batch_index,
      self.label: label_batch_index
    }
    sess.run(self.train_op, feed_dict=feed_dict)


  def inference(self, batch, sess):
    word_batch_index = self.word_vocab.batch_to_index(batch.word[0])
    char_batch_index = self.char_vocab.batch_to_index(batch.char[0])
    feed_dict = {
      self.word: word_batch_index,
      self.char: char_batch_index
    }
    sequences = sess.run(self.decode_sequence, feed_dict=feed_dict)
    label_string = self.label_vocab.batch_to_string(sequences)
    return label_string