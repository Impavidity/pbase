from pbase.pnn.tensorflow.embedding import Embedding
from pbase.pnn.tensorflow.embedding import vocab_file_parser, vocab_vector_file_parser

import tensorflow as tf
import pytest


VOCAB_FILE_PATH = "infrastructure/src/test/python/pbase/pnn/resource/vocab.pkl"
VOCAB_SIZE = 8


@pytest.mark.parametrize("filename, vocab_size",
                         [(VOCAB_FILE_PATH, VOCAB_SIZE)])
def test_vocab_file_parser(filename, vocab_size):
  vocab_list, size = vocab_file_parser(filename=filename)
  assert(size == vocab_size)


VECTOR_FILE_PATH = "infrastructure/src/test/python/pbase/pnn/resource/vector.txt"
VOCAB_SIZE = 10
VECTOR_DIM = 300


@pytest.mark.parametrize("filename, vocab_size, vector_dim",
                         [(VECTOR_FILE_PATH, VOCAB_SIZE, VECTOR_DIM)])
def test_vocab_vector_file_parser(filename, vocab_size, vector_dim):
  vocab_vector, size, dim = vocab_vector_file_parser(filename=filename)
  assert(vocab_size == size)
  assert(vector_dim == dim)


def test_embedding():
  vocab_vector, _, dim = vocab_vector_file_parser(filename=VECTOR_FILE_PATH)
  vocab_list = list(vocab_vector.keys())
  vocab_list_append, _ = vocab_file_parser(filename=VOCAB_FILE_PATH)
  vocab_list += vocab_list_append

  with tf.Session() as sess:
    embedding = Embedding(embed_dim=dim,
                          placeholder_name="word_embedding_placeholder",
                          params_name="word_embedding_params",
                          variable_scope="embedding",
                          trainable=False,
                          load_from_pretrain=True,
                          padding_token="<pad>",
                          unk_token="<unk>",
                          vocab_list=vocab_list,
                          vocab_vector_path=VECTOR_FILE_PATH)
    embedding.init(sess=sess)
    batch = [["An", "apple", "<pad>", "<pad>"],
             ["leptocarydion", "nevoiyeh", "draw_lax", "korpaljo"]]
    batch_index = embedding.string_to_index(batch=batch)
    embed_tensor = embedding(batch_index)
    sess.run(embed_tensor)