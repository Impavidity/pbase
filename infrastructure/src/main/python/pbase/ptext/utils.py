import logging
import pickle
import random


logging.basicConfig(level=logging.INFO)


class RandomShuffler(object):

  def __call__(self, data):
    return random.sample(data, len(data))


def flatten(item):
  """Yield items from any nested iterable; see Reference."""
  if not isinstance(item, list):
    yield item
  else:
    for x in item:
      if isinstance(x, list):
        for sub_x in flatten(x):
          yield sub_x
      else:
        yield x


def build_dataset_vocab(dataset, target, vocab):
  for example in dataset:
    for token in flatten(getattr(example, target)):
      vocab[token] = vocab.get(token, 0) + 1


def build_vocab(
      attributes,
      train_dataset,
      valid_dataset,
      test_dataset):
  for attribute in attributes:
    logging.info("Before adding the vocab in attribute {}, vocab size is {}".format(attribute.target, len(attribute.field.vocab)))
    if not attribute.field.build_vocab:
      continue
    build_dataset_vocab(train_dataset, attribute.target, attribute.field.vocab)
    if attribute.include_valid:
      build_dataset_vocab(valid_dataset, attribute.target, attribute.field.vocab)
    if attribute.include_test:
      build_dataset_vocab(test_dataset, attribute.target, attribute.field.vocab)
    logging.info("Add vocab from attribute {}, vocab size: {}".format(attribute.target, len(attribute.field.vocab)))
  for attribute in attributes:
    if attribute.field.build_vocab and attribute.field.dump_path:
      with open(attribute.field.dump_path, "wb") as fout:
        pickle.dump(attribute.field.vocab, fout, protocol=2)