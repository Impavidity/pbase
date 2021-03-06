import six

from pbase.ptext.pipeline import Pipeline

import numpy as np


class Field(object):
  """Text processing for different data type

  Attributes:
    sequential: Boolean. True if this data type is sequential. Default True
    init_token: String or None. A token that will be prepended to every
      example, or None for no initial token. Default: None
    eos_token: String or None. A token that will be appended to every example,
      or None for no end-of-sentence token. Default: None
    preprocessing: Pipeline. A Pipeline that will be applied to examples
      before tokenizing. Default: None
    postprocessing: Pipeline. A Pipeline that will be applied to examples
      after tokenizing. Default: None
    lower: Boolean. True if lowercase the text in this field. Default False.
    tokenize: Function. The function used to tokenize strings into sequential
      examples. Default: str.split
    include_lengths: Boolean. True if a tuple of a padded batch and a list
      containing the lengths of each examples will be returned. False then
      return a padded batch. Default: False
    pad_token: String. The string token used as padding. Default: "<pad>"
    fix_length: Int. The use case here is to support nested field. You need
      to specify the max length in a tensor, instead getting the max length
      from one dimension
    build_vocab: Boolean. True if the vocab dictionary is built for this Field.
      Default: True
    dump_path: String or None. The path to dump vocab
  """

  def __init__(self,
               sequential=True,
               init_token=None,
               eos_token=None,
               preprocessing=None,
               postprocessing=None,
               lower=False,
               tokenize=(lambda s: s.split()),
               include_lengths=False,
               pad_token="<pad>",
               fix_length=None,
               build_vocab=True,
               dump_path=None):
    self.sequential = sequential
    self.init_token = init_token
    self.eos_token = eos_token
    self.preprocessing = preprocessing
    self.postprocessing = postprocessing
    self.lower = lower
    self.tokenize = tokenize
    self.include_lengths = include_lengths
    self.pad_token = pad_token if self.sequential else None
    self.fix_length = fix_length
    self.build_vocab = build_vocab
    self.dump_path = dump_path
    self.vocab = {}

  def preprocess(self, ex):
    """Load a single example using this field, tokenizing if necessary.

    If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline.

    Args:
      ex: a single example

    Returns: processed example

    """
    text_snippet = get_text_snippet(ex)
    if (six.PY2 and isinstance(text_snippet, six.string_types)
        and not isinstance(text_snippet, six.text_type)):
      ex = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(ex)
    if self.preprocessing is not None:
      ex = self.preprocessing(ex)
    if self.sequential and isinstance(ex, six.text_type):
      ex = self.tokenize(ex.rstrip('\n'))
    if self.lower:
      ex = Pipeline(six.text_type.lower)(ex)
    if self.postprocessing is not None:
      return self.postprocessing(ex)
    else:
      return ex

  def pad(self, batch):
    batch = list(batch)
    if not self.sequential:
      return batch
    if self.fix_length is None:
      max_len = max(len(x) for x in batch)
    else:
      max_len = self.fix_length + (self.init_token,
                                   self.eos_token).count(None) - 2

    padded, lengths = [], []
    for ex in batch:
      padded.append(([] if self.init_token is None else [self.init_token]) +
                    list(ex[:max_len]) +
                    ([] if self.eos_token is None else [self.eos_token]) +
                    [self.pad_token] * max(0, max_len - len(ex)))
      lengths.append(len(padded[-1]) - max(0, max_len - len(ex)))
    if self.include_lengths:
      return (padded, lengths)
    return padded

  def process(self, batch):
    """Process a list of examples to create batch

    preprocess and then pad it

    Args:
      batch: A list of object

    Returns:

    """
    batch = [self.preprocess(ex) for ex in batch]
    padded = self.pad(batch)
    return padded


class NestedField(Field):
  """A nested field.

  Use case: User want to represent a document under character level, then
    the an batch should in the shape of [batch_size, document_length, sentence_length, word_length]
    This nested filed supports build high dimension input with padding.
  """

  def __init__(self,
               nesting_field,
               init_token=None,
               eos_token=None,
               preprocessing=None,
               postprocessing=None,
               tokenize=lambda s: s.split(),
               include_lengths=False,
               pad_token='<pad>',
               fix_length=None,
               build_vocab=True,
               dump_path=None,
               use_parent_vocab=False):
    if nesting_field.sequential:
      pad_token = nesting_field.pad_token
    super(NestedField, self).__init__(
        init_token=init_token,
        eos_token=eos_token,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        tokenize=tokenize,
        include_lengths=include_lengths,
        pad_token=pad_token,
        fix_length=fix_length,
        build_vocab=build_vocab,
        dump_path=dump_path)
    self.nesting_field = nesting_field
    if use_parent_vocab:
      self.vocab = self.nesting_field.vocab
    assert self.nesting_field.include_lengths == self.include_lengths

  def preprocess(self, ex):
    return [
        self.nesting_field.preprocess(x)
        for x in super(NestedField, self).preprocess(ex)
    ]

  def pad(self, batch):
    batch = list(batch)
    if not self.nesting_field.sequential:
      return super(NestedField, self).pad(batch)

    # Save values of attributes to be monkeypatched
    old_pad_token = self.pad_token
    old_init_token = self.init_token
    old_eos_token = self.eos_token
    old_fix_len = self.nesting_field.fix_length

    if self.nesting_field.fix_length is None:
      max_len = max(len(xs) for ex in batch for xs in ex)
      fix_len = max_len + 2 - (self.nesting_field.init_token,
                               self.nesting_field.eos_token).count(None)
      self.nesting_field.fix_length = fix_len

    if self.init_token is not None:
      if self.include_lengths:
        self.init_token = self.nesting_field.pad([[self.init_token]])[0][0]
      else:
        self.init_token = self.nesting_field.pad([[self.init_token]])[0]
    if self.eos_token is not None:
      if self.include_lengths:
        self.eos_token = self.nesting_field.pad([[self.eos_token]])[0][0]
      else:
        self.eos_token = self.nesting_field.pad([[self.eos_token]])[0]
    # Do padding
    if self.include_lengths:
      padded = []
      lengths = []
      for ex in batch:
        pad_ex, len_ex = self.nesting_field.pad(ex)
        padded.append(pad_ex)
        lengths.append(len_ex)
      self.pad_token = [self.nesting_field.pad_token
                        ] * self.nesting_field.fix_length
      padded, high_level_lengths = super(NestedField, self).pad(padded)
      max_len_idx = np.argmax(high_level_lengths)
      max_len = (high_level_lengths[max_len_idx]
                 if self.fix_length is None else self.fix_length)
      pad_len_token = np.zeros_like(lengths[max_len_idx][0]).tolist()
      for i in range(len(lengths)):
        while len(lengths[i]) < max_len:
          lengths[i].append(pad_len_token)
    else:
      padded = [self.nesting_field.pad(ex) for ex in batch]
      self.pad_token = [self.nesting_field.pad_token
                        ] * self.nesting_field.fix_length
      padded = super(NestedField, self).pad(padded)

    # Restore monkeypatched attributes
    self.nesting_field.fix_length = old_fix_len
    self.pad_token = old_pad_token
    self.init_token = old_init_token
    self.eos_token = old_eos_token

    if self.include_lengths:
      return padded, lengths
    return padded


def get_text_snippet(ex):
  """Get an string element from input example. This element is used for
  encoding checking.

  Args:
    ex: Any object, expected to be recursive list/tuple, or string

  Returns: a string as an example

  """
  if isinstance(ex, list):
    if len(ex) > 0:
      return get_text_snippet(ex[0])
    else:
      return None
  elif isinstance(ex, (six.string_types, float, int)):
    return ex
  else:
    raise TypeError("Currently get_text_snippet function "
                    "does not support type {}".format(type(ex)))
